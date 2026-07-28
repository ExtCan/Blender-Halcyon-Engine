"""AST -> vectorised Python.

Divergent control flow is handled the way a GPU does it: with execution masks.
`if` narrows the mask for each branch, assignments inside a masked region become
select-and-write-back, loops iterate until no lane is still active, and `return`
/ `break` / `continue` / `discard` accumulate their own masks. That is why a
shader with per-pixel branching produces the same image it would on hardware
rather than something that only happens to work when every pixel agrees.
"""

from .builtins import BUILTINS
from .gtypes import (BOOL, BY_NAME, FLOAT, INT, VOID, GType, promote,
                     swizzle_indices)
from .lexer import ShaderError
from .parser import parse

MAX_LOOP = 1024

# Names a shader can read. Anything else declared `in`/`varying` gets zeros.
VARYINGS = {
    'gl_FragCoord': ('fragcoord', GType('float', 4)),
    'vPosition': ('position', GType('float', 3)),
    'P': ('position', GType('float', 3)),
    'worldPos': ('position', GType('float', 3)),
    'vNormal': ('normal', GType('float', 3)),
    'N': ('normal', GType('float', 3)),
    'normal': ('normal', GType('float', 3)),
    'vGeoNormal': ('geonormal', GType('float', 3)),
    'vTangent': ('tangent', GType('float', 3)),
    'vBitangent': ('bitangent', GType('float', 3)),
    'vUV': ('uv', GType('float', 2)),
    'uv': ('uv', GType('float', 2)),
    'vUv': ('uv', GType('float', 2)),
    'texCoord': ('uv', GType('float', 2)),
    'vUV2': ('uv2', GType('float', 2)),
    'vColor': ('color', GType('float', 4)),
    'vView': ('view', GType('float', 3)),
    'V': ('view', GType('float', 3)),
    'I': ('incident', GType('float', 3)),
    'vObject': ('object', GType('float', 3)),
    'vScreenUV': ('screenuv', GType('float', 2)),
    'vDepth': ('depth', FLOAT),
    'vBackfacing': ('backfacing', FLOAT),
    'vCamera': ('camera', GType('float', 3)),
    'iTime': ('time', FLOAT),
    'time': ('time', FLOAT),
    'iFrame': ('frame', INT),
    'iResolution': ('resolution', GType('float', 3)),
    'vRandom': ('random', FLOAT),
}

FRAG_OUTS = ('gl_FragColor', 'gl_FragData', 'fragColor', 'FragColor', 'outColor')


class Scope:
    def __init__(self, parent=None):
        self.vars = {}
        self.parent = parent

    def get(self, name):
        s = self
        while s:
            if name in s.vars:
                return s.vars[name]
            s = s.parent
        return None

    def put(self, name, py, gtype):
        self.vars[name] = (py, gtype)
        return py


class Generator:
    def __init__(self, structs, hlsl=False):
        self.structs = structs
        self.hlsl = hlsl
        self.lines = []
        self.indent = 1
        self.tmp = 0
        self.funcs = {}
        self.uniforms = []          # (name, gtype, default_ast)
        self.outs = []              # (name, gtype)
        self.inputs = []            # (name, gtype, binding)
        self.consts = []
        self.warnings = []
        self.loops = []

    # ----------------------------------------------------------- emit helper
    def emit(self, code):
        self.lines.append('    ' * self.indent + code)

    def new_tmp(self, prefix='_t'):
        self.tmp += 1
        return f'{prefix}{self.tmp}'

    def zero(self, t):
        if t.array:
            return '[' + ', '.join([self.zero(t.elem())] * max(t.array, 1)) + ']'
        if t.base == 'struct':
            fields = self.structs.get(t.struct, [])
            return '{' + ', '.join(f'{f[1]!r}: {self.zero(f[0])}' for f in fields) + '}'
        if t.base == 'sampler':
            return 'None'
        dt = {'float': 'np.float32', 'int': 'np.int32', 'uint': 'np.int32',
              'bool': 'np.bool_'}.get(t.base, 'np.float32')
        if t.is_matrix:
            return f'np.zeros((1,{t.n},{t.rows}), np.float32)'
        if t.n > 1:
            return f'np.zeros((1,{t.n}), {dt})'
        return f'np.zeros(1, {dt})'

    # -------------------------------------------------------------- entry
    def generate(self, decls):
        for d in decls:
            if d and d[0] == 'block':
                for sub in d[1]:
                    self.collect(sub)
            elif d:
                self.collect(d)

        body = []
        self.lines = body
        self.indent = 1
        gscope = Scope()

        # bind uniforms -- an unconnected socket falls back to the declared
        # initialiser, so a shader behaves the same in Blender as it reads
        for name, gtype, default in self.uniforms:
            py = 'u_' + _san(name)
            fallback = self.zero(gtype)
            if default is not None:
                try:
                    code, t = self.expr(default, 'True', gscope)
                    fallback = self.coerce(code, t, gtype)
                except ShaderError:
                    pass
            gscope.vars[name] = (py, gtype)
            self.emit(f'{py} = __uni.get({name!r}, None)')
            self.emit(f'if {py} is None: {py} = {fallback}')
        # bind varyings / inputs
        for name, gtype, binding in self.inputs:
            py = 'i_' + _san(name)
            gscope.vars[name] = (py, gtype)
            self.emit(f'{py} = __in.get({binding!r}, {self.zero(gtype)})')
        # declare outputs
        for name, gtype in self.outs:
            py = 'o_' + _san(name)
            gscope.vars[name] = (py, gtype)
            self.emit(f'{py} = __bcast({self.zero(gtype)}, __n)')
        # module-level constants
        for name, gtype, init in self.consts:
            py = 'c_' + _san(name)
            if init is not None:
                code, t = self.expr(init, 'True', gscope)
                code = self.coerce(code, t, gtype)
            else:
                code = self.zero(gtype)
            gscope.vars[name] = (py, gtype)
            self.emit(f'{py} = {code}')

        entry = self.pick_entry()
        if entry is None:
            raise ShaderError('no entry point: define main() (or PSMain/mainImage)')

        # user functions are emitted as nested defs
        fn_lines = []
        for fname, fdef in self.funcs.items():
            if fdef is entry:
                continue
            fn_lines.extend(self.gen_function(fdef, gscope))

        self.lines = body
        self.indent = 1
        self.emit('__discard = False')

        rt_, name, params, fbody = entry[1], entry[2], entry[3], entry[4]
        callscope = Scope(gscope)
        args = []
        for qual, ptype, pname, parr in params:
            py = 'p_' + _san(pname)
            binding = self.bind_param(pname, ptype)
            callscope.vars[pname] = (py, ptype)
            if qual in ('out', 'inout'):
                self.emit(f'{py} = __bcast({self.zero(ptype)}, __n)')
                self.outs.append((pname, ptype))
                gscope.vars[pname] = (py, ptype)
            else:
                self.emit(f'{py} = __in.get({binding!r}, {self.zero(ptype)})')
        ret_t = rt_
        if ret_t.base != 'void':
            self.emit(f'__ret = __bcast({self.zero(ret_t)}, __n)')
            self.ret_type = ret_t
        else:
            self.ret_type = None
        self.emit('__retm = False')
        self.in_entry = True
        self.block(fbody, '__mask', callscope, ctxflags={'ret': ret_t.base != 'void'})

        outs = []
        for name_, t in self.outs:
            py = 'o_' + _san(name_) if any(n == name_ for n, _ in self.outs) else None
            sc = gscope.vars.get(name_)
            if sc:
                outs.append((name_, sc[0], t))
        if ret_t.base != 'void':
            outs.append(('__return', '__ret', ret_t))
        self.emit('return {' + ', '.join(f'{n!r}: {p}' for n, p, _ in outs) +
                  '}, __discard')

        head = ['def __shader(rt, np, __uni, __in, __n):',
                '    __mask = True',
                '    __bcast = rt.bc']
        return '\n'.join(head + fn_lines + body), outs

    def bind_param(self, pname, ptype):
        v = VARYINGS.get(pname)
        if v:
            return v[0]
        low = pname.lower()
        for k, (b, t) in VARYINGS.items():
            if k.lower() == low:
                return b
        if ptype.n == 2 and 'coord' in low:
            return 'fragcoord2'
        return low

    def pick_entry(self):
        for cand in ('main', 'mainImage', 'PSMain', 'psmain', 'MainPS', 'frag',
                     'fragment', 'pixel', 'PS'):
            if cand in self.funcs:
                return self.funcs[cand]
        if self.funcs:
            return list(self.funcs.values())[-1]
        return None

    # ------------------------------------------------------------ collection
    def collect(self, d):
        kind = d[0]
        if kind == 'struct':
            self.structs[d[1]] = d[2]
        elif kind == 'func':
            self.funcs[d[2]] = d
        elif kind == 'proto':
            pass
        elif kind == 'global':
            quals, gtype, name, arr, init = d[1], d[2], d[3], d[4], d[5]
            if arr:
                gtype = GType(gtype.base, gtype.n, gtype.rows, arr, gtype.struct)
            if 'uniform' in quals or gtype.base == 'sampler':
                self.uniforms.append((name, gtype, init))
            elif 'out' in quals:
                self.outs.append((name, gtype))
            elif 'in' in quals or 'varying' in quals or 'attribute' in quals:
                v = VARYINGS.get(name)
                binding = v[0] if v else self.bind_param(name, gtype)
                if v is None:
                    self.warnings.append(f'unknown input {name!r}: bound to zero')
                self.inputs.append((name, gtype, binding))
            else:
                self.consts.append((name, gtype, init))

    # ------------------------------------------------------------- functions
    def gen_function(self, fdef, gscope):
        _, ret_t, name, params, body = fdef
        saved, saved_i = self.lines, self.indent
        self.lines = []
        self.indent = 1
        sc = Scope(gscope)
        argnames = []
        outs = []
        for qual, ptype, pname, parr in params:
            py = 'a_' + _san(pname)
            sc.vars[pname] = (py, ptype)
            argnames.append(py)
            if qual in ('out', 'inout'):
                outs.append((py, ptype))
        if ret_t.base != 'void':
            self.emit(f'__ret = __bcast({self.zero(ret_t)}, __n)')
        self.emit('__retm = False')
        prev = getattr(self, 'ret_type', None)
        self.ret_type = ret_t if ret_t.base != 'void' else None
        self.block(body, '__mask', sc, ctxflags={'ret': ret_t.base != 'void'})
        rets = []
        if ret_t.base != 'void':
            rets.append('__ret')
        rets.extend(p for p, _ in outs)
        self.emit('return ' + (', '.join(rets) if rets else 'None'))
        lines = [f'    def f_{_san(name)}(__mask, ' + ', '.join(argnames or []) +
                 ('):' if argnames else '):')]
        lines = [f'    def f_{_san(name)}(' +
                 ', '.join(['__mask'] + argnames) + '):']
        lines += ['    ' + l for l in self.lines]
        self.lines, self.indent = saved, saved_i
        self.ret_type = prev
        self.func_outs = getattr(self, 'func_outs', {})
        self.func_outs[name] = (ret_t, [(i, q) for i, (q, _, _, _) in
                                        enumerate(params) if q in ('out', 'inout')])
        return lines

    # ------------------------------------------------------------ statements
    def block(self, node, mask, scope, ctxflags=None):
        ctxflags = ctxflags or {}
        if node[0] != 'block':
            return self.stmt(node, mask, scope, ctxflags)
        sc = Scope(scope)
        m = mask
        for s in node[1]:
            m = self.stmt(s, m, sc, ctxflags)
        return m

    def stmt(self, node, mask, scope, flags):
        k = node[0]
        if k == 'nop':
            return mask
        if k == 'block':
            return self.block(node, mask, scope, flags)
        if k == 'expr':
            code, _ = self.expr(node[1], mask, scope)
            self.emit(f'_ = {code}')
            return mask
        if k == 'decl':
            gtype = node[1]
            for name, arr, init in node[2]:
                t = GType(gtype.base, gtype.n, gtype.rows, arr, gtype.struct) \
                    if arr else gtype
                py = 'v_' + _san(name) + f'_{self.new_tmp("")[1:]}'
                if init is not None:
                    code, it = self.expr(init, mask, scope)
                    code = self.coerce(code, it, t)
                else:
                    code = f'__bcast({self.zero(t)}, __n)'
                self.emit(f'{py} = {code}')
                scope.put(name, py, t)
            return mask
        if k == 'if':
            cond, ct = self.expr(node[1], mask, scope)
            c = self.new_tmp('_c')
            self.emit(f'{c} = rt.to_bool({cond})')
            mt = self.new_tmp('_m')
            self.emit(f'{mt} = ({mask}) & {c}')
            self.emit(f'if rt.any_({mt}):')
            self.indent += 1
            self.block(node[2], mt, scope, flags)
            self.indent -= 1
            if node[3] is not None:
                me = self.new_tmp('_m')
                self.emit(f'{me} = ({mask}) & (~{c})')
                self.emit(f'if rt.any_({me}):')
                self.indent += 1
                self.block(node[3], me, scope, flags)
                self.indent -= 1
            return self.narrow(mask, flags)
        if k in ('for', 'while', 'do'):
            return self.loop(node, mask, scope, flags)
        if k == 'return':
            if node[1] is not None and getattr(self, 'ret_type', None) is not None:
                code, t = self.expr(node[1], mask, scope)
                code = self.coerce(code, t, self.ret_type)
                self.emit(f'__ret = rt.sel({mask}, {code}, __ret)')
            self.emit(f'__retm = __retm | ({mask})')
            nm = self.new_tmp('_m')
            self.emit(f'{nm} = ({mask}) & (~__retm)')
            return nm
        if k == 'break':
            if not self.loops:
                raise ShaderError('break outside of a loop')
            brk = self.loops[-1][0]
            self.emit(f'{brk} = {brk} | ({mask})')
            return self.narrow(mask, flags)
        if k == 'continue':
            if not self.loops:
                raise ShaderError('continue outside of a loop')
            cnt = self.loops[-1][1]
            self.emit(f'{cnt} = {cnt} | ({mask})')
            return self.narrow(mask, flags)
        if k == 'discard':
            self.emit(f'__discard = __discard | ({mask})')
            nm = self.new_tmp('_m')
            self.emit(f'{nm} = ({mask}) & (~({mask}))')
            return nm
        raise ShaderError(f'unsupported statement {k}')

    def narrow(self, mask, flags=None):
        """Remove lanes that have returned, broken or continued."""
        terms = ['__retm']
        if self.loops:
            terms.extend(self.loops[-1][:2])
        nm = self.new_tmp('_m')
        expr = ' & '.join(f'(~{t})' for t in terms)
        self.emit(f'{nm} = ({mask}) & {expr}')
        return nm

    def loop(self, node, mask, scope, flags):
        sc = Scope(scope)
        kind = node[0]
        step = None
        if kind == 'for':
            init, cond, step, body = node[1], node[2], node[3], node[4]
            if init and init[0] != 'nop':
                self.stmt(init, mask, sc, flags)
        elif kind == 'while':
            cond, body = node[1], node[2]
        else:
            cond, body = node[2], node[1]

        lid = self.new_tmp('')[1:]
        brk = f'__brk{lid}'
        cnt = f'__cnt{lid}'
        act = f'_act{lid}'
        it = f'_it{lid}'
        self.emit(f'{brk} = False')
        self.emit(f'{act} = {mask}')
        self.emit(f'{it} = 0')
        self.emit('while True:')
        self.indent += 1
        self.emit(f'{cnt} = False')
        if cond is not None and kind != 'do':
            ccode, _ = self.expr(cond, act, sc)
            self.emit(f'{act} = ({act}) & rt.to_bool({ccode})')
            self.emit(f'if not rt.any_({act}): break')
        self.loops.append((brk, cnt))
        self.block(body, act, sc, flags)
        self.loops.pop()
        self.emit(f'{act} = ({act}) & (~{brk})')
        if kind == 'do' and cond is not None:
            ccode, _ = self.expr(cond, act, sc)
            self.emit(f'{act} = ({act}) & rt.to_bool({ccode})')
        self.emit(f'if not rt.any_({act}): break')
        if step is not None:
            code, _ = self.expr(step, act, sc)
            self.emit(f'_ = {code}')
        self.emit(f'{it} += 1')
        self.emit(f'if {it} >= {MAX_LOOP}: break')
        self.indent -= 1
        return self.narrow(mask, flags)

    # ----------------------------------------------------------- expressions
    def expr(self, node, mask, scope):
        k = node[0]
        m = getattr(self, 'e_' + k, None)
        if m is None:
            raise ShaderError(f'unsupported expression {k}')
        return m(node, mask, scope)

    def e_num(self, node, mask, scope):
        txt, is_int = node[1], node[2]
        clean = txt.rstrip('uUfFhH')
        if is_int:
            val = int(clean, 0) if clean.lower().startswith('0x') else int(float(clean))
            return f'np.int32({val})', INT
        return f'np.float32({float(clean)})', FLOAT

    def e_bool(self, node, mask, scope):
        return f'np.bool_({bool(node[1])})', BOOL

    def e_var(self, node, mask, scope):
        name = node[1]
        v = scope.get(name)
        if v is None:
            if name in FRAG_OUTS:
                t = GType('float', 4)
                py = 'o_' + _san(name)
                self.outs.append((name, t))
                scope.put(name, py, t)
                self.emit(f'{py} = __bcast({self.zero(t)}, __n)')
                return py, t
            vb = VARYINGS.get(name)
            if vb:
                py = 'i_' + _san(name)
                self.inputs.append((name, vb[1], vb[0]))
                scope.put(name, py, vb[1])
                self.emit(f'{py} = __in.get({vb[0]!r}, {self.zero(vb[1])})')
                return py, vb[1]
            raise ShaderError(f'undeclared identifier {name!r}')
        return v[0], v[1]

    def e_seq(self, node, mask, scope):
        a, _ = self.expr(node[1], mask, scope)
        self.emit(f'_ = {a}')
        return self.expr(node[2], mask, scope)

    def e_member(self, node, mask, scope):
        obj, t = self.expr(node[1], mask, scope)
        field = node[2]
        if t.base == 'struct':
            fields = self.structs.get(t.struct, [])
            for ft, fname, farr in fields:
                if fname == field:
                    return f'({obj})[{field!r}]', ft
            raise ShaderError(f'{t.struct} has no field {field!r}')
        idx = swizzle_indices(field)
        if idx is None:
            raise ShaderError(f'bad swizzle {field!r}')
        if max(idx) >= t.n:
            raise ShaderError(f'swizzle {field!r} out of range for {t.name()}')
        rt_ = GType(t.base, len(idx))
        return f'rt.sw({obj}, {idx})', rt_

    def e_index(self, node, mask, scope):
        obj, t = self.expr(node[1], mask, scope)
        icode, it = self.expr(node[2], mask, scope)
        const = _const_int_expr(node[2])
        if t.array:
            if const is None:
                return f'rt.idx_list({obj}, {icode})', t.elem()
            return f'({obj})[{const}]', t.elem()
        if t.is_matrix:
            if const is None:
                return f'rt.idx_axis({obj}, {icode}, 1)', GType(t.base, t.rows)
            return f'({obj})[:, {const}]', GType(t.base, t.rows)
        if const is None:
            return f'rt.idx_axis({obj}, {icode}, 1)', t.scalar()
        return f'rt.sw({obj}, [{const}])', t.scalar()

    def e_un(self, node, mask, scope):
        op = node[1]
        if op in ('++', '--'):
            one = ('num', '1', True)
            return self.e_assign(('assign', '+=' if op == '++' else '-=',
                                  node[2], one), mask, scope)
        code, t = self.expr(node[2], mask, scope)
        if op == '-':
            return f'(-({code}))', t
        if op == '+':
            return code, t
        if op == '!':
            return f'rt.not_({code})', GType('bool', t.n)
        if op == '~':
            return f'(~rt.to_int({code}))', t
        raise ShaderError(f'bad unary {op}')

    def e_post(self, node, mask, scope):
        code, t = self.expr(node[2], mask, scope)
        tmp = self.new_tmp('_p')
        self.emit(f'{tmp} = {code}')
        one = ('num', '1', True)
        self.e_assign(('assign', '+=' if node[1] == '++' else '-=', node[2], one),
                      mask, scope)
        return tmp, t

    def e_cond(self, node, mask, scope):
        c, _ = self.expr(node[1], mask, scope)
        a, ta = self.expr(node[2], mask, scope)
        b, tb = self.expr(node[3], mask, scope)
        t = ta if ta.comps >= tb.comps else tb
        a = self.coerce(a, ta, t)
        b = self.coerce(b, tb, t)
        return f'rt.sel(rt.to_bool({c}), {a}, {b})', t

    def e_bin(self, node, mask, scope):
        op = node[1]
        a, ta = self.expr(node[2], mask, scope)
        b, tb = self.expr(node[3], mask, scope)
        if op in ('&&', '||', '^^'):
            pya = f'rt.to_bool({a})'
            pyb = f'rt.to_bool({b})'
            sym = {'&&': '&', '||': '|', '^^': '^'}[op]
            return f'(({pya}) {sym} ({pyb}))', BOOL
        if op in ('==', '!='):
            if ta.comps > 1 or tb.comps > 1:
                fn = 'rt.all_comp(rt.equal' if op == '==' else 'rt.any_comp(rt.notEqual'
                return f'{fn}({a}, {b}))', BOOL
            return f'(({a}) {op} ({b}))', BOOL
        if op in ('<', '>', '<=', '>='):
            return f'(({a}) {op} ({b}))', BOOL
        if op == '*':
            if ta.is_matrix and tb.is_matrix:
                return f'rt.mat_mul_mat({a}, {b})', ta
            if ta.is_matrix and tb.is_vector:
                return f'rt.mat_mul_vec({a}, {b})', GType('float', ta.rows)
            if ta.is_vector and tb.is_matrix:
                return f'rt.vec_mul_mat({a}, {b})', GType('float', tb.n)
            if ta.is_matrix or tb.is_matrix:
                return f'(({a}) * ({b}))', ta if ta.is_matrix else tb
            t = promote(ta, tb)
            return f'rt.mul_c({a}, {b})', t
        if op in ('+', '-', '/'):
            t = promote(ta, tb)
            fn = {'+': 'add_c', '-': 'sub_c', '/': 'div_c'}[op]
            return f'rt.{fn}({a}, {b})', t
        if op == '%':
            t = promote(ta, tb)
            if t.base in ('int', 'uint'):
                return f'rt.imod({a}, {b})', t
            return f'rt.mod({a}, {b})', t
        if op in ('&', '|', '^', '<<', '>>'):
            t = promote(ta, tb)
            return f'(rt.to_int({a}) {op} rt.to_int({b}))', t.with_base('int')
        raise ShaderError(f'bad operator {op}')

    def e_assign(self, node, mask, scope):
        op, target, value = node[1], node[2], node[3]
        vcode, vt = self.expr(value, mask, scope)
        if op != '=':
            cur, ct = self.expr(target, mask, scope)
            binop = op[:-1]
            vcode, vt = self.e_bin(('bin', binop, target, value), mask, scope)
        return self.store(target, vcode, vt, mask, scope)

    def store(self, target, vcode, vt, mask, scope):
        k = target[0]
        if k == 'var':
            py, t = self.expr(target, mask, scope)
            code = self.coerce(vcode, vt, t)
            self.emit(f'{py} = rt.sel({mask}, {code}, {py})')
            return py, t
        if k == 'member':
            base, bt = self.expr(target[1], mask, scope)
            field = target[2]
            if bt.base == 'struct':
                fields = {f[1]: f[0] for f in self.structs.get(bt.struct, [])}
                ft = fields.get(field)
                if ft is None:
                    raise ShaderError(f'{bt.struct} has no field {field!r}')
                code = self.coerce(vcode, vt, ft)
                self.emit(f'{base} = rt.struct_set({base}, {field!r}, '
                          f'rt.sel({mask}, {code}, ({base})[{field!r}]))')
                self.rebind(target[1], base, scope)
                return f'({base})[{field!r}]', ft
            idx = swizzle_indices(field)
            if idx is None:
                raise ShaderError(f'bad swizzle {field!r}')
            tmp = self.new_tmp('_s')
            self.emit(f'{tmp} = rt.sw_set({base}, {idx}, {vcode})')
            self.emit(f'{base} = rt.sel({mask}, {tmp}, {base})')
            self.rebind(target[1], base, scope)
            return base, bt
        if k == 'index':
            base, bt = self.expr(target[1], mask, scope)
            const = _const_int_expr(target[2])
            icode, _ = self.expr(target[2], mask, scope)
            if bt.array:
                if const is None:
                    raise ShaderError('dynamic array writes are not supported')
                self.emit(f'{base}[{const}] = rt.sel({mask}, {vcode}, {base}[{const}])')
                return f'{base}[{const}]', bt.elem()
            if bt.is_matrix:
                if const is None:
                    raise ShaderError('dynamic matrix column writes are not supported')
                tmp = self.new_tmp('_s')
                self.emit(f'{tmp} = rt.col_set({base}, {const}, {vcode})')
                self.emit(f'{base} = rt.sel({mask}, {tmp}, {base})')
                self.rebind(target[1], base, scope)
                return f'({base})[:, {const}]', GType(bt.base, bt.rows)
            if const is None:
                raise ShaderError('dynamic vector component writes are not supported')
            tmp = self.new_tmp('_s')
            self.emit(f'{tmp} = rt.sw_set({base}, [{const}], {vcode})')
            self.emit(f'{base} = rt.sel({mask}, {tmp}, {base})')
            self.rebind(target[1], base, scope)
            return base, bt
        raise ShaderError('invalid assignment target')

    def rebind(self, node, pycode, scope):
        return

    # ------------------------------------------------------------ calls
    def e_call(self, node, mask, scope):
        name, args = node[1], node[2]
        if name in BY_NAME and BY_NAME[name].base != 'struct':
            return self.constructor(BY_NAME[name], args, mask, scope)
        if name in self.structs:
            fields = self.structs[name]
            parts = []
            for (ft, fname, farr), a in zip(fields, args):
                code, t = self.expr(a, mask, scope)
                parts.append(f'{fname!r}: {self.coerce(code, t, ft)}')
            return '{' + ', '.join(parts) + '}', GType('struct', 1, 0, 0, name)
        if name in self.funcs:
            return self.user_call(name, args, mask, scope)
        b = BUILTINS.get(name)
        if b is None and self.hlsl:
            b = BUILTINS.get(name.lower())
        if b is None:
            raise ShaderError(f'unknown function {name!r}')
        fn, rule, lo, hi = b
        if not (lo <= len(args) <= hi):
            raise ShaderError(f'{name}() takes {lo}..{hi} arguments, got {len(args)}')
        codes = []
        types = []
        for a in args:
            c, t = self.expr(a, mask, scope)
            codes.append(c)
            types.append(t)
        rtype = rule(types)
        return f'rt.{fn}(' + ', '.join(codes) + ')', rtype

    def user_call(self, name, args, mask, scope):
        fdef = self.funcs[name]
        params = fdef[3]
        ret_t = fdef[1]
        codes = []
        outs = []
        for i, (qual, ptype, pname, parr) in enumerate(params):
            if i < len(args):
                c, t = self.expr(args[i], mask, scope)
                codes.append(self.coerce(c, t, ptype))
                if qual in ('out', 'inout'):
                    outs.append((args[i], ptype))
            else:
                codes.append(self.zero(ptype))
        call = f'f_{_san(name)}({mask}, ' + ', '.join(codes) + ')'
        if not outs:
            if ret_t.base == 'void':
                self.emit(f'_ = {call}')
                return 'None', BY_NAME['void']
            tmp = self.new_tmp('_r')
            self.emit(f'{tmp} = {call}')
            return tmp, ret_t
        tmp = self.new_tmp('_r')
        self.emit(f'{tmp} = {call}')
        base = 0
        if ret_t.base != 'void':
            base = 1
        for j, (arg_node, ptype) in enumerate(outs):
            self.store(arg_node, f'{tmp}[{base + j}]', ptype, mask, scope)
        if ret_t.base == 'void':
            return 'None', BY_NAME['void']
        return f'{tmp}[0]', ret_t

    def constructor(self, t, args, mask, scope):
        codes = []
        types = []
        for a in args:
            c, ct = self.expr(a, mask, scope)
            codes.append(c)
            types.append(ct)
        if t.is_matrix:
            return f'rt.mat({t.n}, {t.rows}, ' + ', '.join(codes) + ')', t
        if t.n > 1:
            code = f'rt.vec({t.n}, ' + ', '.join(codes) + ')'
            if t.base == 'int' or t.base == 'uint':
                code = f'rt.to_int({code})'
            elif t.base == 'bool':
                code = f'rt.to_bool({code})'
            return code, t
        if not codes:
            return self.zero(t), t
        if t.base == 'float':
            return f'rt.to_float({codes[0]})', t
        if t.base in ('int', 'uint'):
            return f'rt.to_int({codes[0]})', t
        if t.base == 'bool':
            return f'rt.to_bool({codes[0]})', t
        return codes[0], t

    # ------------------------------------------------------------- coercion
    def coerce(self, code, src, dst):
        if src == dst or dst.base == 'struct' or dst.array:
            return code
        if dst.is_matrix:
            return code
        if dst.n > 1 and src.n == 1:
            code = f'rt.splat({code}, {dst.n})'
        elif dst.n > 1 and src.n > dst.n:
            code = f'rt.sw({code}, {list(range(dst.n))})'
        elif dst.n == 1 and src.n > 1:
            code = f'rt.sw({code}, [0])'
        if dst.base == 'float' and src.base != 'float':
            code = f'rt.to_float({code})'
        elif dst.base in ('int', 'uint') and src.base not in ('int', 'uint'):
            code = f'rt.to_int({code})'
        elif dst.base == 'bool' and src.base != 'bool':
            code = f'rt.to_bool({code})'
        return code


def _san(name):
    return ''.join(c if c.isalnum() or c == '_' else '_' for c in name)


def _const_int_expr(node):
    if node[0] == 'num' and node[2]:
        try:
            return int(float(node[1].rstrip('uUfFhH')))
        except ValueError:
            return None
    if node[0] == 'un' and node[1] == '-':
        v = _const_int_expr(node[2])
        return None if v is None else -v
    return None


def _calls_in(node, found):
    """Collect every function name called anywhere inside an AST node."""
    if isinstance(node, tuple):
        if node and node[0] == 'call' and len(node) > 1 and isinstance(node[1], str):
            found.add(node[1])
        for item in node:
            _calls_in(item, found)
    elif isinstance(node, list):
        for item in node:
            _calls_in(item, found)
    return found


def check_recursion(decls):
    """Reject recursive shaders at compile time.

    Under SIMT masking every lane walks both sides of a branch, so a recursive
    call never hits its own base case and the interpreter simply runs out of
    stack. Catching it here turns a hang into a message the user can act on.
    """
    graph = {}
    for d in decls:
        if isinstance(d, tuple) and d and d[0] == 'func':
            graph.setdefault(d[2], set()).update(_calls_in(d[4], set()))
    state = {}

    def visit(name, stack):
        if state.get(name) == 'done':
            return
        if state.get(name) == 'open':
            cycle = ' -> '.join(stack[stack.index(name):] + [name])
            raise ShaderError(f"recursive call not supported: {cycle}")
        state[name] = 'open'
        for callee in sorted(graph.get(name, ())):
            if callee in graph:
                visit(callee, stack + [callee])
        state[name] = 'done'

    for name in sorted(graph):
        visit(name, [name])


def compile_source(src, hlsl=False, defines=None):
    """Returns (python_source, outputs, uniforms, inputs, warnings)."""
    decls, structs = parse(src, hlsl=hlsl, defines=defines)
    check_recursion(decls)
    g = Generator(structs, hlsl=hlsl)
    code, outs = g.generate(decls)
    return code, outs, g.uniforms, g.inputs, g.warnings
