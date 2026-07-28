"""Shader execution without generating or executing Python source.

The previous backend emitted Python text and ran it through `exec`. Blender's
extension platform does not permit that, and reasonably so. This walks the
parsed tree instead, calling the same runtime library the generated code used
to call -- `rt.sel`, `rt.sw_set`, `rt.mat_mul_vec` and the rest are all still
there, and none of them changed.

The execution model is unchanged too, which is the important part: every value
is a NumPy array with one entry per lane, and divergent control flow is handled
with masks rather than branches. An `if` evaluates both sides and selects; a
loop runs until every lane has left it. That is what lets one shader take
different paths for different pixels.

Cost of the change: dispatch now happens per AST node per execution rather than
once at compile time. Because each node operates on a whole array of lanes, that
overhead is amortised across the batch -- it is a constant factor on the number
of nodes, not on the number of pixels.
"""

import numpy as np

from . import builtins as rt
from .builtins import BUILTINS
from .gtypes import BOOL, FLOAT, INT, VOID, GType, promote, swizzle_indices
from .lexer import ShaderError

MAX_LOOP = 4096


class Scope:
    """Lexical scope holding (value, type) per name."""

    __slots__ = ('vars', 'parent')

    def __init__(self, parent=None):
        self.vars = {}
        self.parent = parent

    def get(self, name):
        s = self
        while s is not None:
            if name in s.vars:
                return s.vars[name]
            s = s.parent
        return None

    def put(self, name, value, gtype):
        self.vars[name] = [value, gtype]

    def set(self, name, value):
        s = self
        while s is not None:
            if name in s.vars:
                s.vars[name][0] = value
                return True
            s = s.parent
        return False


class Flags:
    """Lanes that have left the current statement by some route."""

    __slots__ = ('ret', 'brk', 'cont', 'discard')

    def __init__(self, n):
        self.ret = np.zeros(n, bool)
        self.brk = np.zeros(n, bool)
        self.cont = np.zeros(n, bool)
        self.discard = np.zeros(n, bool)

    def live(self, mask):
        return mask & ~self.ret & ~self.brk & ~self.cont


class Interpreter:
    def __init__(self, decls, structs, hlsl=False):
        self.structs = structs
        self.hlsl = hlsl
        self.funcs = {}
        self.globals = []
        self.uniforms = []
        self.inputs = []
        self.outputs = []
        self.warnings = []
        for d in decls:
            kind = d[0]
            if kind == 'func':
                self.funcs.setdefault(d[2], []).append(d)
            elif kind == 'global':
                _k, quals, gtype, name, arr, init = d
                t = GType(gtype.base, gtype.n, gtype.rows, arr, gtype.struct) \
                    if arr else gtype
                entry = (name, t, init, quals)
                if 'uniform' in quals:
                    self.uniforms.append(entry)
                elif 'in' in quals or 'attribute' in quals or 'varying' in quals:
                    self.inputs.append(entry)
                elif 'out' in quals:
                    self.outputs.append(entry)
                else:
                    self.globals.append(entry)

    def validate(self):
        """Reject unknown functions before the shader ever runs.

        The generator caught these while emitting code. An interpreter would
        otherwise not notice until a pixel happened to reach the call, which
        turns a compile error into a render that fails halfway.
        """
        from .gtypes import BY_NAME

        def walk(node):
            if isinstance(node, tuple) and node and node[0] == 'call':
                name = node[1]
                known = (name in self.funcs or name in BUILTINS
                         or name in BY_NAME or name in self.structs)
                if not known:
                    raise ShaderError(f'unknown function {name!r}')
            if isinstance(node, (tuple, list)):
                for part in node:
                    walk(part)

        for cands in self.funcs.values():
            for fn in cands:
                walk(fn[4])
        return self

    # ------------------------------------------------------------- entry
    def run(self, uniforms, inputs, n):
        n = int(n)
        root = Scope()
        for name, t, init, _q in self.uniforms:
            v = uniforms.get(name)
            if v is None:
                v = self.const(init, t, n) if init is not None else self.zero(t, n)
            root.put(name, self.adapt(v, t, n), t)
        for name, t, init, _q in self.inputs:
            # a varying can be supplied under its declared name or under the
            # binding it maps to, and the generator accepted either
            from .codegen import VARYINGS
            v = inputs.get(name)
            if v is None:
                bound = VARYINGS.get(name)
                if bound is not None:
                    v = inputs.get(bound[0])
            if v is None:
                v = self.varying(name, t, n)
            root.put(name, self.adapt(v, t, n), t)
        for name, t, init, _q in self.globals:
            root.put(name, self.const(init, t, n) if init is not None
                     else self.zero(t, n), t)
        for name, t, init, _q in self.outputs:
            root.put(name, self.zero(t, n), t)

        self.root = root
        entry = self.pick('main', 0) or self.pick('mainImage', 0)
        if entry is None:
            for cands in self.funcs.values():
                entry = cands[0]
                break
        if entry is None:
            raise ShaderError('no main() in shader')

        flags = Flags(n)
        mask = np.ones(n, bool)
        self.call_user(entry, [], mask, root, flags, n)

        outs = {}
        for name, t, _i, _q in self.outputs:
            slot = root.get(name)
            outs[name] = slot[0] if slot else self.zero(t, n)
        return outs, flags.discard

    # -------------------------------------------------------- statements
    def block(self, node, mask, scope, flags, n):
        inner = Scope(scope)
        for st in (node[1] if node[0] == 'block' else [node]):
            mask = self.stmt(st, mask, inner, flags, n)
            if not mask.any():
                break
        return mask

    def stmt(self, node, mask, scope, flags, n):
        k = node[0]
        if k == 'nop' or not mask.any():
            return mask
        if k == 'block':
            return self.block(node, mask, scope, flags, n)
        if k == 'expr':
            self.expr(node[1], mask, scope, flags, n)
            return mask
        if k == 'decl':
            gtype = node[1]
            for name, arr, init in node[2]:
                t = GType(gtype.base, gtype.n, gtype.rows, arr, gtype.struct) \
                    if arr else gtype
                if init is not None:
                    v, it = self.expr(init, mask, scope, flags, n)
                    v = self.coerce(v, it, t, n)
                else:
                    v = self.zero(t, n)
                scope.put(name, v, t)
            return mask
        if k == 'if':
            cond, _ct = self.expr(node[1], mask, scope, flags, n)
            c = self.as_lane_mask(cond, n)
            taken = mask & c
            if taken.any():
                self.block(node[2], taken, scope, flags, n)
            if node[3] is not None:
                other = mask & ~c
                if other.any():
                    self.block(node[3], other, scope, flags, n)
            return flags.live(mask)
        if k in ('for', 'while', 'do'):
            return self.loop(node, mask, scope, flags, n)
        if k == 'return':
            if node[1] is not None and self.ret_type is not None:
                v, t = self.expr(node[1], mask, scope, flags, n)
                v = self.coerce(v, t, self.ret_type, n)
                self.ret_val = rt.sel(mask, v, self.ret_val)
            flags.ret = flags.ret | mask
            return flags.live(mask)
        if k == 'break':
            flags.brk = flags.brk | mask
            return flags.live(mask)
        if k == 'continue':
            flags.cont = flags.cont | mask
            return flags.live(mask)
        if k == 'discard':
            flags.discard = flags.discard | mask
            flags.ret = flags.ret | mask
            return flags.live(mask)
        raise ShaderError(f'unsupported statement {k!r}')

    def loop(self, node, mask, scope, flags, n):
        kind = node[0]
        sc = Scope(scope)
        if kind == 'for':
            init, cond, step, body = node[1], node[2], node[3], node[4]
            if init and init[0] != 'nop':
                self.stmt(init, mask, sc, flags, n)
        elif kind == 'while':
            cond, body, step = node[1], node[2], None
        else:
            body, cond, step = node[1], node[2], None

        active = mask.copy()
        saved_brk = flags.brk.copy()
        flags.brk = np.zeros(n, bool)
        first = True
        for _ in range(MAX_LOOP):
            if not active.any():
                break
            if cond is not None and not (kind == 'do' and first):
                c, _t = self.expr(cond, active, sc, flags, n)
                active = active & self.as_lane_mask(c, n)
                if not active.any():
                    break
            first = False
            saved_cont = flags.cont.copy()
            flags.cont = np.zeros(n, bool)
            self.block(body, active, sc, flags, n)
            active = active & ~flags.brk & ~flags.ret
            flags.cont = saved_cont
            if step is not None and step[0] != 'nop':
                if active.any():
                    self.expr(step, active, sc, flags, n)
        flags.brk = saved_brk
        return flags.live(mask)

    # ------------------------------------------------------- expressions
    def expr(self, node, mask, scope, flags, n):
        k = node[0]
        if k == 'num':
            return self.number(node, n)
        if k == 'var':
            slot = scope.get(node[1])
            if slot is None:
                raise ShaderError(f'unknown identifier {node[1]!r}')
            return slot[0], slot[1]
        if k == 'bin':
            return self.binary(node, mask, scope, flags, n)
        if k == 'assign':
            return self.assign(node, mask, scope, flags, n)
        if k == 'cond':
            c, _ct = self.expr(node[1], mask, scope, flags, n)
            cb = rt.to_bool(c)
            a, at = self.expr(node[2], mask, scope, flags, n)
            b, bt = self.expr(node[3], mask, scope, flags, n)
            t = promote(at, bt)
            return rt.sel(cb, self.coerce(a, at, t, n),
                          self.coerce(b, bt, t, n)), t
        if k == 'call':
            return self.call(node, mask, scope, flags, n)
        if k == 'member':
            base, bt = self.expr(node[1], mask, scope, flags, n)
            field = node[2]
            if bt.base == 'struct':
                fields = {f[1]: f[0] for f in self.structs.get(bt.struct, [])}
                if field not in fields:
                    raise ShaderError(f'{bt.struct} has no field {field!r}')
                return base[field], fields[field]
            idx = swizzle_indices(field)
            if idx is None:
                raise ShaderError(f'bad swizzle {field!r}')
            return rt.sw(base, idx), GType(bt.base, len(idx))
        if k == 'index':
            base, bt = self.expr(node[1], mask, scope, flags, n)
            i, _it = self.expr(node[2], mask, scope, flags, n)
            return self.subscript(base, bt, i, n)
        if k in ('post', 'pre'):
            # ('post', '++', target) -- the operator comes first
            op, target = node[1], node[2]
            cur, t = self.expr(target, mask, scope, flags, n)
            delta = np.ones(n, np.float32) * (1.0 if op == '++' else -1.0)
            newv = self.coerce(cur + self.coerce(delta, FLOAT, t, n), t, t, n)
            self.store(target, newv, t, mask, scope, flags, n)
            return (cur if k == 'post' else newv), t
        if k == 'un':
            op, operand = node[1], node[2]
            if op in ('++', '--'):
                # ++x is a write as well as a read, so it shares the post path
                cur, t = self.expr(operand, mask, scope, flags, n)
                delta = np.ones(n, np.float32) * (1.0 if op == '++' else -1.0)
                newv = self.coerce(cur + self.coerce(delta, FLOAT, t, n),
                                   t, t, n)
                self.store(operand, newv, t, mask, scope, flags, n)
                return newv, t
            v, t = self.expr(operand, mask, scope, flags, n)
            if op == '-':
                return -v, t
            if op == '+':
                return v, t
            if op == '!':
                return ~rt.to_bool(v), BOOL
            if op == '~':
                return ~rt.to_int(v), INT
            raise ShaderError(f'unsupported unary {op!r}')
        if k == 'arr':
            parts = [self.expr(e, mask, scope, flags, n) for e in node[2]]
            return [p[0] for p in parts], node[1]
        raise ShaderError(f'unsupported expression {k!r}')

    def binary(self, node, mask, scope, flags, n):
        op = node[1]
        a, at = self.expr(node[2], mask, scope, flags, n)
        b, bt = self.expr(node[3], mask, scope, flags, n)
        if op in ('&&', 'and'):
            return rt.to_bool(a) & rt.to_bool(b), BOOL
        if op in ('||', 'or'):
            return rt.to_bool(a) | rt.to_bool(b), BOOL
        if op in ('==', '!=', '<', '>', '<=', '>='):
            t = promote(at, bt)
            x = self.coerce(a, at, t, n)
            y = self.coerce(b, bt, t, n)
            if t.n > 1:
                x, y = rt.sw(x, [0]), rt.sw(y, [0])
            r = {'==': np.equal, '!=': np.not_equal, '<': np.less,
                 '>': np.greater, '<=': np.less_equal,
                 '>=': np.greater_equal}[op](x, y)
            return np.reshape(r, (-1,)), BOOL
        if op == '*' and (at.is_matrix or bt.is_matrix):
            if at.is_matrix and bt.is_matrix:
                return rt.mat_mul_mat(a, b), at
            if at.is_matrix:
                return rt.mat_mul_vec(a, b), GType(at.base, at.rows)
            return rt.vec_mul_mat(a, b), GType(bt.base, bt.n)
        t = promote(at, bt)
        x = self.coerce(a, at, t, n)
        y = self.coerce(b, bt, t, n)
        if op == '+':
            r = x + y
        elif op == '-':
            r = x - y
        elif op == '*':
            r = x * y
        elif op == '/':
            r = x / np.where(np.asarray(y) == 0, np.asarray(y, np.float32) + 1e-30, y)
        elif op == '%':
            r = rt.mod(x, y) if t.base == 'float' else np.remainder(
                rt.to_int(x), np.where(np.asarray(rt.to_int(y)) == 0, 1,
                                       rt.to_int(y)))
        elif op == '&':
            r = rt.to_int(x) & rt.to_int(y)
        elif op == '|':
            r = rt.to_int(x) | rt.to_int(y)
        elif op == '^':
            r = rt.to_int(x) ^ rt.to_int(y)
        elif op == '<<':
            r = rt.to_int(x) << rt.to_int(y)
        elif op == '>>':
            r = rt.to_int(x) >> rt.to_int(y)
        else:
            raise ShaderError(f'unsupported operator {op!r}')
        return r, t

    def assign(self, node, mask, scope, flags, n):
        op, target, value = node[1], node[2], node[3]
        v, vt = self.expr(value, mask, scope, flags, n)
        if op != '=':
            cur, ct = self.expr(target, mask, scope, flags, n)
            v, vt = self.binary(('bin', op[:-1], target, value), mask, scope,
                                flags, n)
        return self.store(target, v, vt, mask, scope, flags, n)

    def store(self, target, value, vtype, mask, scope, flags, n):
        k = target[0]
        if k == 'var':
            slot = scope.get(target[1])
            if slot is None:
                raise ShaderError(f'assignment to unknown {target[1]!r}')
            t = slot[1]
            slot[0] = rt.sel(mask, self.coerce(value, vtype, t, n), slot[0])
            return slot[0], t
        if k == 'member':
            base, bt = self.expr(target[1], mask, scope, flags, n)
            field = target[2]
            if bt.base == 'struct':
                fields = {f[1]: f[0] for f in self.structs.get(bt.struct, [])}
                if field not in fields:
                    raise ShaderError(f'{bt.struct} has no field {field!r}')
                ft = fields[field]
                merged = dict(base)
                merged[field] = rt.sel(mask, self.coerce(value, vtype, ft, n),
                                       base[field])
                self.store(target[1], merged, bt, np.ones(n, bool), scope,
                           flags, n)
                return merged[field], ft
            idx = swizzle_indices(field)
            if idx is None:
                raise ShaderError(f'bad swizzle {field!r}')
            updated = rt.sw_set(base, idx, value)
            self.store(target[1], rt.sel(mask, updated, base), bt,
                       np.ones(n, bool), scope, flags, n)
            return base, bt
        if k == 'index':
            base, bt = self.expr(target[1], mask, scope, flags, n)
            const = _const_int(target[2])
            if bt.array:
                if const is None:
                    raise ShaderError('dynamic array writes are not supported')
                items = list(base)
                items[const] = rt.sel(mask, value, items[const])
                self.store(target[1], items, bt, np.ones(n, bool), scope,
                           flags, n)
                return items[const], bt.elem()
            if bt.is_matrix:
                if const is None:
                    raise ShaderError('dynamic matrix writes are not supported')
                cols = list(base)
                cols[const] = rt.sel(mask, value, cols[const])
                self.store(target[1], cols, bt, np.ones(n, bool), scope,
                           flags, n)
                return cols[const], GType(bt.base, bt.rows)
            if const is None:
                raise ShaderError('dynamic component writes are not supported')
            updated = rt.sw_set(base, [const], value)
            self.store(target[1], rt.sel(mask, updated, base), bt,
                       np.ones(n, bool), scope, flags, n)
            return base, bt
        raise ShaderError(f'cannot assign to {k!r}')

    # -------------------------------------------------------------- calls
    def pick(self, name, argc):
        for f in self.funcs.get(name, []):
            if len(f[3]) == argc:
                return f
        cands = self.funcs.get(name)
        return cands[0] if cands else None

    def call(self, node, mask, scope, flags, n):
        name, args = node[1], node[2]
        ctor = self.constructor(name, args, mask, scope, flags, n)
        if ctor is not None:
            return ctor
        fn = self.pick(name, len(args))
        if fn is not None:
            return self.call_user(fn, args, mask, scope, flags, n)
        impl = BUILTINS.get(name)
        if impl is None:
            raise ShaderError(f"unknown function {name!r}")
        # (runtime name, return-type rule, min args, max args)
        rt_name, rule, lo, hi = impl
        if not (lo <= len(args) <= hi):
            raise ShaderError(f'{name} takes {lo}..{hi} arguments, '
                              f'got {len(args)}')
        fn = getattr(rt, rt_name, None)
        if fn is None:
            raise ShaderError(f'runtime has no {rt_name!r}')
        vals, types = [], []
        for a in args:
            v, t = self.expr(a, mask, scope, flags, n)
            vals.append(v)
            types.append(t)
        out = fn(*vals)
        # the rules take the argument-type list, not varargs
        rtype = rule(types) if callable(rule) else rule
        return out, rtype if rtype is not None else (types[0] if types
                                                     else FLOAT)

    def call_user(self, fn, args, mask, scope, flags, n):
        _k, rtype, name, params, body = fn
        inner = Scope()
        for name_g, t, init, _q in (self.uniforms + self.inputs + self.globals
                                    + self.outputs):
            slot = self.root.get(name_g) if self.root else None
            if slot is not None:
                inner.vars[name_g] = slot
        writeback = []
        for i, p in enumerate(params):
            direction, ptype, pname = p[0], p[1], p[2]
            if i < len(args):
                v, vt = self.expr(args[i], mask, scope, flags, n)
                v = self.coerce(v, vt, ptype, n)
            else:
                v = self.zero(ptype, n)
            inner.put(pname, v, ptype)
            if direction in ('out', 'inout') and i < len(args):
                writeback.append((args[i], pname, ptype))

        saved = (self.ret_type, self.ret_val)
        self.ret_type = None if rtype == VOID else rtype
        self.ret_val = self.zero(rtype, n) if rtype != VOID else None
        sub = Flags(n)
        sub.discard = flags.discard
        self.block(body, mask, inner, sub, n)
        flags.discard = flags.discard | sub.discard
        result = self.ret_val
        self.ret_type, self.ret_val = saved

        for target, pname, ptype in writeback:
            slot = inner.get(pname)
            if slot is not None:
                self.store(target, slot[0], ptype, mask, scope, flags, n)
        return (result if result is not None else np.zeros(n, np.float32),
                rtype if rtype != VOID else FLOAT)

    def constructor(self, name, args, mask, scope, flags, n):
        from .gtypes import BY_NAME
        t = BY_NAME.get(name)
        if t is None:
            if name in self.structs:
                fields = self.structs[name]
                vals = [self.expr(a, mask, scope, flags, n) for a in args]
                out = {}
                for i, (ftype, fname, _arr) in enumerate(fields):
                    out[fname] = self.coerce(vals[i][0], vals[i][1], ftype, n) \
                        if i < len(vals) else self.zero(ftype, n)
                return out, GType('struct', 1, 1, 0, name)
            return None
        parts = []
        for a in args:
            v, vt = self.expr(a, mask, scope, flags, n)
            parts.append(v)
        if t.is_matrix:
            return rt.mat(t.n, t.rows, *parts), t
        if t.n == 1:
            v = parts[0] if parts else np.zeros(n, np.float32)
            src = FLOAT
            return self.coerce(v, src, t, n), t
        return rt.vec(t.n, *parts), t

    # ------------------------------------------------------------ helpers
    def subscript(self, base, bt, i, n):
        const = None
        try:
            arr = np.asarray(i).reshape(-1)
            if arr.size and np.all(arr == arr[0]):
                const = int(arr[0])
        except Exception:                                       # noqa: BLE001
            const = None
        if bt.array:
            if const is not None:
                return base[max(0, min(const, len(base) - 1))], bt.elem()
            idx = rt.to_int(i)
            out = base[0]
            for j in range(1, len(base)):
                out = rt.sel(np.reshape(idx, (-1,)) == j, base[j], out)
            return out, bt.elem()
        if bt.is_matrix:
            col = base[max(0, min(const or 0, len(base) - 1))]
            return col, GType(bt.base, bt.rows)
        if const is not None:
            return rt.sw(base, [max(0, min(const, bt.n - 1))]), GType(bt.base, 1)
        idx = np.reshape(rt.to_int(i), (-1,))
        out = rt.sw(base, [0])
        for j in range(1, bt.n):
            out = rt.sel(idx == j, rt.sw(base, [j]), out)
        return out, GType(bt.base, 1)

    def as_lane_mask(self, value, n):
        """One boolean per lane, whatever shape the condition arrived in."""
        b = np.asarray(rt.to_bool(value))
        if b.ndim > 1:
            b = b[:, 0]
        b = np.reshape(b, (-1,)).astype(bool)
        if b.size == n:
            return b
        if b.size == 1:
            return np.full(n, bool(b[0]))
        out = np.zeros(n, bool)
        out[:min(n, b.size)] = b[:min(n, b.size)]
        return out

    def coerce(self, value, src, dst, n):
        if src is dst or dst.base == 'struct' or dst.array or dst.is_matrix:
            return value
        v = value
        if dst.n > 1 and src.n == 1:
            v = rt.splat(v, dst.n)
        elif dst.n > 1 and src.n > dst.n:
            v = rt.sw(v, list(range(dst.n)))
        elif dst.n == 1 and src.n > 1:
            v = rt.sw(v, [0])
        if dst.base == 'float' and src.base != 'float':
            v = rt.to_float(v)
        elif dst.base in ('int', 'uint') and src.base not in ('int', 'uint'):
            v = rt.to_int(v)
        elif dst.base == 'bool' and src.base != 'bool':
            v = rt.to_bool(v)
        return v

    def zero(self, t, n):
        if t == VOID:
            return None
        if t.array:
            return [self.zero(t.elem(), n) for _ in range(max(t.array, 1))]
        if t.base == 'struct':
            return {f[1]: self.zero(f[0], n) for f in self.structs.get(t.struct, [])}
        if t.is_matrix:
            return [np.zeros((n, t.rows), np.float32) for _ in range(t.n)]
        if t.n == 1:
            return np.zeros(n, bool if t.base == 'bool' else
                            np.int32 if t.base in ('int', 'uint') else np.float32)
        return np.zeros((n, t.n), np.float32)

    def number(self, node, n):
        raw = node[1]
        is_int = not node[2] if len(node) > 2 else False
        text = raw.rstrip('uUfFhH')
        try:
            if any(c in text for c in '.eE') or not node[2]:
                val = float(text)
                return np.full(n, val, np.float32), FLOAT
            val = int(text, 0)
            return np.full(n, val, np.int32), INT
        except ValueError:
            return np.zeros(n, np.float32), FLOAT

    def const(self, init, t, n):
        try:
            v, vt = self.expr(init, np.ones(n, bool), Scope(), Flags(n), n)
            return self.coerce(v, vt, t, n)
        except Exception:                                       # noqa: BLE001
            return self.zero(t, n)

    def adapt(self, v, t, n):
        if t.base == 'sampler':
            return v
        if t.array or t.base == 'struct' or t.is_matrix:
            return v
        arr = rt.bc(v, n)
        if t.n > 1 and (np.ndim(arr) == 1 or np.shape(arr)[-1] != t.n):
            arr = rt.splat(rt.sw(arr, [0]) if np.ndim(arr) > 1 else arr, t.n)
        return arr

    def varying(self, name, t, n):
        # the same binding table the generator used, rather than a second guess
        from .codegen import VARYINGS
        entry = VARYINGS.get(name)
        if entry is None:
            return self.zero(t, n)
        v = getattr(rt.ctx(), entry[0], None)
        return self.adapt(v, t, n) if v is not None else self.zero(t, n)

    # ------------------------------------------------- schemas for the UI
    def uniform_schema_list(self):
        return [(name, t, init) for name, t, init, _q in self.uniforms]

    def input_schema_list(self):
        return [(name, t, name) for name, t, _i, _q in self.inputs]

    def output_schema_list(self):
        """(glsl name, key in the returned dict, type) -- the same for both,
        because the interpreter returns its outputs keyed by their real name
        rather than by a generated variable."""
        outs = [(name, name, t) for name, t, _i, _q in self.outputs]
        for cands in self.funcs.values():
            fn = cands[0]
            if fn[2] in ('main', 'mainImage') and fn[1] != VOID:
                outs.append(('__return', '__return', fn[1]))
        return outs

    root = None
    ret_type = None
    ret_val = None


def _const_int(node):
    if node and node[0] == 'num':
        try:
            return int(float(node[1].rstrip('uUfFhH')))
        except ValueError:
            return None
    return None
