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
from .gtypes import (BOOL, BY_NAME, FLOAT, INT, UINT, VOID, GType, promote,
                     swizzle_indices)
from .lexer import ShaderError

MAX_LOOP = 4096


def param_type(p):
    """A parameter's type, its array size included (`vec3 pa[4]`)."""
    ptype = p[1]
    if len(p) > 3 and p[3]:
        return GType(ptype.base, ptype.n, ptype.rows, p[3], ptype.struct)
    return ptype


def param_default(p):
    """The expression a parameter is declared with (HLSL), or None."""
    return p[4] if len(p) > 4 else None


def field_type(f):
    """A struct field's type, its array size included (`float w[3]`): the
    third element of the parser's field tuple. Everything that asks what a
    field is asks here -- the value a declared struct starts with, a field
    read, a field write, the constructor and the name check. (Each read the
    bare element type: the field was stored and typed as ONE element, so
    `s.w[1]` indexed a float.)"""
    ftype, _name, arr = f
    if arr:
        return GType(ftype.base, ftype.n, ftype.rows, arr, ftype.struct)
    return ftype


def arity(fn):
    """(fewest, most) arguments a call of one function may pass: every
    parameter up to the last one written without a default, to all of
    them."""
    params = fn[3]
    need = 0
    for i, p in enumerate(params):
        if param_default(p) is None:
            need = i + 1
    return need, len(params)


def count_text(spans):
    """'1 argument', '1 to 2 arguments', '1 or 3 arguments' -- the
    argument counts a name accepts, for the message that refuses another."""
    spans = sorted(set(spans))
    parts = [str(lo) if lo == hi else f'{lo} to {hi}' for lo, hi in spans]
    return ' or '.join(parts) + (' argument' if spans == [(1, 1)]
                                 else ' arguments')


def struct_type(name):
    return GType('struct', 1, 0, 0, name)


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
        #: struct name -> {field name: type}, in declaration order, built
        #: when first asked for (`fields`)
        self.field_types = {}
        self.hlsl = hlsl
        self.funcs = {}
        self.globals = []
        self.uniforms = []
        self.inputs = []
        self.outputs = []
        self.warnings = []
        #: every global declaration as (role, entry), in SOURCE order: the
        #: order run() binds them in and names.py resolves them in, so an
        #: initialiser sees exactly what is declared above it
        self.order = []
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
                    role, into = 'uniform', self.uniforms
                elif 'in' in quals or 'attribute' in quals or 'varying' in quals:
                    role, into = 'input', self.inputs
                elif 'out' in quals:
                    role, into = 'output', self.outputs
                else:
                    role, into = 'global', self.globals
                into.append(entry)
                self.order.append((role, entry))
        #: name -> [(fewest, most, function)]: what `pick` reads on every
        #: call, counted once
        self.overloads = {name: [arity(f) + (f,) for f in cands]
                          for name, cands in self.funcs.items()}
        #: the function the run starts at (None: the shader has none)
        self.entry = self.find_entry()
        self.warn_unbound()

    def find_entry(self):
        """The entry point: `main`, `mainImage`, the HLSL names, else the
        first function. Among overloads of the name, the one that takes no
        parameters."""
        from .codegen import ENTRY_NAMES
        for name in ENTRY_NAMES:
            cands = self.funcs.get(name)
            if cands:
                return next((f for f in cands if not f[3]), cands[0])
        for cands in self.funcs.values():
            return cands[0]
        return None

    def warn_unbound(self):
        """What compiles and can only read zero: a declared input, or an
        entry parameter, whose name the renderer binds nothing to (a
        misspelled `vNormals`). The generator said so; the interpreter that
        replaced it kept the list and never wrote to it, so the node's
        warning box had nothing to show."""
        from .codegen import INPUT_KEYS, known_input, param_binding
        for name, _t, _i, _q in self.inputs:
            if not known_input(name):
                self.warnings.append(f'unknown input {name!r}: bound to zero')
        for p in (self.entry[3] if self.entry else ()):
            if p[0] not in ('out', 'inout') and p[2] and \
                    param_binding(p[2], param_type(p)) not in INPUT_KEYS:
                self.warnings.append(f'unknown input {p[2]!r}: bound to zero')

    def validate(self):
        """Reject what the run would refuse before the shader ever runs.

        The generator caught these while emitting code. An interpreter would
        otherwise not notice until a pixel happened to reach the line, which
        turns a compile error into a render that fails halfway -- or into no
        error at all, when the line sits in a branch no lane takes. Unknown
        functions were checked here first; identifiers, struct fields and
        swizzles came with shaders/names.py, and the argument count of every
        call and two vector widths that cannot be combined are checked there
        too now, each reported with its line.
        """
        from .names import NameCheck
        NameCheck(self).check()
        return self

    # ------------------------------------------------------------- entry
    def run(self, uniforms, inputs, n):
        n = int(n)
        # the root is the scope every function body starts from (call_user)
        # and the scope a global's initialiser is evaluated in: set before
        # the first initialiser runs, so it never reads a previous run's
        self.root = root = Scope()
        # gl_FragCoord is a builtin, in scope without a declaration -- as it
        # is in GLSL. Bound first so a declared `in vec4 gl_FragCoord;`
        # (redundant but accepted) takes the same value over it
        fc_t = GType('float', 4)
        fc = inputs.get('fragcoord') if inputs else None
        root.put('gl_FragCoord',
                 self.adapt(fc if fc is not None
                            else np.zeros((n, 4), np.float32), fc_t, n), fc_t)
        from .codegen import VARYINGS
        for role, (name, t, init, _q) in self.order:
            if role == 'uniform':
                v = uniforms.get(name)
                if v is None:
                    v = self.const(init, t, n, root) if init is not None \
                        else self.zero(t, n)
                root.put(name, self.adapt(v, t, n), t)
            elif role == 'input':
                # a varying can be supplied under its declared name or under
                # the binding it maps to, and the generator accepted either
                v = inputs.get(name)
                if v is None:
                    bound = VARYINGS.get(name)
                    if bound is not None:
                        v = inputs.get(bound[0])
                if v is None:
                    v = self.varying(name, t, n)
                root.put(name, self.adapt(v, t, n), t)
            elif role == 'global':
                root.put(name, self.const(init, t, n, root)
                         if init is not None else self.zero(t, n), t)
            else:
                root.put(name, self.zero(t, n), t)

        entry = self.entry
        if entry is None:
            raise ShaderError('no main() in shader')

        flags = Flags(n)
        mask = np.ones(n, bool)
        # the entry is not called by any expression, so its parameters are
        # bound here: an `in` reads the renderer's input of its name, an
        # `out` / `inout` starts at zero and is an output of the program.
        # (It was called with no arguments at all: every parameter read
        # zero and nothing an `out` parameter was given left the call, so
        # `void mainImage(out vec4 fragColor, in vec2 fragCoord)` compiled
        # into a program with no outputs.)
        values = [self.zero(param_type(p), n) if p[0] in ('out', 'inout')
                  else self.entry_input(p, inputs, n) for p in entry[3]]
        result, _rt, inner = self.invoke(entry, values, mask, flags, n)

        outs = {}
        for name, t, _i, _q in self.outputs:
            slot = root.get(name)
            outs[name] = slot[0] if slot else self.zero(t, n)
        for p in entry[3]:
            # over a global `out` of the same name: inside the entry the
            # parameter is the one the shader writes
            if p[0] in ('out', 'inout') and p[2]:
                outs[p[2]] = inner.get(p[2])[0]
        if entry[1] != VOID:
            outs['__return'] = result
        return outs, flags.discard

    def entry_input(self, p, inputs, n):
        """The value an `in` parameter of the entry point starts with."""
        from .codegen import param_binding
        name, t = p[2], param_type(p)
        v = inputs.get(name) if name else None
        if v is None and name:
            key = param_binding(name, t)
            v = inputs.get(key)
            if v is None and key == 'fragcoord2':
                # a caller that hands over gl_FragCoord alone: its xy
                v = inputs.get('fragcoord')
        if v is None:
            return self.zero(t, n)
        if t.base in ('struct', 'sampler') or t.array or t.is_matrix:
            return v
        v = np.asarray(rt.bc(v, n))
        if t.n > 1:
            # the parameter's width decides, not the binding's: a vec2 of
            # the fragment coordinate is its xy, a vec3 of the vertex
            # colour its rgb, and a scalar fills the vector
            v = rt.vec(t.n, v)
        elif v.ndim > 1:
            v = rt.sw(v, [0])
        if t.base == 'float':
            return rt.to_float(v)
        kind = v.dtype.kind
        return self.coerce(v, GType('bool' if kind == 'b' else 'float'
                                    if kind == 'f' else 'int', t.n), t, n)

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
        if k == 'bool':
            # R249: the true / false literals (the parser always made
            # them; the generator handled them; the interpreter had no
            # case, so `bool x = false;` was an unsupported expression)
            return np.full(n, bool(node[1]), bool), BOOL
        if k == 'var':
            slot = scope.get(node[1])
            if slot is None:
                raise ShaderError(f'unknown identifier {node[1]!r}')
            return slot[0], slot[1]
        if k == 'held':
            # an index `hold` evaluated ahead of the write it belongs to
            return node[1], node[2]
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
        if k == 'seq':
            # the comma operator: the left side for what it writes, the
            # right side for the value (`for (...; i++, b += 1.0)`). The
            # parser has always built the node; the interpreter had no case
            self.expr(node[1], mask, scope, flags, n)
            return self.expr(node[2], mask, scope, flags, n)
        if k == 'arrinit':
            et = BY_NAME.get(node[1])
            if et is None:
                raise ShaderError(f'unknown array element type {node[1]!r}')
            vals = []
            for arg in node[2]:
                v, vt = self.expr(arg, mask, scope, flags, n)
                vals.append(self.coerce(v, vt, et, n))
            return vals, GType(et.base, et.n, et.rows, len(vals), et.struct)
        if k == 'call':
            return self.call(node, mask, scope, flags, n)
        if k == 'member':
            base, bt = self.expr(node[1], mask, scope, flags, n)
            field = node[2]
            if bt.base == 'struct':
                fields = self.fields(bt.struct)
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
            op, target = node[1], self.hold(node[2], mask, scope, flags, n)
            cur, t = self.expr(target, mask, scope, flags, n)
            delta = np.ones(n, np.float32) * (1.0 if op == '++' else -1.0)
            newv = self.coerce(cur + self.coerce(delta, FLOAT, t, n), t, t, n)
            self.store(target, newv, t, mask, scope, flags, n)
            return (cur if k == 'post' else newv), t
        if k == 'un':
            op, operand = node[1], node[2]
            if op in ('++', '--'):
                # ++x is a write as well as a read, so it shares the post path
                operand = self.hold(operand, mask, scope, flags, n)
                cur, t = self.expr(operand, mask, scope, flags, n)
                delta = np.ones(n, np.float32) * (1.0 if op == '++' else -1.0)
                newv = self.coerce(cur + self.coerce(delta, FLOAT, t, n),
                                   t, t, n)
                self.store(operand, newv, t, mask, scope, flags, n)
                return newv, t
            v, t = self.expr(operand, mask, scope, flags, n)
            if op == '-':
                if t.base == 'uint':
                    # unsigned negation wraps, exactly as on the GPU
                    return (-np.asarray(v, np.int64)) & 0xffffffff, t
                return -v, t
            if op == '+':
                return v, t
            if op == '!':
                return ~rt.to_bool(v), BOOL
            if op == '~':
                if t.base == 'uint':
                    return (~np.asarray(v, np.int64)) & 0xffffffff, t
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
        if op == '^^':
            # the one operator the parser knew and this did not (it fell
            # through to 'unsupported operator' when a lane reached it)
            return rt.to_bool(a) ^ rt.to_bool(b), BOOL
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
            if t.base == 'int':
                # two signed integers divide to an integer, cut toward
                # zero: 7 / 2 is 3 and -7 / 2 is -3, where the quotient was
                # kept whole (3.5, typed int) -- on a driver, which compiles
                # the same text when the shader is inlined in the deferred
                # pass, it never was. `uint` is cut below, as it always
                # was. The cut is ALL that changes: the quotient is held as
                # before (NumPy's float64, exact here for any two 32-bit
                # operands), so one that had no fraction is the same number
                # in the same dtype and nothing computed from it moves by a
                # bit.
                r = np.trunc(r)
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
            if t.base == 'uint':
                # logical shift: the operand is already masked non-negative
                r = np.asarray(x, np.int64) >> np.asarray(y, np.int64)
            else:
                r = rt.to_int(x) >> rt.to_int(y)
        else:
            raise ShaderError(f'unsupported operator {op!r}')
        if t.base == 'uint':
            # uint arithmetic wraps at 2^32, exactly as the driver's does.
            # Values live in int64 so products are exact before the mask --
            # and masking to 32 bits commutes with masking any narrower
            # window, which is what makes hash functions portable at all
            r = np.asarray(r)
            if r.dtype.kind == 'f':
                r = r.astype(np.int64)         # uint division truncates
            r = r.astype(np.int64) & 0xffffffff
        return r, t

    def assign(self, node, mask, scope, flags, n):
        op, target, value = node[1], node[2], node[3]
        v, vt = self.expr(value, mask, scope, flags, n)
        if op != '=':
            # the target's indices once, for the read and the write alike
            # (`a[i++] += x` names one element)
            target = self.hold(target, mask, scope, flags, n)
            cur, ct = self.expr(target, mask, scope, flags, n)
            v, vt = self.binary(('bin', op[:-1], target, value), mask, scope,
                                flags, n)
        return self.store(target, v, vt, mask, scope, flags, n)

    def hold(self, target, mask, scope, flags, n):
        """The same assignment target with every index it is reached
        through evaluated -- once, here.

        A write through `a[i].x` reads `a[i]`, changes the component and
        writes `a[i]` back (`store`), and `a[i] += x` reads the element
        before it writes it: each step names the element again. With a
        number written in the brackets that costs nothing, and it was the
        only index a write accepted. An expression is evaluated here and
        its value kept in the node, so every step names the element the
        first one did -- `a[i++] = x` moves `i` once."""
        k = target[0]
        if k == 'member':
            inner = self.hold(target[1], mask, scope, flags, n)
            if inner is target[1]:
                return target
            return ('member', inner) + tuple(target[2:])
        if k != 'index':
            return target
        inner = self.hold(target[1], mask, scope, flags, n)
        idx = target[2]
        if idx[0] not in ('num', 'held'):
            idx = ('held',) + tuple(self.expr(idx, mask, scope, flags, n))
        if inner is target[1] and idx is target[2]:
            return target
        return ('index', inner, idx)

    def store(self, target, value, vtype, mask, scope, flags, n):
        k = target[0]
        if k in ('member', 'index'):
            # every index on the way to the target, once (`hold`). A caller
            # that read the target before writing it -- a compound
            # assignment, ++, an out argument -- held it then, and this
            # answers that same node
            target = self.hold(target, mask, scope, flags, n)
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
                fields = self.fields(bt.struct)
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
            i, _it = self.expr(target[2], mask, scope, flags, n)
            everyone = np.ones(n, bool)
            if bt.array:
                # one element per lane: the lanes that index element j are
                # the ones it is written for
                items = list(base)
                for j, hit in self.indexed(target[2], i, len(items), bt,
                                           mask):
                    items[j] = rt.sel(hit, value, items[j])
                self.store(target[1], items, bt, everyone, scope, flags, n)
                return self.subscript(items, bt, i, n)
            if bt.is_matrix:
                # a column of the one array a matrix is (it was a Python
                # list of columns here, and an array where it was built)
                ct = GType(bt.base, bt.rows)
                column = self.coerce(value, vtype, ct, n)
                m = base
                for j, hit in self.indexed(target[2], i, bt.n, bt, mask):
                    m = rt.sel(hit, rt.col_set(base, j, column), m)
                self.store(target[1], m, bt, everyone, scope, flags, n)
                return self.subscript(m, bt, i, n)
            updated = base
            for j, hit in self.indexed(target[2], i, bt.n, bt, mask):
                updated = rt.sel(hit, rt.sw_set(base, [j], value), updated)
            self.store(target[1], updated, bt, everyone, scope, flags, n)
            return base, bt
        raise ShaderError(f'cannot assign to {k!r}')

    def indexed(self, node, i, count, bt, mask):
        """What a write through an index reaches: (element, the lanes of
        `mask` that write it), one pair per element some lane writes.
        `node` is the index as written, `i` its value, `count` the elements
        a `bt` holds.

        A read through a variable index has always chosen per lane
        (`subscript`); a write accepted a written number alone and raised
        `dynamic array writes are not supported` on `a[i] = x` inside the
        loop that fills an array. This is the write's half of the same
        rule, and the index is clamped into range as the read's is, so
        what is written through an index is what a read through it finds.

        Two writes stay refused, as they were. A NUMBER written past the
        size (`a[3] = x` of a `float a[3]`) raised an IndexError and no
        driver compiles it: it is said by name. And a variable index into
        what holds no elements -- a float -- has nothing to choose
        between: clamping it would send every write to the one value in
        silence."""
        count = max(int(count), 1)
        const = self.same_index(i)
        if node[0] == 'num':
            if const is None or not 0 <= const < count:
                raise ShaderError(f'index {node[1]} is out of range for '
                                  f'{bt.name()}')
            return [(const, mask)]
        if not (bt.array or bt.is_matrix or bt.n > 1):
            raise ShaderError(f'{bt.name()} is not an array, a vector or a '
                              'matrix: a variable index cannot write to it')
        if const is not None:
            return [(max(0, min(const, count - 1)), mask)]
        idx = np.clip(np.reshape(rt.to_int(i), (-1,)).astype(np.int64),
                      0, count - 1)
        out = []
        for j in range(count):
            hit = mask & (idx == j)
            if hit.any():
                out.append((j, hit))
        return out

    @staticmethod
    def same_index(i):
        """The index every lane holds, or None when the lanes differ."""
        try:
            arr = np.asarray(i).reshape(-1)
            if arr.size and np.all(arr == arr[0]):
                return int(arr[0])
        except Exception:                                       # noqa: BLE001
            pass
        return None

    # -------------------------------------------------------------- calls
    def pick(self, name, argc):
        """The function of that name a call with `argc` arguments reaches,
        or None. It used to answer the first function of the name when no
        count fitted, and the call went through with the extra arguments
        dropped and the missing ones zero: the wrong count is refused at
        compile time now (`wrong_count`, raised from shaders/names.py)."""
        for lo, hi, f in self.overloads.get(name, ()):
            if lo <= argc <= hi:
                return f
        return None

    def wrong_count(self, name, argc):
        """The message for a call no function, constructor or builtin of
        that name takes -- or None when the count is fine. One wording for
        the three, and one place that knows the counts: the name check asks
        before the shader runs, `call` below is the backstop."""
        fields = self.structs.get(name)
        if fields is not None:
            if argc == len(fields):
                return None
            want = count_text([(len(fields), len(fields))])
            return f'{name}() takes {want} (one per field), got {argc}'
        if name in BY_NAME:
            return None                 # vec3(...), float(...): not judged
        cands = self.overloads.get(name)
        if cands:
            spans = [c[:2] for c in cands]
        else:
            impl = BUILTINS.get(name)
            if impl is None:
                return None             # an unknown function, said elsewhere
            spans = [(impl[2], impl[3])]
        if any(lo <= argc <= hi for lo, hi in spans):
            return None
        return f'{name}() takes {count_text(spans)}, got {argc}'

    def call(self, node, mask, scope, flags, n):
        name, args = node[1], node[2]
        ctor = self.constructor(name, args, mask, scope, flags, n)
        if ctor is not None:
            return ctor
        fn = self.pick(name, len(args))
        if fn is not None:
            return self.call_user(fn, args, mask, scope, flags, n)
        # a shader's own function hides the builtin of its name, whatever
        # the count -- as on a driver
        impl = None if name in self.funcs else BUILTINS.get(name)
        if impl is None or not (impl[2] <= len(args) <= impl[3]):
            raise ShaderError(self.wrong_count(name, len(args))
                              or f'unknown function {name!r}')
        # (runtime name, return-type rule, min args, max args)
        rt_name, rule, lo, hi = impl
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
        values, writeback = [], []
        for i, p in enumerate(fn[3]):
            ptype = param_type(p)
            if i < len(args):
                arg = args[i]
                if p[0] in ('out', 'inout'):
                    # the element the call names is the one it is written
                    # back to, whatever the call does to the index
                    arg = self.hold(arg, mask, scope, flags, n)
                    writeback.append((arg, p[2], ptype))
                v, vt = self.expr(arg, mask, scope, flags, n)
                v = self.coerce(v, vt, ptype, n)
            else:
                # the call left the parameter out, which the count check
                # allows for one declared with a default alone: the default
                # is read where it was written, among the globals
                v, vt = self.expr(param_default(p), mask, self.root, flags, n)
                v = self.coerce(v, vt, ptype, n)
            values.append(v)
        result, rtype, inner = self.invoke(fn, values, mask, flags, n)
        for target, pname, ptype in writeback:
            slot = inner.get(pname)
            if slot is not None:
                self.store(target, slot[0], ptype, mask, scope, flags, n)
        return result, rtype

    def invoke(self, fn, values, mask, flags, n):
        """Run one function over values already bound to its parameters.
        Returns (result, type, the scope the body ran in) -- the scope is
        where a caller finds what an `out` parameter was left holding."""
        _k, rtype, name, params, body = fn
        # a function body starts from the root's own slots: every global the
        # shader declares AND the builtin the root binds (gl_FragCoord used
        # to stop at the root -- declared names were copied one by one, so
        # the builtin was in scope everywhere except inside a function,
        # which is everywhere a shader can read it)
        inner = Scope()
        if self.root is not None:
            inner.vars.update(self.root.vars)
        for p, v in zip(params, values):
            inner.put(p[2], v, param_type(p))

        saved = (self.ret_type, self.ret_val)
        self.ret_type = None if rtype == VOID else rtype
        self.ret_val = self.zero(rtype, n) if rtype != VOID else None
        sub = Flags(n)
        sub.discard = flags.discard
        self.block(body, mask, inner, sub, n)
        flags.discard = flags.discard | sub.discard
        result = self.ret_val
        self.ret_type, self.ret_val = saved
        return (result if result is not None else np.zeros(n, np.float32),
                rtype if rtype != VOID else FLOAT, inner)

    def constructor(self, name, args, mask, scope, flags, n):
        fields = self.structs.get(name)
        if fields is not None:
            # S(a, b): one argument per field, in the order they are
            # declared, each converted to its field's type. The struct is
            # looked up in THIS shader's own table and before the builtin
            # types -- the parser used to register it among those, where
            # the scalar path below found it first and answered `a`
            if len(args) != len(fields):
                raise ShaderError(self.wrong_count(name, len(args)))
            vals = [self.expr(a, mask, scope, flags, n) for a in args]
            out = {}
            for (fname, ftype), (v, vt) in zip(self.fields(name).items(),
                                               vals):
                out[fname] = self.coerce(v, vt, ftype, n)
            return out, struct_type(name)
        t = BY_NAME.get(name)
        if t is None:
            return None
        parts = []
        ptypes = []
        for a in args:
            v, vt = self.expr(a, mask, scope, flags, n)
            parts.append(v)
            ptypes.append(vt)
        if t.is_matrix:
            return rt.mat(t.n, t.rows, *parts), t
        if t.n == 1:
            # R243: the argument's REAL type, so float(u) of a uint hash
            # converts; assuming float here left the integer in place and
            # the division after it ran in float64 -- a precision no GPU
            # has, and a twin drift the tiles map finally showed
            v = parts[0] if parts else np.zeros(n, np.float32)
            src = ptypes[0] if ptypes else FLOAT
            return self.coerce(v, src, t, n), t
        v = rt.vec(t.n, *parts)
        if t.base in ('int', 'uint'):
            v = rt.to_int(v)
        elif t.base == 'bool':
            v = rt.to_bool(v)
        return v, t

    # ------------------------------------------------------------ helpers
    def subscript(self, base, bt, i, n):
        const = self.same_index(i)
        if bt.array:
            if const is not None:
                return base[max(0, min(const, len(base) - 1))], bt.elem()
            idx = np.clip(np.reshape(rt.to_int(i), (-1,)).astype(np.int64),
                          0, len(base) - 1)
            # constant-element arrays (lookup tables, unrolled colorbands)
            # gather in one indexing op instead of a 512-way select chain
            first = base[0]
            if all(np.ndim(e) == np.ndim(first)
                   and np.shape(e) == np.shape(first) for e in base):
                if np.ndim(first) == 0:
                    return np.asarray(base)[idx], bt.elem()
                if np.ndim(first) == 1 and bt.n > 1 \
                        and np.shape(first)[0] == bt.n:
                    return np.asarray(base)[idx], bt.elem()
            out = base[0]
            for j in range(1, len(base)):
                out = rt.sel(idx == j, base[j], out)
            return out, bt.elem()
        if bt.is_matrix:
            # the column axis of (N, C, R), whichever way the matrix was
            # made; a variable index chooses a column per lane, as it
            # chooses an element of an array (it read column 0)
            ct = GType(bt.base, bt.rows)
            if const is not None:
                return rt.col(base, max(0, min(const, bt.n - 1))), ct
            return rt.idx_axis(base, np.reshape(rt.to_int(i), (-1,))), ct
        if const is not None:
            return rt.sw(base, [max(0, min(const, bt.n - 1))]), GType(bt.base, 1)
        # clamped at both ends, as the line above clamps the index every
        # lane agrees on and as an array's is: past the last component the
        # lanes that DIFFERED read component 0, so one index read two
        # components depending on its neighbours -- and a write through it
        # (`indexed`) has to land where the read looks
        idx = np.clip(np.reshape(rt.to_int(i), (-1,)), 0, bt.n - 1)
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
        if dst.base == 'struct' or dst.array or dst.is_matrix:
            return value
        if src is dst:
            # the same declared type can still carry the wrong dtype (a
            # float64 that slipped in through numpy promotion): a float
            # destination is float32, full stop
            if dst.base == 'float' and np.asarray(value).dtype != np.float32:
                return rt.to_float(value)
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
        elif dst.base == 'uint' and src.base != 'uint':
            # uint(x) wraps: uint(-7) is 4294967289 on every driver
            if src.base not in ('int', 'bool'):
                v = rt.to_int(v)
            v = np.asarray(v).astype(np.int64) & 0xffffffff
        elif dst.base == 'int' and src.base == 'uint':
            # int(u) keeps the bit pattern, as on the GPU
            v = np.asarray(v).astype(np.int64).astype(np.uint32).astype(np.int32)
        elif dst.base == 'int' and src.base not in ('int', 'uint'):
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
            return {name: self.zero(ft, n)
                    for name, ft in self.fields(t.struct).items()}
        if t.is_matrix:
            # the shape the constructor builds and the runtime multiplies
            return np.zeros((n, t.n, t.rows), np.float32)
        if t.n == 1:
            return np.zeros(n, bool if t.base == 'bool' else
                            np.int64 if t.base == 'uint' else
                            np.int32 if t.base == 'int' else np.float32)
        return np.zeros((n, t.n), np.float32)

    def fields(self, struct):
        """{field name: type} of one struct, in the order it declares them
        -- each type with its array size (`field_type`)."""
        known = self.field_types.get(struct)
        if known is None:
            known = self.field_types[struct] = {
                f[1]: field_type(f) for f in self.structs.get(struct, ())}
        return known

    def number(self, node, n):
        raw = node[1]
        is_hex = raw[:2].lower() == '0x'
        # hex digits include f: stripping float suffixes from a hex literal
        # ate them ('0x7fffffff' -> '0x7'), which is how 0x7fffffffu once
        # read as 7.0. Hex strips only integer suffixes.
        text = raw.rstrip('uUlL') if is_hex else raw.rstrip('uUfFhHlL')
        is_uint = raw.rstrip('lL')[-1:] in ('u', 'U')
        try:
            if not is_hex and (any(c in text for c in '.eE') or not node[2]):
                val = float(text)
                return np.full(n, val, np.float32), FLOAT
            val = int(text, 0)
            if is_uint:
                return np.full(n, val & 0xffffffff, np.int64), UINT
            if val > 2**31 - 1 or val < -2**31:
                val &= 0xffffffff              # int literals wrap, as GLSL's
                if val >= 2**31:
                    val -= 2**32
            return np.full(n, val, np.int32), INT
        except ValueError:
            return np.zeros(n, np.float32), FLOAT

    def const(self, init, t, n, scope=None):
        """A global's initialiser, evaluated in the root as it stands when
        the declaration is reached: `const float TAU = 2.0 * PI;` reads the
        PI above it. (It was evaluated in an empty scope, so any name in it
        raised here, was caught below and the global read zero without a
        word.)"""
        try:
            v, vt = self.expr(init, np.ones(n, bool),
                              scope if scope is not None else Scope(),
                              Flags(n), n)
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
        if t.base == 'float' and np.asarray(arr).dtype != np.float32:
            arr = rt.to_float(arr)
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
        rather than by a generated variable. The declared `out` globals,
        then what the entry point hands back: its `out` / `inout`
        parameters and the value it returns."""
        outs = [(name, name, t) for name, t, _i, _q in self.outputs]
        fn = self.entry
        if fn is not None:
            seen = {o[0] for o in outs}
            for p in fn[3]:
                if p[0] in ('out', 'inout') and p[2] and p[2] not in seen:
                    seen.add(p[2])
                    outs.append((p[2], p[2], param_type(p)))
            if fn[1] != VOID:
                outs.append(('__return', '__return', fn[1]))
        return outs

    root = None
    ret_type = None
    ret_val = None

