"""Name resolution at compile time -- and what else the run would refuse.

The interpreter resolves a name when a lane reaches it: an identifier
nothing declares, a field a struct does not have, a swizzle past the
vector's width all compiled, and failed only when the shader ran -- or
never, when the line sat in a branch no lane took or in a function nothing
called, which is the text a driver refuses whole. The source generator this
interpreter replaced caught every one of them while it emitted code
(`undeclared identifier`, `has no field`, `bad swizzle`, `swizzle out of
range`); `Interpreter.validate` brought back the unknown-function half and
this module is the rest.

Three more refusals ride the same walk, because it already knows every
call and the type of every expression:

  counts     a call must pass a count of arguments its function takes. The
             generator refused the wrong count for a shader's own function
             (R235); the interpreter dropped the extra arguments and read
             the missing ones as zero. A builtin with the wrong count
             raised when a lane reached it, a struct constructor never
             ran at all. One message for the three
             (`Interpreter.wrong_count`).
  widths     two vectors of different widths cannot be combined (`vec2 +
             vec3`): `gtypes.promote` raised when the operator ran. The
             types here are the interpreter's own, rule for rule, so the
             operator is refused exactly where the run would raise. Where
             the interpreter's type is not the language's -- a matrix
             times a scalar (typed as a vector), two functions of one
             name and argument count (the first is taken, whatever the
             argument types) -- the type is kept, marked `_Loose`, and no
             width is judged on it or on anything computed from it: a
             text a driver compiles is not refused on this front-end's
             own shortcut. (Not modelled: the type table has one sampler
             type, so `texture()` of a sampler2DShadow is a vec4 here as
             it is when the shader runs.)
  members    a field or a swizzle is taken from a struct, a vector or a
             scalar -- never from a matrix or an array, which is indexed
             first (`m[0].xyz`). The swizzle check used to step around
             both, and `m.xyz` raised from NumPy when a lane reached it. A
             struct's field is typed with its array size (`float w[3]` is
             an array of three), by the same `field_type` the interpreter
             builds the struct from.

One thing a shader names is not in the tree this walks: the size of an
array. A size is a number by the time the parser hands the tree over -- it
is a constant expression, and the parser computes it where it stands
(parser.Parser.array_size), resolving its names in scopes of its own that
follow the rules below. What this walk refuses in an expression it refuses
in a size too, in the same words: `check_constant`, at the foot of this
module, is the walk over that one expression, called by the parser before
it computes it. (A size was read as a number or not at all, so `float
a[UNDECLARED];` compiled.)

It is a walk of the same tree with the scopes `Interpreter.run`,
`call_user`, `block` and `loop` build, holding a type where they hold a
value. The two must agree on what is in scope where, so the rules are
written once each, side by side:

  root       gl_FragCoord, then every global declaration in source order
             (`Interpreter.order`). A uniform's or a plain global's
             initialiser sees what is declared above it.
  function   every global wherever it is written, then the parameters; the
             body is a block of its own.
  block      a scope of its own; a declaration is visible from the
             statement after it (its initialiser still sees the outer
             name). An if / else arm is a block even without braces.
  loop       one scope for the init, the condition and the step; the body
             is a block inside it.

A varying is in scope when the shader declares it (`in vec3 vNormal;`), as
on a driver: `codegen.VARYINGS` is the table of what such a declaration
binds to, not a list of names that exist without one. The error for one of
them says what to write.
"""

from .builtins import BUILTINS
from .codegen import FRAG_OUTS, VARYINGS
from .gtypes import (BOOL, BY_NAME, FLOAT, INT, UINT, VOID, GType, promote,
                     swizzle_indices)
from .interp import field_type, param_default, param_type, struct_type
from .lexer import ShaderError

#: in scope with no declaration, as GLSL defines it (Interpreter.run binds
#: the same name into the root)
BUILTIN_VARS = {'gl_FragCoord': GType('float', 4)}

#: a value nothing is known about: its names were checked, and a field,
#: a swizzle or an operator on it is not judged
UNKNOWN = (None, 0)

_COMPARES = ('==', '!=', '<', '>', '<=', '>=')


class _Loose(GType):
    """The interpreter's type for a value whose type in the language may be
    another. Every check that asks what the RUN will do reads it as the
    plain type it is; the width check alone steps around it."""
    __slots__ = ()


def _loose(t, *parents):
    """`t`, marked loose when any type it was computed from is."""
    if t is None or isinstance(t, _Loose) or \
            not any(isinstance(p, _Loose) for p in parents):
        return t
    return _Loose(t.base, t.n, t.rows, t.array, t.struct)


def _mark(t):
    return _Loose(t.base, t.n, t.rows, t.array, t.struct)


class _Scope:
    __slots__ = ('names', 'parent')

    def __init__(self, parent=None):
        self.names = {}
        self.parent = parent

    def get(self, name):
        s = self
        while s is not None:
            t = s.names.get(name)
            if t is not None:
                return t
            s = s.parent
        return None


def _line(node, at):
    """The line a name node carries (parser.py), or 0 for a tree built
    without one."""
    v = node[at] if len(node) > at else 0
    return v if isinstance(v, int) else 0


def _hint(name):
    bound = VARYINGS.get(name)
    if bound is not None:
        return f' (declare it: in {bound[1].name()} {name};)'
    if name in FRAG_OUTS:
        return f' (declare it: out vec4 {name};)'
    return ''


class NameCheck:
    """Every name of one shader, resolved before it runs.

    A value is (type, width): the type by the interpreter's own rules, and
    the widest the value can be at run time. The two differ only where the
    runtime is more forgiving than its types -- `pow(2.0, v)` is typed by
    its first argument and computed across the vector -- so a swizzle is
    refused as out of range only when it is past that width, which is past
    anything the run could index.
    """

    def __init__(self, interp):
        self.it = interp
        self.structs = interp.structs
        self.funcs = interp.funcs

    # ------------------------------------------------------------- entry
    def check(self):
        it = self.it
        root = _Scope()
        root.names.update(BUILTIN_VARS)
        for role, (name, t, init, _q) in it.order:
            # the two roles whose initialiser the run evaluates
            if init is not None and role in ('uniform', 'global'):
                self.expr(init, root)
            root.names[name] = t
        for cands in self.funcs.values():
            for fn in cands:
                inner = _Scope(root)
                for p in fn[3]:
                    if param_default(p) is not None:
                        # read among the globals, where the run reads it
                        self.expr(param_default(p), root)
                    inner.names[p[2]] = param_type(p)
                self.block(fn[4], inner)

    # -------------------------------------------------------- statements
    def block(self, node, scope):
        inner = _Scope(scope)
        for st in (node[1] if node[0] == 'block' else [node]):
            self.stmt(st, inner)

    def stmt(self, node, scope):
        k = node[0]
        if k == 'block':
            self.block(node, scope)
        elif k == 'expr':
            self.expr(node[1], scope)
        elif k == 'decl':
            gtype = node[1]
            for name, arr, init in node[2]:
                if init is not None:
                    self.expr(init, scope)
                scope.names[name] = GType(gtype.base, gtype.n, gtype.rows,
                                          arr, gtype.struct) if arr else gtype
        elif k == 'if':
            self.expr(node[1], scope)
            self.block(node[2], scope)
            if node[3] is not None:
                self.block(node[3], scope)
        elif k in ('for', 'while', 'do'):
            sc = _Scope(scope)
            if k == 'for':
                init, cond, step, body = node[1], node[2], node[3], node[4]
                if init and init[0] != 'nop':
                    self.stmt(init, sc)
            elif k == 'while':
                cond, body, step = node[1], node[2], None
            else:
                body, cond, step = node[1], node[2], None
            if cond is not None and k != 'do':
                self.expr(cond, sc)
            self.block(body, sc)
            if cond is not None and k == 'do':
                self.expr(cond, sc)
            if step is not None and step[0] != 'nop':
                self.expr(step, sc)
        elif k == 'return':
            if node[1] is not None:
                self.expr(node[1], scope)

    # ------------------------------------------------------- expressions
    def expr(self, node, scope):
        """(type, width) of an expression, raising on a name that does not
        resolve. UNKNOWN wherever the type is not this module's business."""
        k = node[0]
        if k == 'num':
            if not node[2]:
                return FLOAT, 1
            return (UINT if node[1].rstrip('lL')[-1:] in ('u', 'U')
                    else INT), 1
        if k == 'bool':
            return BOOL, 1
        if k == 'var':
            t = scope.get(node[1])
            if t is None:
                raise ShaderError(f'undeclared identifier {node[1]!r}'
                                  + _hint(node[1]), _line(node, 2))
            return t, t.n
        if k == 'member':
            bt, bw = self.expr(node[1], scope)
            field = node[2]
            if bt is None:
                return UNKNOWN
            if (bt.array or bt.is_matrix) and not isinstance(bt, _Loose):
                # an array or a matrix is indexed before anything is taken
                # from it: `m.xyz`, `a.x`, `a.field` are no driver's GLSL.
                # (The swizzle check below stepped around both, and the run
                # then swizzled the lane axis: a NumPy error, or the wrong
                # numbers when the shapes happened to fit. A loose type is
                # left to the run, as everywhere: it may not be the
                # language's.)
                what = 'an array' if bt.array else 'a matrix'
                eg = f' ({node[1][1]}[0].{field})' \
                    if node[1][0] == 'var' else ''
                raise ShaderError(f'{field!r} of {bt.name()}: {what} is '
                                  f'indexed first{eg}', _line(node, 3))
            if bt.base == 'struct':
                for f in self.structs.get(bt.struct, ()):
                    if f[1] == field:
                        # with its array size: `float w[3]` is an array
                        ftype = field_type(f)
                        return _loose(ftype, bt), ftype.n
                raise ShaderError(f'{bt.struct} has no field {field!r}',
                                  _line(node, 3))
            idx = swizzle_indices(field)
            if idx is None:
                raise ShaderError(f'bad swizzle {field!r}', _line(node, 3))
            if not bt.array and not bt.is_matrix and \
                    bt.base in ('float', 'int', 'uint', 'bool') and \
                    max(idx) >= max(bw, 1):
                raise ShaderError(f'swizzle {field!r} out of range for '
                                  f'{bt.name()}', _line(node, 3))
            return GType(bt.base, len(idx)), len(idx)
        if k == 'index':
            bt, _bw = self.expr(node[1], scope)
            self.expr(node[2], scope)
            if bt is None:
                return UNKNOWN
            if bt.array:
                et = bt.elem()
                return _loose(et, bt), et.n
            if bt.is_matrix:
                return _loose(GType(bt.base, bt.rows), bt), bt.rows
            return _loose(GType(bt.base, 1), bt), 1
        if k == 'call':
            return self.call(node, scope)
        if k == 'bin':
            a, b = self.expr(node[2], scope), self.expr(node[3], scope)
            return self.binary(node[1], a, b, _line(node, 4))
        if k == 'assign':
            v = self.expr(node[3], scope)
            tt, tw = self.expr(node[2], scope)
            if node[1] != '=':
                # `a += b` runs as `a + b` (Interpreter.assign), so it
                # combines the two the same way
                self.binary(node[1][:-1], (tt, tw), v, _line(node, 4))
            # the interpreter answers a plain variable's own type; what a
            # field, swizzle or element write answers is its own affair
            return (tt, tw) if node[2][0] == 'var' else UNKNOWN
        if k == 'cond':
            self.expr(node[1], scope)
            (at, aw), (bt, bw) = (self.expr(node[2], scope),
                                  self.expr(node[3], scope))
            if at is None or bt is None:
                return UNKNOWN
            return self._promote(at, bt, max(aw, bw), _line(node, 4))
        if k in ('post', 'pre'):
            return self.expr(node[2], scope)
        if k == 'un':
            t, w = self.expr(node[2], scope)
            if t is None:
                return UNKNOWN
            if node[1] == '!':
                return BOOL, w
            if node[1] == '~':
                return (t if t.base == 'uint' else INT), w
            return t, w
        if k == 'arrinit':
            for arg in node[2]:
                self.expr(arg, scope)
            et = BY_NAME.get(node[1])
            if et is None:
                return UNKNOWN
            t = GType(et.base, et.n, et.rows, len(node[2]), et.struct)
            return t, t.n
        if k == 'seq':
            self.expr(node[1], scope)
            return self.expr(node[2], scope)
        return UNKNOWN

    def binary(self, op, a, b, line):
        """(type, width) of `a op b` by Interpreter.binary's rules, refusing
        the two widths its promote would refuse."""
        (at, aw), (bt, bw) = a, b
        if op in ('&&', 'and', '||', 'or', '^^'):
            return BOOL, 1
        if at is None or bt is None:
            return (BOOL, 1) if op in _COMPARES else UNKNOWN
        if op in _COMPARES:
            # compared through the same promote as the arithmetic
            self._promote(at, bt, 1, line)
            return BOOL, 1
        if op == '*' and (at.is_matrix or bt.is_matrix):
            if at.is_matrix and bt.is_matrix:
                return _loose(at, bt), at.n
            if at.is_matrix:
                t, w = GType(at.base, at.rows), max(at.rows, bw)
            else:
                t, w = GType(bt.base, bt.n), max(bt.n, aw)
            other = bt if at.is_matrix else at
            if other.n == 1:
                # matrix * scalar: a vector by the interpreter's rule (which
                # cannot run it), a matrix in the language
                return _mark(t), w
            return _loose(t, at, bt), w
        return self._promote(at, bt, max(aw, bw), line)

    @staticmethod
    def _promote(at, bt, width, line=0):
        if at.base == 'struct' or bt.base == 'struct' or at.array or bt.array:
            return UNKNOWN
        try:
            return _loose(promote(at, bt), at, bt), width
        except ShaderError as exc:
            if isinstance(at, _Loose) or isinstance(bt, _Loose):
                return UNKNOWN      # not this walk's to judge; the run's
            # promote's own refusal, with the line of the operator: this is
            # where the run raised it, when a lane got that far
            raise ShaderError(exc.msg, line)

    def call(self, node, scope):
        name, args = node[1], node[2]
        vals = []
        for a in args:              # a loop, not a comprehension: the walk
            vals.append(self.expr(a, scope))    # is never deeper than the run
        line = _line(node, 3)
        bad = self.it.wrong_count(name, len(args))
        if bad:
            raise ShaderError(bad, line)
        if name in self.structs:                # a constructor, one argument
            return struct_type(name), 1         # per field
        t = BY_NAME.get(name)
        if t is not None:                       # a constructor
            return t, t.n
        cands = self.funcs.get(name)
        if cands:
            fn = self.it.pick(name, len(args))
            rtype = fn[1] if fn[1] != VOID else FLOAT
            fits = [f for lo, hi, f in self.it.overloads[name]
                    if lo <= len(args) <= hi]
            if any(f[1] != fn[1] for f in fits):
                # two functions of this name take this many arguments and
                # answer different types: the run calls the first, a driver
                # the one the argument types choose
                return _mark(rtype), rtype.n
            return rtype, rtype.n
        impl = BUILTINS.get(name)
        if impl is None:
            raise ShaderError(f'unknown function {name!r}', line)
        types = [v[0] for v in vals]
        width = max([v[1] for v in vals] or [1])
        if any(a is None for a in types):
            return UNKNOWN
        rule = impl[1]
        try:
            rtype = rule(types) if callable(rule) else rule
        except ShaderError as exc:
            if any(isinstance(t, _Loose) for t in types):
                return UNKNOWN
            # min(vec2, vec3): the rule combines its arguments through the
            # same promote, and the run raised from it (or from NumPy, on
            # the two shapes, a step earlier)
            raise ShaderError(exc.msg, line)
        except Exception:                                       # noqa: BLE001
            return UNKNOWN          # not a rule this walk can read
        if rtype is None:
            rtype = types[0] if types else FLOAT
        return _loose(rtype, *types), max(rtype.n, width)


def check_constant(node, types, structs, hlsl=False):
    """This walk over ONE constant expression -- the size of an array, or
    the initialiser of a `const` a size names -- before the parser computes
    it (parser.Parser.fold). `types` is {name: type} for the constants in
    it; the parser has already refused every other name.

    A size is parsed and gone before the check above walks the tree: it is
    a number in every node that carries one. So what is wrong in the
    brackets is refused from here, by the same rules in the same words (a
    swizzle past the width, a field the struct does not have, a call with
    the wrong count, two widths that cannot be combined), where computing
    it would answer with whatever NumPy raised."""
    from .interp import Interpreter
    scope = _Scope()
    scope.names.update(types)
    NameCheck(Interpreter([], structs, hlsl=hlsl)).expr(node, scope)
