"""Name resolution at compile time.

The interpreter resolves a name when a lane reaches it: an identifier
nothing declares, a field a struct does not have, a swizzle past the
vector's width all compiled, and failed only when the shader ran -- or
never, when the line sat in a branch no lane took or in a function nothing
called, which is the text a driver refuses whole. The source generator this
interpreter replaced caught every one of them while it emitted code
(`undeclared identifier`, `has no field`, `bad swizzle`, `swizzle out of
range`); `Interpreter.validate` brought back the unknown-function half and
this module is the rest.

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
from .lexer import ShaderError

#: in scope with no declaration, as GLSL defines it (Interpreter.run binds
#: the same name into the root)
BUILTIN_VARS = {'gl_FragCoord': GType('float', 4)}

#: a value nothing is known about: its names were checked, and a field or
#: swizzle on it is not judged
UNKNOWN = (None, 0)


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
                    ptype = p[1]
                    if len(p) > 3 and p[3]:
                        ptype = GType(ptype.base, ptype.n, ptype.rows, p[3],
                                      ptype.struct)
                    inner.names[p[2]] = ptype
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
            if bt.base == 'struct':
                for ftype, fname, _farr in self.structs.get(bt.struct, ()):
                    if fname == field:
                        return ftype, ftype.n
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
                return et, et.n
            if bt.is_matrix:
                return GType(bt.base, bt.rows), bt.rows
            return GType(bt.base, 1), 1
        if k == 'call':
            return self.call(node, scope)
        if k == 'bin':
            (at, aw), (bt, bw) = (self.expr(node[2], scope),
                                  self.expr(node[3], scope))
            op = node[1]
            if op in ('&&', 'and', '||', 'or', '==', '!=', '<', '>', '<=',
                      '>='):
                return BOOL, 1
            if at is None or bt is None:
                return UNKNOWN
            if op == '*' and (at.is_matrix or bt.is_matrix):
                if at.is_matrix and bt.is_matrix:
                    return at, at.n
                if at.is_matrix:
                    return GType(at.base, at.rows), max(at.rows, bw)
                return GType(bt.base, bt.n), max(bt.n, aw)
            return self._promote(at, bt, max(aw, bw))
        if k == 'assign':
            self.expr(node[3], scope)
            tt, tw = self.expr(node[2], scope)
            # the interpreter answers a plain variable's own type; what a
            # field, swizzle or element write answers is its own affair
            return (tt, tw) if node[2][0] == 'var' else UNKNOWN
        if k == 'cond':
            self.expr(node[1], scope)
            (at, aw), (bt, bw) = (self.expr(node[2], scope),
                                  self.expr(node[3], scope))
            if at is None or bt is None:
                return UNKNOWN
            return self._promote(at, bt, max(aw, bw))
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

    @staticmethod
    def _promote(at, bt, width):
        if at.base == 'struct' or bt.base == 'struct' or at.array or bt.array:
            return UNKNOWN
        try:
            return promote(at, bt), width
        except ShaderError:
            return UNKNOWN          # a TYPE error; the run reports those

    def call(self, node, scope):
        name, args = node[1], node[2]
        vals = []
        for a in args:              # a loop, not a comprehension: the walk
            vals.append(self.expr(a, scope))    # is never deeper than the run
        t = BY_NAME.get(name)
        if t is None and name in self.structs:
            t = GType('struct', 1, 1, 0, name)
        if t is not None:                       # a constructor
            return t, t.n
        fn = self.it.pick(name, len(args))
        if fn is not None:
            rtype = fn[1] if fn[1] != VOID else FLOAT
            return rtype, rtype.n
        impl = BUILTINS.get(name)
        if impl is None:
            raise ShaderError(f'unknown function {name!r}', _line(node, 3))
        types = [v[0] for v in vals]
        width = max([v[1] for v in vals] or [1])
        if any(a is None for a in types):
            return UNKNOWN
        rule = impl[1]
        try:
            rtype = rule(types) if callable(rule) else rule
        except Exception:                                       # noqa: BLE001
            return UNKNOWN          # too few arguments for the rule to read
        if rtype is None:
            rtype = types[0] if types else FLOAT
        return rtype, max(rtype.n, width)
