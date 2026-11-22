"""Recursive-descent parser.

AST nodes are plain tuples whose first element is the kind, which keeps the
code generator a simple dispatch and avoids a pile of classes.

Expressions
    ('num', text, is_int)      ('bool', True/False)   ('var', name, line)
    ('call', name, [args], line)                       ('index', obj, expr)
    ('member', obj, field, line)
    ('bin', op, l, r, line)    ('un', op, expr)        ('post', op, expr)
    ('assign', op, target, value, line)                ('cond', c, a, b, line)
    ('seq', l, r)
    The three kinds that carry a NAME -- a variable, a function, a field or
    swizzle -- end with the source line of that name, so a name nothing
    declares is reported where it was written (shaders/names.py). The three
    that COMBINE two values -- a binary operator, an assignment, a ternary
    -- end with the line of their operator, so two vector widths that cannot
    be combined are reported there. A node built without a line (the
    interpreter's own compound-assignment rewrite) simply has none.
Statements
    ('block', [stmt])          ('decl', gtype, [(name, arrsize, init)])
    ('expr', e)                ('if', c, then, else_or_None)
    ('for', init, cond, step, body)                    ('while', c, body)
    ('do', body, c)            ('return', e_or_None)
    ('break',) ('continue',) ('discard',) ('nop',)
Top level
    ('func', rettype, name, [(qual, gtype, name, arrsize, default)], body)
        `default` is the expression an HLSL parameter is declared with
        (`float gain = 1.0`), or None: a call may leave such a parameter out.
    ('global', qual, gtype, name, arrsize, init)
    ('struct', name, [(gtype, name, arrsize)])
    One declaration per entry: `uniform float a, b;` and the members of an
    HLSL cbuffer arrive as one 'global' each (Parser.parse flattens them).

Array sizes
    `arrsize` is a number in every entry above: 0 when the declarator has
    no brackets, -1 for `[]` (sized by its initialiser), else the size --
    and a size is at least 1. What is written in the brackets is a
    CONSTANT EXPRESSION and is evaluated here, where it is parsed
    (Parser.array_size): a number, a `const` declared above it with a
    constant initialiser, and whatever the language builds from those
    (`N`, `N * 2`, `(N + 1) / 2`, `int(R) + 1`, `max(A, B)`). The value is
    the one `Interpreter.expr` gives the same expression in the body of the
    shader (interp.fold), so a size and a loop bound written alike are the
    same number. A size that is not a constant, not an integer or below 1
    is a compile error at the line of its bracket; so is a name in it that
    nothing above declares.

    To know what a name in the brackets IS, the parser keeps the scopes
    the name check and the interpreter build (shaders/names.py lists
    them), holding for each declared name either its constant or the
    reason it is not one (Parser.declare). (Until this was written the size
    was read by `_const_int`, which knew a number and answered 0 --
    "no brackets" -- for anything else: `const int N = 3; float a[N];`
    declared ONE float, every index read it, and a size nothing declared
    compiled.)
"""

from .builtins import BUILTINS, CONSTANT
from .gtypes import BY_NAME, GType
from .lexer import ShaderError, preprocess, tokenize

ASSIGN_OPS = ('=', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>=')

BIN_PRECEDENCE = [
    ('||',), ('^^',), ('&&',), ('|',), ('^',), ('&',),
    ('==', '!='), ('<', '>', '<=', '>='), ('<<', '>>'),
    ('+', '-'), ('*', '/', '%'),
]

QUALIFIERS = {'uniform', 'varying', 'attribute', 'in', 'out', 'inout', 'const',
              'static', 'centroid', 'flat', 'smooth', 'noperspective', 'invariant',
              'highp', 'mediump', 'lowp', 'inline',
              # `precise` forbids the DRIVER from contracting the marked
              # arithmetic (FMA, reassociation). This front-end computes
              # strictly ordered float32 already, so accepting the
              # qualifier and ignoring it IS its semantics here
              'precise'}

#: the qualifiers of a global the renderer or the node fills in, and what
#: each makes of the name: whatever else such a declaration says, it is
#: not a constant
BOUND_QUALIFIERS = {'uniform': 'is a uniform', 'in': 'is an input',
                    'attribute': 'is an input', 'varying': 'is an input',
                    'out': 'is an output', 'inout': 'is an output'}

#: the largest size an `int` can hold
MAX_ARRAY = 2 ** 31 - 1

#: the expressions that are never a constant, by node kind: the ones that
#: write, and the comma (which the language leaves out by name)
_WRITES = {'assign': 'it assigns', 'post': 'it writes (++ or --)',
           'pre': 'it writes (++ or --)', 'seq': 'it holds a comma'}


class _Const:
    """A `const` whose initialiser is a constant expression: what a size
    may name. `uses` is what the initialiser names itself, as it stood
    where it was written; `slot` is its [value, type], computed the first
    time a size asks (Parser.fold) -- a shader that sizes its arrays with
    numbers evaluates nothing here."""
    __slots__ = ('type', 'init', 'uses', 'slot')

    def __init__(self, gtype, init, uses):
        self.type = gtype
        self.init = init
        self.uses = uses
        self.slot = None


class _Uses:
    """What a constant expression names: its constants by name, and the
    builtins it calls."""
    __slots__ = ('consts', 'calls')

    def __init__(self):
        self.consts = {}
        self.calls = set()


class Parser:
    def __init__(self, src, hlsl=False, defines=None):
        self.src = preprocess(src, defines)
        self.toks = tokenize(self.src, hlsl=hlsl)
        self.i = 0
        self.structs = {}
        self.hlsl = hlsl
        #: what is declared where the parser stands, the innermost scope
        #: last: name -> its _Const, or the words for why it is not a
        #: constant (`declare`). The root starts with the one name GLSL
        #: declares for the shader (names.BUILTIN_VARS)
        self.scopes = [{'gl_FragCoord': "'gl_FragCoord' is not a constant"}]
        #: the functions the shader has declared so far
        self.own_funcs = set()
        #: builtin -> line, for each one a size was computed through
        self.folded = {}
        #: inside the braces of an HLSL cbuffer
        self.in_cbuffer = False

    # ------------------------------------------------------------- utilities
    @property
    def tok(self):
        return self.toks[self.i]

    def peek(self, k=1):
        j = min(self.i + k, len(self.toks) - 1)
        return self.toks[j]

    def next(self):
        t = self.toks[self.i]
        self.i += 1
        return t

    def at(self, value, kind=None):
        t = self.tok
        return t.value == value and (kind is None or t.kind == kind)

    def accept(self, value, kind=None):
        if self.at(value, kind):
            return self.next()
        return None

    def expect(self, value, kind=None):
        if not self.at(value, kind):
            raise ShaderError(f"expected {value!r}, found {self.tok.value!r}",
                              self.tok.line, self.tok.col)
        return self.next()

    def err(self, msg):
        raise ShaderError(msg, self.tok.line, self.tok.col)

    # -------------------------------------------------------------- top level
    def parse(self):
        decls = []
        while self.tok.kind != 'eof':
            d = self.parse_toplevel()
            if d is not None:
                _flatten(d, decls)
        for name, line in self.folded.items():
            if name in self.own_funcs:
                # the size was computed through the builtin, and when the
                # shader runs every call of the name reaches its own
                # function (Interpreter.call), wherever that is written
                raise ShaderError(
                    'array size is not a compile-time constant: '
                    f"{name}() is the shader's own function", line)
        return decls

    def parse_toplevel(self):
        if self.accept(';'):
            return None
        if self.at('precision', 'keyword'):
            while not self.at(';') and self.tok.kind != 'eof':
                self.next()
            self.accept(';')
            return None
        if self.at('layout', 'keyword'):
            self.next()
            self.expect('(')
            depth = 1
            while depth and self.tok.kind != 'eof':
                v = self.next().value
                depth += (v == '(') - (v == ')')
        if self.at('struct', 'keyword'):
            return self.parse_struct()
        if self.at('cbuffer', 'keyword'):          # HLSL constant buffer
            self.next()
            self.next()
            while not self.at('{') and self.tok.kind != 'eof':
                self.next()
            self.expect('{')
            out = []
            self.in_cbuffer = True
            while not self.at('}') and self.tok.kind != 'eof':
                out.append(self.parse_toplevel())
            self.in_cbuffer = False
            self.expect('}')
            self.accept(';')
            return ('block', [_as_uniform(o) for o in out if o])

        quals = []
        while self.tok.value in QUALIFIERS and self.tok.kind == 'keyword':
            quals.append(self.next().value)
        gtype = self.parse_type()
        if gtype is None:
            self.err(f'unknown type {self.tok.value!r}')
        name_tok = self.tok
        if name_tok.kind not in ('ident', 'type'):
            self.err('expected declarator name')
        name = self.next().value

        if self.at('('):
            return self.parse_function(gtype, name)

        # a global is a constant when it says so and nothing fills it in:
        # not a uniform, an input or an output, and not a cbuffer's member
        # (those ARE the shader's uniforms, `_as_uniform`)
        why = "is a cbuffer's member" if self.in_cbuffer else \
            next((BOUND_QUALIFIERS[q] for q in quals
                  if q in BOUND_QUALIFIERS), None)
        const = 'const' in quals and why is None
        why = why or 'is not declared const'
        arr = self.parse_array_suffix()
        self.skip_semantic()
        init = None
        if self.accept('='):
            init = self.parse_assign()
        self.declare(name, const, gtype, arr, init, why)
        results = [('global', quals, gtype, name, arr, init)]
        while self.accept(','):
            nm = self.next().value
            a2 = self.parse_array_suffix()
            self.skip_semantic()
            ini = self.parse_assign() if self.accept('=') else None
            self.declare(nm, const, gtype, a2, ini, why)
            results.append(('global', quals, gtype, nm, a2, ini))
        self.expect(';')
        return results[0] if len(results) == 1 else ('block', results)

    def skip_semantic(self):
        """HLSL ': SV_Target' / ': register(c0)' annotations."""
        if self.at(':'):
            self.next()
            self.next()
            if self.at('('):
                depth = 0
                while self.tok.kind != 'eof':
                    v = self.next().value
                    depth += (v == '(') - (v == ')')
                    if depth == 0:
                        break

    def parse_struct(self):
        self.expect('struct')
        name = self.next().value
        self.expect('{')
        fields = []
        while not self.at('}'):
            while self.tok.value in QUALIFIERS and self.tok.kind == 'keyword':
                self.next()
            ft = self.parse_type()
            if ft is None:
                self.err('bad struct field type')
            while True:
                fname = self.next().value
                fa = self.parse_array_suffix()
                self.skip_semantic()
                fields.append((ft, fname, fa))
                if not self.accept(','):
                    break
            self.expect(';')
        self.expect('}')
        self.accept(';')
        # the struct belongs to THIS source: `self.structs` is where the
        # parser, the name check and the interpreter look it up. (It was
        # also written into gtypes.BY_NAME, the module's table of builtin
        # types: there the interpreter found it ahead of its own struct
        # branch and built `S(a, b)` as a scalar cast of `a`, and there it
        # stayed for every source compiled afterwards in the session.)
        self.structs[name] = fields
        return ('struct', name, fields)

    def parse_function(self, rettype, name):
        self.own_funcs.add(name)
        # the parameters' scope: the body is a block inside it
        self.scopes.append({})
        fn = self.parse_function_rest(rettype, name)
        self.scopes.pop()
        return fn

    def parse_function_rest(self, rettype, name):
        self.expect('(')
        params = []
        if not self.at(')'):
            while True:
                qual = 'in'
                while self.tok.value in QUALIFIERS and self.tok.kind == 'keyword':
                    q = self.next().value
                    if q in ('in', 'out', 'inout', 'const'):
                        qual = q if q != 'const' else qual
                pt = self.parse_type()
                if pt is None:
                    if self.at(')'):
                        break
                    self.err('bad parameter type')
                if pt.base == 'void' and self.at(')'):
                    break
                pname = self.next().value if self.tok.kind == 'ident' else ''
                pa = self.parse_array_suffix()
                self.skip_semantic()
                # an HLSL default (`float gain = 1.0`) is kept: a call that
                # leaves the parameter out reads it (it was parsed and
                # dropped, and the missing argument read zero)
                default = self.parse_assign() if self.accept('=') else None
                # a parameter is never a constant, `const` or not: it is
                # whatever the call passes
                self.declare(pname, why='is a parameter')
                params.append((qual, pt, pname, pa, default))
                if not self.accept(','):
                    break
        self.expect(')')
        self.skip_semantic()
        if self.accept(';'):
            return ('proto', rettype, name, params)
        body = self.parse_block()
        return ('func', rettype, name, params, body)

    def parse_type(self):
        t = self.tok
        if t.kind == 'type':
            self.next()
            gt = BY_NAME.get(t.value)
            if gt is None:
                return None
            return gt
        if t.kind == 'ident' and t.value in self.structs:
            self.next()
            return GType('struct', 1, 0, 0, t.value)
        if t.kind == 'keyword' and t.value == 'void':
            self.next()
            return BY_NAME['void']
        return None

    def parse_array_suffix(self):
        size = 0
        while True:
            bracket = self.accept('[')
            if not bracket:
                return size
            if self.at(']'):
                size = -1
            else:
                size = self.array_size(self.parse_assign(), bracket.line)
            self.expect(']')

    # ------------------------------------------------- constants and sizes
    def scoped(self, parse):
        """What `parse` returns, read in a scope of its own: a
        declaration it meets is gone when it is done."""
        self.scopes.append({})
        out = parse()
        self.scopes.pop()
        return out

    def declare(self, name, const=False, gtype=None, arr=0, init=None,
                why='is not declared const'):
        """Record a name the source declares, in the scope the parser
        stands in -- AFTER its initialiser was parsed, which still saw the
        outer name (the block rule of shaders/names.py).

        Every declared name is recorded, constant or not: `int N = 2;` in a
        block hides the `const int N` above it, and a size that names it
        there is refused for what it is rather than given the outer 3. A
        `const` with a constant initialiser becomes a _Const; anything else
        is recorded as the words that say why it is not one."""
        if not name:
            return
        if not const:
            slot = f'{name!r} {why}'
        elif init is None:
            slot = f'{name!r} has no initialiser'
        else:
            uses = _Uses()
            try:
                no = self.not_constant(init, uses)
            except RecursionError:
                # an initialiser nested this deep is the name check's to
                # refuse, in its own words (compiler.try_compile)
                no = ('it is too deeply nested', None)
            if no is None:
                t = GType(gtype.base, gtype.n, gtype.rows, arr,
                          gtype.struct) if arr else gtype
                slot = _Const(t, init, uses)
            else:
                slot = (f'the initialiser of {name!r} is not constant '
                        f'({no[0]})')
        self.scopes[-1][name] = slot

    def not_constant(self, node, uses):
        """None when `node` is a constant expression, and `uses` then
        holds the _Const of every name in it. Otherwise (why, error): the
        words for what is not constant, and -- for a name nothing declares
        or a function that does not exist -- the error it is on its own.

        The language's rule (GLSL 4.3.3): a literal; a `const` variable
        whose initialiser is a constant expression; an operator, a
        constructor or a builtin function applied to constant expressions.
        Never a call of the shader's own function, and nothing that
        writes. It never raises: a `const` it cannot follow is simply not
        one a size may name."""
        k = node[0]
        if k in ('num', 'bool'):
            return None
        if k == 'var':
            name = node[1]
            for scope in reversed(self.scopes):
                slot = scope.get(name)
                if slot is not None:
                    if isinstance(slot, str):
                        return slot, None
                    uses.consts[name] = slot
                    return None
            line = node[2] if len(node) > 2 else 0
            return (f'{name!r} is not declared above it',
                    ShaderError(f'undeclared identifier {name!r}', line))
        if k == 'un':
            if node[1] in ('++', '--'):
                return _WRITES['pre'], None
            return self.not_constant(node[2], uses)
        if k == 'bin':
            parts = node[2:4]
        elif k == 'cond':
            parts = node[1:4]
        elif k == 'index':
            parts = node[1:3]
        elif k == 'member':
            parts = node[1:2]
        elif k == 'arrinit':
            parts = node[2]
        elif k == 'call':
            name, parts = node[1], node[2]
            t = BY_NAME.get(name)
            if name in self.structs or \
                    (t is not None and t.base not in ('void', 'sampler')):
                pass                                    # a constructor
            elif name in self.own_funcs:
                return f"{name}() is the shader's own function", None
            elif name in CONSTANT:
                uses.calls.add(name)
            elif name in BUILTINS:
                return f'{name}() is not evaluated at compile time', None
            else:
                line = node[3] if len(node) > 3 else 0
                return (f'{name}() is not a function',
                        ShaderError(f'unknown function {name!r}', line))
        else:
            return _WRITES.get(k, f'{k!r} is not a constant'), None
        for part in parts:
            no = self.not_constant(part, uses)
            if no is not None:
                return no
        return None

    def fold(self, node, uses, line, as_type=None):
        """[value, type] of a constant expression: its constants first,
        each once, then the expression itself -- through the name check
        (names.check_constant), which refuses here what it refuses in the
        body of a shader, in the same words, and then computed
        (interp.fold)."""
        from .interp import fold
        from .names import check_constant
        for called in uses.calls:
            # held against the functions the shader goes on to declare
            # (`parse`)
            self.folded.setdefault(called, line)
        consts = {}
        for name, const in uses.consts.items():
            if const.slot is None:
                const.slot = self.fold(const.init, const.uses, line,
                                       const.type)
            consts[name] = const.slot
        try:
            check_constant(node, {name: slot[1]
                                  for name, slot in consts.items()},
                           self.structs, self.hlsl)
        except ShaderError as exc:
            raise ShaderError(exc.msg, exc.line or line)
        try:
            return fold(node, consts, self.structs, self.hlsl, as_type)
        except ShaderError as exc:
            raise ShaderError(f'array size: {exc.msg}', line)
        except RecursionError:
            raise
        except Exception as exc:                                # noqa: BLE001
            raise ShaderError('array size cannot be evaluated '
                              f'({type(exc).__name__}: {exc})', line)

    def array_size(self, node, line):
        """The size written in a declarator's brackets, as a number.

        A number is read as it always was (`3`, `3u`, and `3.0` as 3).
        Anything else is a constant expression and is computed: it has to
        BE one, of type int or uint. Either way the size is at least 1 --
        0 is how a declaration with no brackets is told apart, so a size
        of 0 declared a plain float, and -1 is `[]`."""
        if node[0] == 'num':
            size = _const_int(node)
        else:
            uses = _Uses()
            no = self.not_constant(node, uses)
            if no is not None:
                raise no[1] or ShaderError(
                    f'array size is not a compile-time constant: {no[0]}',
                    line)
            v, t = self.fold(node, uses, line)
            if t.base not in ('int', 'uint') or t.n != 1 or t.array \
                    or t.is_matrix:
                raise ShaderError('array size must be an int, this is '
                                  f'{_a(t.name())}', line)
            one = _one(v)
            try:
                size = int(one)
            except (TypeError, ValueError, OverflowError):
                size = None             # nan, inf
            if size is None or abs(size) > MAX_ARRAY:
                # `1 / 0` and its kin: the interpreter answers a division
                # by zero with the dividend times 1e30
                raise ShaderError('array size does not fit an int', line)
            if size != one:
                # typed int by the interpreter's shortcut (a builtin
                # answers its argument's type: `sqrt(2)`), not a whole
                # number
                raise ShaderError('array size must be an int, this is '
                                  f'{float(one):g}', line)
        if size < 1:
            raise ShaderError(f'array size must be at least 1, not {size}',
                              line)
        return size

    # ------------------------------------------------------------- statements
    def parse_block(self):
        self.expect('{')
        stmts = []
        self.scopes.append({})
        while not self.at('}'):
            if self.tok.kind == 'eof':
                self.err('unexpected end of source in block')
            stmts.append(self.parse_statement())
        self.scopes.pop()
        self.expect('}')
        return ('block', stmts)

    def looks_like_decl(self):
        t = self.tok
        if t.kind == 'type':
            return True
        if t.kind == 'keyword' and t.value in ('const', 'static',
                                               'precise'):
            return True
        if t.kind == 'ident' and t.value in self.structs and \
                self.peek().kind == 'ident':
            return True
        return False

    def parse_statement(self):
        if self.at('{'):
            return self.parse_block()
        if self.accept(';'):
            return ('nop',)
        t = self.tok
        if t.kind == 'keyword':
            if t.value == 'if':
                return self.parse_if()
            if t.value == 'for':
                return self.parse_for()
            if t.value == 'while':
                return self.parse_while()
            if t.value == 'do':
                return self.parse_do()
            if t.value == 'return':
                self.next()
                e = None if self.at(';') else self.parse_expr()
                self.expect(';')
                return ('return', e)
            if t.value == 'break':
                self.next()
                self.expect(';')
                return ('break',)
            if t.value == 'continue':
                self.next()
                self.expect(';')
                return ('continue',)
            if t.value == 'discard':
                self.next()
                self.expect(';')
                return ('discard',)
            if t.value == 'switch':
                return self.parse_switch()
        if self.looks_like_decl():
            d = self.parse_decl()
            self.expect(';')
            return d
        e = self.parse_expr()
        self.expect(';')
        return ('expr', e)

    def parse_decl(self):
        const = False
        while self.tok.value in QUALIFIERS and self.tok.kind == 'keyword':
            const |= self.next().value == 'const'
        gtype = self.parse_type()
        if gtype is None:
            self.err('unknown type in declaration')
        items = []
        while True:
            name = self.next().value
            arr = self.parse_array_suffix()
            init = None
            if self.accept('='):
                init = self.parse_assign()
            self.declare(name, const, gtype, arr, init)
            items.append((name, arr, init))
            if not self.accept(','):
                break
        return ('decl', gtype, items)

    def parse_if(self):
        self.expect('if')
        self.expect('(')
        c = self.parse_expr()
        self.expect(')')
        # an arm is a block of its own, braces or not (shaders/names.py)
        then = self.scoped(self.parse_statement)
        els = None
        if self.accept('else'):
            els = self.scoped(self.parse_statement)
        return ('if', c, then, els)

    def parse_for(self):
        # one scope for the init, the condition and the step; the body is
        # a block inside it
        return self.scoped(self.parse_for_rest)

    def parse_for_rest(self):
        self.expect('for')
        self.expect('(')
        if self.accept(';'):
            init = ('nop',)
        elif self.looks_like_decl():
            init = self.parse_decl()
            self.expect(';')
        else:
            init = ('expr', self.parse_expr())
            self.expect(';')
        cond = None if self.at(';') else self.parse_expr()
        self.expect(';')
        step = None if self.at(')') else self.parse_expr()
        self.expect(')')
        body = self.scoped(self.parse_statement)
        return ('for', init, cond, step, body)

    def parse_while(self):
        self.expect('while')
        self.expect('(')
        c = self.parse_expr()
        self.expect(')')
        return ('while', c, self.scoped(self.parse_statement))

    def parse_do(self):
        self.expect('do')
        body = self.scoped(self.parse_statement)
        self.expect('while')
        self.expect('(')
        c = self.parse_expr()
        self.expect(')')
        self.expect(';')
        return ('do', body, c)

    def parse_switch(self):
        self.expect('switch')
        self.expect('(')
        sel = self.parse_expr()
        self.expect(')')
        self.expect('{')
        cases = []
        cur = None
        # each case is lowered to a block of its own below, so what one
        # declares the next does not see
        self.scopes.append({})
        while not self.at('}'):
            ctok = self.accept('case')
            if ctok:
                v = self.parse_expr()
                self.expect(':')
                self.scopes[-1] = {}
                cur = (v, [], ctok.line)
                cases.append(cur)
                continue
            if self.accept('default'):
                self.expect(':')
                self.scopes[-1] = {}
                cur = (None, [], 0)
                cases.append(cur)
                continue
            s = self.parse_statement()
            if cur is None:
                continue
            cur[1].append(s)
        self.scopes.pop()
        self.expect('}')
        # lower to if/else-if; 'break' inside becomes a no-op at this level
        node = None
        for v, body, line in reversed(cases):
            stripped = ('block', [s for s in body if s[0] != 'break'])
            if v is None:
                node = stripped
            else:
                node = ('if', ('bin', '==', sel, v, line), stripped, node)
        return node or ('nop',)

    # ------------------------------------------------------------ expressions
    def parse_expr(self):
        e = self.parse_assign()
        while self.accept(','):
            e = ('seq', e, self.parse_assign())
        return e

    def parse_assign(self):
        left = self.parse_cond()
        if self.tok.kind == 'punct' and self.tok.value in ASSIGN_OPS:
            optok = self.next()
            right = self.parse_assign()
            return ('assign', optok.value, left, right, optok.line)
        return left

    def parse_cond(self):
        c = self.parse_binary(0)
        qtok = self.accept('?')
        if qtok:
            a = self.parse_assign()
            self.expect(':')
            b = self.parse_assign()
            return ('cond', c, a, b, qtok.line)
        return c

    def parse_binary(self, level):
        if level >= len(BIN_PRECEDENCE):
            return self.parse_unary()
        ops = BIN_PRECEDENCE[level]
        left = self.parse_binary(level + 1)
        while self.tok.kind == 'punct' and self.tok.value in ops:
            optok = self.next()
            right = self.parse_binary(level + 1)
            left = ('bin', optok.value, left, right, optok.line)
        return left

    def parse_unary(self):
        t = self.tok
        if t.kind == 'punct' and t.value in ('+', '-', '!', '~', '++', '--'):
            self.next()
            return ('un', t.value, self.parse_unary())
        return self.parse_postfix()

    def parse_postfix(self):
        e = self.parse_primary()
        while True:
            if self.accept('.'):
                ftok = self.next()
                field = ftok.value
                if self.at('('):
                    self.err('method calls are not supported')
                e = ('member', e, field, ftok.line)
            elif self.accept('['):
                idx = self.parse_expr()
                self.expect(']')
                e = ('index', e, idx)
            elif self.tok.kind == 'punct' and self.tok.value in ('++', '--'):
                op = self.next().value
                e = ('post', op, e)
            else:
                break
        return e

    def parse_primary(self):
        t = self.tok
        if t.kind == 'number':
            self.next()
            txt = t.value
            is_int = not any(c in txt for c in '.eEfFhH') or txt.lower().startswith('0x')
            return ('num', txt, is_int)
        if t.kind == 'keyword' and t.value in ('true', 'false'):
            self.next()
            return ('bool', t.value == 'true')
        if t.value == '(' and t.kind == 'punct':
            self.next()
            e = self.parse_expr()
            self.expect(')')
            return e
        if t.kind in ('ident', 'type'):
            name = self.next().value
            if t.kind == 'type' and self.at('['):
                # array constructor: float[4](a, b, c, d)
                self.next()
                if not self.at(']'):
                    self.parse_assign()          # declared size (informative)
                self.expect(']')
                self.expect('(')
                args = []
                if not self.at(')'):
                    while True:
                        args.append(self.parse_assign())
                        if not self.accept(','):
                            break
                self.expect(')')
                return ('arrinit', name, args)
            if self.at('('):
                self.next()
                args = []
                if not self.at(')'):
                    while True:
                        args.append(self.parse_assign())
                        if not self.accept(','):
                            break
                self.expect(')')
                return ('call', name, args, t.line)
            return ('var', name, t.line)
        self.err(f'unexpected token {t.value!r}')


def _const_int(node):
    """A number written as a size: `3`, `3u`, `0x10`, and `3.0` or `3f`
    as 3 (no driver takes a float there; this front-end always has, and a
    shader that runs is not refused for it). Anything but a number is
    `Parser.array_size`'s."""
    if node[0] == 'num':
        text = node[1]
        try:
            if text[:2].lower() == '0x':
                # (it was `float(text, 0)`: a TypeError, so a size written
                # in hex failed the compile with Python's own message)
                return int(text.rstrip('uU'), 16)
            return int(float(text.rstrip('uUfFhH')))
        except ValueError:
            return 0
    return 0


def _one(v):
    """The one number a folded constant holds (one lane, one component)."""
    import numpy as np
    return np.asarray(v).reshape(-1)[0]


def _a(noun):
    return ('an ' if noun[:1] in 'aeiou' else 'a ') + noun


def _as_uniform(d):
    """A cbuffer's members ARE the shader's uniforms: HLSL writes no
    qualifier inside the braces, so the member is given the one the rest
    of the front-end files a socket under."""
    if d[0] == 'block':
        return ('block', [_as_uniform(sub) for sub in d[1]])
    if d[0] == 'global' and 'uniform' not in d[1]:
        return (d[0], list(d[1]) + ['uniform']) + tuple(d[2:])
    return d


def _flatten(d, out):
    """A multi-declarator global (`uniform float a, b;`) and an HLSL
    cbuffer both parse to ('block', [...]). Every reader of the top level
    -- the interpreter, the coded-shader port -- walks 'global' entries, so
    a block it did not open was a declaration it never saw: the names read
    as unknown identifiers when the shader ran."""
    if d[0] == 'block':
        for sub in d[1]:
            _flatten(sub, out)
    else:
        out.append(d)


def parse(src, hlsl=False, defines=None):
    p = Parser(src, hlsl=hlsl, defines=defines)
    decls = p.parse()
    return decls, p.structs
