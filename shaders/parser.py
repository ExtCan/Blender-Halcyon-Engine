"""Recursive-descent parser.

AST nodes are plain tuples whose first element is the kind, which keeps the
code generator a simple dispatch and avoids a pile of classes.

Expressions
    ('num', text, is_int)      ('bool', True/False)   ('var', name)
    ('call', name, [args])     ('member', obj, field)  ('index', obj, expr)
    ('bin', op, l, r)          ('un', op, expr)        ('post', op, expr)
    ('assign', op, target, value)                      ('cond', c, a, b)
    ('seq', l, r)
Statements
    ('block', [stmt])          ('decl', gtype, [(name, arrsize, init)])
    ('expr', e)                ('if', c, then, else_or_None)
    ('for', init, cond, step, body)                    ('while', c, body)
    ('do', body, c)            ('return', e_or_None)
    ('break',) ('continue',) ('discard',) ('nop',)
Top level
    ('func', rettype, name, [(qual, gtype, name, arrsize)], body)
    ('global', qual, gtype, name, arrsize, init)
    ('struct', name, [(gtype, name, arrsize)])
"""

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
              'highp', 'mediump', 'lowp', 'inline'}


class Parser:
    def __init__(self, src, hlsl=False, defines=None):
        self.src = preprocess(src, defines)
        self.toks = tokenize(self.src, hlsl=hlsl)
        self.i = 0
        self.structs = {}
        self.hlsl = hlsl

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
                decls.append(d)
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
            while not self.at('}') and self.tok.kind != 'eof':
                out.append(self.parse_toplevel())
            self.expect('}')
            self.accept(';')
            return ('block', [o for o in out if o])

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

        arr = self.parse_array_suffix()
        self.skip_semantic()
        init = None
        if self.accept('='):
            init = self.parse_assign()
        results = [('global', quals, gtype, name, arr, init)]
        while self.accept(','):
            nm = self.next().value
            a2 = self.parse_array_suffix()
            self.skip_semantic()
            ini = self.parse_assign() if self.accept('=') else None
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
        self.structs[name] = fields
        BY_NAME[name] = GType('struct', 1, 0, 0, name)
        return ('struct', name, fields)

    def parse_function(self, rettype, name):
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
                if self.accept('='):
                    self.parse_assign()
                params.append((qual, pt, pname, pa))
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
        while self.accept('['):
            if self.at(']'):
                size = -1
            else:
                e = self.parse_assign()
                size = _const_int(e)
            self.expect(']')
        return size

    # ------------------------------------------------------------- statements
    def parse_block(self):
        self.expect('{')
        stmts = []
        while not self.at('}'):
            if self.tok.kind == 'eof':
                self.err('unexpected end of source in block')
            stmts.append(self.parse_statement())
        self.expect('}')
        return ('block', stmts)

    def looks_like_decl(self):
        t = self.tok
        if t.kind == 'type':
            return True
        if t.kind == 'keyword' and t.value in ('const', 'static'):
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
        while self.tok.value in QUALIFIERS and self.tok.kind == 'keyword':
            self.next()
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
            items.append((name, arr, init))
            if not self.accept(','):
                break
        return ('decl', gtype, items)

    def parse_if(self):
        self.expect('if')
        self.expect('(')
        c = self.parse_expr()
        self.expect(')')
        then = self.parse_statement()
        els = None
        if self.accept('else'):
            els = self.parse_statement()
        return ('if', c, then, els)

    def parse_for(self):
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
        body = self.parse_statement()
        return ('for', init, cond, step, body)

    def parse_while(self):
        self.expect('while')
        self.expect('(')
        c = self.parse_expr()
        self.expect(')')
        return ('while', c, self.parse_statement())

    def parse_do(self):
        self.expect('do')
        body = self.parse_statement()
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
        while not self.at('}'):
            if self.accept('case'):
                v = self.parse_expr()
                self.expect(':')
                cur = (v, [])
                cases.append(cur)
                continue
            if self.accept('default'):
                self.expect(':')
                cur = (None, [])
                cases.append(cur)
                continue
            s = self.parse_statement()
            if cur is None:
                continue
            cur[1].append(s)
        self.expect('}')
        # lower to if/else-if; 'break' inside becomes a no-op at this level
        node = None
        for v, body in reversed(cases):
            stripped = ('block', [s for s in body if s[0] != 'break'])
            if v is None:
                node = stripped
            else:
                node = ('if', ('bin', '==', sel, v), stripped, node)
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
            op = self.next().value
            right = self.parse_assign()
            return ('assign', op, left, right)
        return left

    def parse_cond(self):
        c = self.parse_binary(0)
        if self.accept('?'):
            a = self.parse_assign()
            self.expect(':')
            b = self.parse_assign()
            return ('cond', c, a, b)
        return c

    def parse_binary(self, level):
        if level >= len(BIN_PRECEDENCE):
            return self.parse_unary()
        ops = BIN_PRECEDENCE[level]
        left = self.parse_binary(level + 1)
        while self.tok.kind == 'punct' and self.tok.value in ops:
            op = self.next().value
            right = self.parse_binary(level + 1)
            left = ('bin', op, left, right)
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
                field = self.next().value
                if self.at('('):
                    self.err('method calls are not supported')
                e = ('member', e, field)
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
            if self.at('('):
                self.next()
                args = []
                if not self.at(')'):
                    while True:
                        args.append(self.parse_assign())
                        if not self.accept(','):
                            break
                self.expect(')')
                return ('call', name, args)
            return ('var', name)
        self.err(f'unexpected token {t.value!r}')


def _const_int(node):
    if node[0] == 'num':
        try:
            return int(float(node[1].rstrip('uUfFhH'), 0) if node[1].lower().startswith('0x')
                       else float(node[1].rstrip('uUfFhH')))
        except ValueError:
            return 0
    if node[0] == 'un' and node[1] == '-':
        return -_const_int(node[2])
    return 0


def parse(src, hlsl=False, defines=None):
    p = Parser(src, hlsl=hlsl, defines=defines)
    decls = p.parse()
    return decls, p.structs
