"""Tokeniser and minimal preprocessor for the coded-shader nodes.

Handles GLSL ES 1.00/3.00-flavoured source and an HLSL dialect; the parser maps
HLSL type and function names onto the GLSL ones, so both share this lexer.
"""

import re

KEYWORDS = {
    'const', 'uniform', 'varying', 'attribute', 'in', 'out', 'inout', 'struct',
    'if', 'else', 'for', 'while', 'do', 'break', 'continue', 'return', 'discard',
    'switch', 'case', 'default', 'true', 'false', 'void', 'precision',
    'highp', 'mediump', 'lowp', 'flat', 'smooth', 'noperspective', 'layout',
    'centroid', 'invariant', 'static', 'inline', 'cbuffer', 'register',
}

TYPES = {
    'void', 'bool', 'int', 'uint', 'float', 'double',
    'vec2', 'vec3', 'vec4', 'bvec2', 'bvec3', 'bvec4',
    'ivec2', 'ivec3', 'ivec4', 'uvec2', 'uvec3', 'uvec4',
    'mat2', 'mat3', 'mat4', 'mat2x2', 'mat3x3', 'mat4x4',
    'sampler2D', 'samplerCube', 'sampler3D', 'sampler2DShadow',
}

HLSL_TYPE_ALIASES = {
    'float2': 'vec2', 'float3': 'vec3', 'float4': 'vec4',
    'half': 'float', 'half2': 'vec2', 'half3': 'vec3', 'half4': 'vec4',
    'fixed': 'float', 'fixed2': 'vec2', 'fixed3': 'vec3', 'fixed4': 'vec4',
    'int2': 'ivec2', 'int3': 'ivec3', 'int4': 'ivec4',
    'uint2': 'uvec2', 'uint3': 'uvec3', 'uint4': 'uvec4',
    'bool2': 'bvec2', 'bool3': 'bvec3', 'bool4': 'bvec4',
    'float2x2': 'mat2', 'float3x3': 'mat3', 'float4x4': 'mat4',
    'matrix': 'mat4', 'sampler': 'sampler2D', 'Texture2D': 'sampler2D',
    'SamplerState': 'sampler2D', 'min16float': 'float',
}

PUNCT = [
    '<<=', '>>=', '++', '--', '<<', '>>', '<=', '>=', '==', '!=', '&&', '||', '^^',
    '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=',
    '{', '}', '(', ')', '[', ']', ';', ',', '.', ':', '?',
    '+', '-', '*', '/', '%', '<', '>', '=', '!', '&', '|', '^', '~',
]

_NUM = re.compile(r'''
    (?:0[xX][0-9a-fA-F]+[uU]?)
  | (?:\d+\.\d*(?:[eE][+-]?\d+)?[fFhH]?)
  | (?:\.\d+(?:[eE][+-]?\d+)?[fFhH]?)
  | (?:\d+[eE][+-]?\d+[fFhH]?)
  | (?:\d+[fFhH])
  | (?:\d+[uU]?)
''', re.X)

_IDENT = re.compile(r'[A-Za-z_][A-Za-z_0-9]*')


class Token:
    __slots__ = ('kind', 'value', 'line', 'col')

    def __init__(self, kind, value, line, col):
        self.kind = kind          # ident|type|keyword|number|punct|eof
        self.value = value
        self.line = line
        self.col = col

    def __repr__(self):
        return f'<{self.kind}:{self.value}@{self.line}>'


class ShaderError(Exception):
    def __init__(self, msg, line=0, col=0):
        super().__init__(f'line {line}: {msg}' if line else msg)
        self.msg = msg
        self.line = line
        self.col = col


def strip_comments(src):
    out = []
    i = 0
    n = len(src)
    while i < n:
        c = src[i]
        if c == '/' and i + 1 < n:
            if src[i + 1] == '/':
                j = src.find('\n', i)
                if j < 0:
                    break
                i = j
                continue
            if src[i + 1] == '*':
                j = src.find('*/', i + 2)
                if j < 0:
                    raise ShaderError('unterminated block comment')
                out.append('\n' * src.count('\n', i, j))
                i = j + 2
                continue
        out.append(c)
        i += 1
    return ''.join(out)


def preprocess(src, defines=None):
    """#define (object and function-like), #ifdef/#ifndef/#if/#else/#endif, #undef.

    Deliberately small: enough for the macro-heavy shader snippets people paste
    in, without pretending to be a full C preprocessor.
    """
    macros = dict(defines or {})
    out_lines = []
    stack = []            # (parent_active, taken, active)
    lines = src.split('\n')
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        active = all(s[2] for s in stack)
        if line.startswith('#'):
            d = line[1:].strip()
            head = d.split(None, 1)[0] if d else ''
            rest = d[len(head):].strip()
            if head == 'define' and active:
                m = _IDENT.match(rest)
                if m:
                    name = m.group(0)
                    after = rest[m.end():]
                    if after.startswith('('):
                        depth = 0
                        for k, ch in enumerate(after):
                            if ch == '(':
                                depth += 1
                            elif ch == ')':
                                depth -= 1
                                if depth == 0:
                                    params = [p.strip() for p in
                                              after[1:k].split(',') if p.strip()]
                                    macros[name] = (params, after[k + 1:].strip())
                                    break
                    else:
                        macros[name] = ([], after.strip())
            elif head == 'undef' and active:
                macros.pop(rest.strip(), None)
            elif head in ('ifdef', 'ifndef'):
                cond = (rest.strip() in macros) == (head == 'ifdef')
                stack.append((active, cond and active, cond and active))
            elif head == 'if':
                cond = _eval_pp(rest, macros)
                stack.append((active, cond and active, cond and active))
            elif head == 'elif':
                if stack:
                    parent, taken, _ = stack[-1]
                    cond = (not taken) and parent and _eval_pp(rest, macros)
                    stack[-1] = (parent, taken or cond, cond)
            elif head == 'else':
                if stack:
                    parent, taken, _ = stack[-1]
                    stack[-1] = (parent, True, parent and not taken)
            elif head == 'endif':
                if stack:
                    stack.pop()
            out_lines.append('')
            i += 1
            continue
        out_lines.append(_expand(raw, macros) if active else '')
        i += 1
    return '\n'.join(out_lines)


def _eval_pp(expr, macros):
    e = expr.replace('defined', ' defined ')
    toks = re.findall(r'defined|\w+|\S', e)
    out = []
    k = 0
    while k < len(toks):
        t = toks[k]
        if t == 'defined':
            name = toks[k + 1] if k + 1 < len(toks) else ''
            if name == '(':
                name = toks[k + 2] if k + 2 < len(toks) else ''
                k += 4
            else:
                k += 2
            out.append('1' if name in macros else '0')
            continue
        if _IDENT.fullmatch(t):
            v = macros.get(t)
            out.append(v[1] if isinstance(v, tuple) and not v[0] else ('0' if v is None else '1'))
        elif t == '&&':
            out.append(' and ')
        elif t == '||':
            out.append(' or ')
        elif t == '!':
            out.append(' not ')
        else:
            out.append(t)
        k += 1
    try:
        return bool(_const_expr(out))
    except Exception:                                           # noqa: BLE001
        return False


# A preprocessor conditional is integer arithmetic and comparisons and nothing
# else, so it gets a small parser of its own rather than handing the string to
# the Python interpreter.
_PREC = (('or',), ('and',), ('|',), ('^',), ('&',), ('==', '!='),
         ('<', '>', '<=', '>='), ('<<', '>>'), ('+', '-'), ('*', '/', '%'))


def _const_expr(tokens):
    toks = []
    for t in tokens:
        t = t.strip()
        if t:
            toks.extend(t.split() if ' ' in t else [t])
    pos = [0]

    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else None

    def take():
        t = peek()
        pos[0] += 1
        return t

    def unary():
        t = peek()
        if t == 'not':
            take()
            return 0 if unary() else 1
        if t == '-':
            take()
            return -unary()
        if t == '+':
            take()
            return unary()
        if t == '~':
            take()
            return ~unary()
        if t == '(':
            take()
            v = binary(0)
            if peek() == ')':
                take()
            return v
        t = take()
        if t is None:
            return 0
        try:
            return int(t, 0)
        except ValueError:
            try:
                return int(float(t))
            except ValueError:
                return 0

    def binary(level):
        if level >= len(_PREC):
            return unary()
        left = binary(level + 1)
        while peek() in _PREC[level]:
            op = take()
            right = binary(level + 1)
            left = _APPLY[op](left, right)
        return left

    return binary(0)


_APPLY = {
    'or': lambda a, b: 1 if (a or b) else 0,
    'and': lambda a, b: 1 if (a and b) else 0,
    '|': lambda a, b: a | b, '^': lambda a, b: a ^ b, '&': lambda a, b: a & b,
    '==': lambda a, b: 1 if a == b else 0,
    '!=': lambda a, b: 1 if a != b else 0,
    '<': lambda a, b: 1 if a < b else 0, '>': lambda a, b: 1 if a > b else 0,
    '<=': lambda a, b: 1 if a <= b else 0,
    '>=': lambda a, b: 1 if a >= b else 0,
    '<<': lambda a, b: a << b, '>>': lambda a, b: a >> b,
    '+': lambda a, b: a + b, '-': lambda a, b: a - b,
    '*': lambda a, b: a * b,
    '/': lambda a, b: a // b if b else 0,
    '%': lambda a, b: a % b if b else 0,
}


def _expand(line, macros, depth=0):
    if depth > 12 or '#' in line[:1]:
        return line
    changed = False
    out = line
    for name, (params, body) in macros.items():
        if name not in out:
            continue
        if not params:
            new = re.sub(r'\b%s\b' % re.escape(name), body, out)
            if new != out:
                out, changed = new, True
        else:
            pat = re.compile(r'\b%s\s*\(' % re.escape(name))
            while True:
                m = pat.search(out)
                if not m:
                    break
                depth_p = 0
                args = []
                cur = ''
                j = m.end() - 1
                for k in range(m.end() - 1, len(out)):
                    ch = out[k]
                    if ch == '(':
                        depth_p += 1
                        if depth_p == 1:
                            continue
                    elif ch == ')':
                        depth_p -= 1
                        if depth_p == 0:
                            args.append(cur)
                            j = k
                            break
                    if depth_p == 1 and ch == ',':
                        args.append(cur)
                        cur = ''
                        continue
                    cur += ch
                rep = body
                for p, a in zip(params, args):
                    rep = re.sub(r'\b%s\b' % re.escape(p), '(' + a.strip() + ')', rep)
                out = out[:m.start()] + '(' + rep + ')' + out[j + 1:]
                changed = True
    return _expand(out, macros, depth + 1) if changed else out


def tokenize(src, hlsl=False):
    src = strip_comments(src)
    toks = []
    line = 1
    col = 1
    i = 0
    n = len(src)
    while i < n:
        c = src[i]
        if c == '\n':
            line += 1
            col = 1
            i += 1
            continue
        if c in ' \t\r':
            i += 1
            col += 1
            continue
        m = _IDENT.match(src, i)
        if m:
            word = m.group(0)
            if hlsl and word in HLSL_TYPE_ALIASES:
                word = HLSL_TYPE_ALIASES[word]
            elif word in HLSL_TYPE_ALIASES and word.startswith('float') and \
                    word not in TYPES:
                word = HLSL_TYPE_ALIASES[word]
            if word in TYPES:
                kind = 'type'
            elif word in KEYWORDS:
                kind = 'keyword'
            else:
                kind = 'ident'
            toks.append(Token(kind, word, line, col))
            col += m.end() - i
            i = m.end()
            continue
        m = _NUM.match(src, i)
        if m:
            toks.append(Token('number', m.group(0), line, col))
            col += m.end() - i
            i = m.end()
            continue
        for p in PUNCT:
            if src.startswith(p, i):
                toks.append(Token('punct', p, line, col))
                i += len(p)
                col += len(p)
                break
        else:
            raise ShaderError(f'unexpected character {c!r}', line, col)
    toks.append(Token('eof', '', line, col))
    return toks
