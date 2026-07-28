"""The shader type system."""

from .lexer import ShaderError


class GType:
    __slots__ = ('base', 'n', 'rows', 'array', 'struct')

    def __init__(self, base, n=1, rows=0, array=0, struct=None):
        self.base = base        # float|int|uint|bool|void|sampler|struct
        self.n = n              # components (vector) or columns (matrix)
        self.rows = rows        # 0 for scalars/vectors, else matrix rows
        self.array = array      # 0 = not an array
        self.struct = struct

    # ------------------------------------------------------------- predicates
    @property
    def is_scalar(self):
        return self.rows == 0 and self.n == 1 and self.base in ('float', 'int', 'uint', 'bool')

    @property
    def is_vector(self):
        return self.rows == 0 and self.n > 1 and self.base != 'struct'

    @property
    def is_matrix(self):
        return self.rows > 0

    @property
    def is_numeric(self):
        return self.base in ('float', 'int', 'uint')

    @property
    def comps(self):
        return self.n * self.rows if self.is_matrix else self.n

    def elem(self):
        if self.array:
            return GType(self.base, self.n, self.rows, 0, self.struct)
        if self.is_matrix:
            return GType(self.base, self.rows, 0)
        return GType(self.base, 1, 0)

    def scalar(self):
        return GType(self.base, 1, 0)

    def with_base(self, base):
        return GType(base, self.n, self.rows, self.array, self.struct)

    def __eq__(self, o):
        return (isinstance(o, GType) and self.base == o.base and self.n == o.n
                and self.rows == o.rows and self.array == o.array
                and self.struct == o.struct)

    def __hash__(self):
        return hash((self.base, self.n, self.rows, self.array, self.struct))

    def __repr__(self):
        return self.name()

    def name(self):
        if self.base == 'struct':
            s = self.struct
        elif self.base == 'sampler':
            s = 'sampler2D'
        elif self.is_matrix:
            s = f'mat{self.n}' if self.n == self.rows else f'mat{self.n}x{self.rows}'
        elif self.n > 1:
            pre = {'float': '', 'int': 'i', 'uint': 'u', 'bool': 'b'}[self.base]
            s = f'{pre}vec{self.n}'
        else:
            s = self.base
        return s + (f'[{self.array}]' if self.array else '')


VOID = GType('void')
FLOAT = GType('float')
INT = GType('int')
UINT = GType('uint')
BOOL = GType('bool')
VEC2 = GType('float', 2)
VEC3 = GType('float', 3)
VEC4 = GType('float', 4)
IVEC2 = GType('int', 2)
IVEC3 = GType('int', 3)
IVEC4 = GType('int', 4)
BVEC2 = GType('bool', 2)
BVEC3 = GType('bool', 3)
BVEC4 = GType('bool', 4)
MAT2 = GType('float', 2, 2)
MAT3 = GType('float', 3, 3)
MAT4 = GType('float', 4, 4)
SAMPLER = GType('sampler')

BY_NAME = {
    'void': VOID, 'float': FLOAT, 'double': FLOAT, 'int': INT, 'uint': UINT,
    'bool': BOOL,
    'vec2': VEC2, 'vec3': VEC3, 'vec4': VEC4,
    'ivec2': IVEC2, 'ivec3': IVEC3, 'ivec4': IVEC4,
    'uvec2': GType('uint', 2), 'uvec3': GType('uint', 3), 'uvec4': GType('uint', 4),
    'bvec2': BVEC2, 'bvec3': BVEC3, 'bvec4': BVEC4,
    'mat2': MAT2, 'mat3': MAT3, 'mat4': MAT4,
    'mat2x2': MAT2, 'mat3x3': MAT3, 'mat4x4': MAT4,
    'sampler2D': SAMPLER, 'samplerCube': SAMPLER, 'sampler3D': SAMPLER,
    'sampler2DShadow': SAMPLER,
}

SWIZZLE_SETS = ('xyzw', 'rgba', 'stpq')


def swizzle_indices(field):
    """'xyz' / 'rgb' / 'st' -> [0,1,2]; returns None if not a swizzle."""
    if not field or len(field) > 4:
        return None
    for s in SWIZZLE_SETS:
        if all(c in s for c in field):
            return [s.index(c) for c in field]
    return None


def promote(a, b, line=0):
    """Result type of an arithmetic binop."""
    if a.is_matrix or b.is_matrix:
        return a if a.is_matrix else b
    base = 'float'
    if a.base == b.base:
        base = a.base
    elif 'float' in (a.base, b.base):
        base = 'float'
    elif 'uint' in (a.base, b.base):
        base = 'uint'
    else:
        base = 'int'
    n = max(a.n, b.n)
    if a.n != b.n and a.n != 1 and b.n != 1:
        raise ShaderError(f'cannot combine {a.name()} with {b.name()}', line)
    return GType(base, n)
