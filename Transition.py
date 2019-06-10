import numpy as np # type: ignore
import z3 # type: ignore

from typing import Type, TypeVar, Generic, NewType, TYPE_CHECKING, overload, Any, Tuple, Union, Callable
from dataclasses import dataclass

if not(TYPE_CHECKING):
    def reveal_type(x):
        pass

def error(msg: str):
    print("error", msg) # TODO: stack trace
    raise ValueError()

E = TypeVar('E') # Backend
F = TypeVar('F') # Numerical type
Q = TypeVar('Q')
R = TypeVar('R')
S = TypeVar('S')
T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')
W = TypeVar('W')
X = TypeVar('X')
Y = TypeVar('Y')
Z = TypeVar('Z')

# See https://keras.io/backend/#backend for inspiration
# Note this is a 'do nothing' implementation and would have to be derived for a real case
class Backend(Generic[E]):
    def __init__(self):
        self._epsilon : float = 1e-7
        self._learning_phase = True
    def backend(self) -> str:
        return 'abstract'
    def epsilon(self) -> float:
        return self._epsilon
    def set_epsilon(self, e:float) -> 'Backend[E]':
        self._epsilon = e
        return self
    def floatx(self) -> str:
        return 'float64'
    def set_floatx(self, f: str) -> 'Backend[E]':
        raise ValueError()
        return self
    def cast_to_floatx(self, n: np.ndarray) -> np.ndarray: # type: ignore
        return n.astype(float)
    def learning_phase(self) -> bool:
        return self._learning_phase
    def set_learning_phase(self, b: bool) -> 'Backend[E]':
        self._learning_phase = b
        return self
    def update(self, x: E, new_x: E) -> E:
        return x
    def update_add(self, x: E, increment: E) -> E:
        return x
    def update_sub(self, x: E, decrement: E) -> E:
        return x
    def dot(self, x: E, y: E) -> E:
        return x
    def einop(self, x: E, spec: str) -> E:
        return x
    def einop2(self, x:E, y:E, spec: str) -> E:
        return x
    def einop3(self, x:E, y:E, z:E, spec: str) -> E:
        return x
    # https://pytorch.org/docs/stable/nn.html#torch.nn.Conv1d
    # input (N, C in, L in)
    # output (N, C out, L out)
    # where L out = round lower( (l in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1 )
    def conv1d(self, x:E, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros') -> E:
        return x
    def exp(self, x: E) -> E:
        return x
    def variable(self, s: E, value, dtype=None, name=None, constraint=None) -> E:
        return s
    def constant(self, s:E, value, dtype=None, shape=None, name=None) -> E:
        return s
    def placeholder(self, s:E, shape=None, ndim=None, dtype=None, sparse=False, name=None) -> E:
        return s
    def zeros(self, s: E, shape, dtype=None, name=None) -> E:
        return s
    def ones(self, s: E, shape, dtype=None, name=None) -> E:
        return s
    def random_uniform_variable(self, s: E, shape, low, high, dtype=None, name=None, seed=None) -> E:
        return s
    def random_normal_variable(self, s: E, shape, mean, scale, dtype=None, name=None, seed=None) -> E:
        return s
    
class Tensor1(Generic[E, F, T]):
    def __init__(self, b: Backend[E], f: Type[F], t: T):
        pass
    def get_backend(self) -> Backend[E]:
        pass
    def get_handle(self) -> E:
        pass
    def set_handle(self, e: E):
        pass
    def get_order(self) -> int:
        return 1
    def get_ftype(self) -> Type[F]:
        pass
    def get_dim1_type(self) -> T:
        pass
    def get_dim1(self) -> T:
        pass        
class Tensor2(Generic[E, F, T, U]):
    def __init__(self, b: Backend[E], f: Type[F], t: T, u: U):
        pass
    def get_backend(self) -> Backend[E]:
        pass
    def get_handle(self) -> E:
        pass
    def set_handle(self, e: E):
        pass
    def get_order(self) -> int:
        return 2
    def get_ftype(self) -> Type[F]:
        pass
    def get_dim1_type(self) -> T:
        pass
    def get_dim2_type(self) -> U:
        pass
# Similarly, with order 3,4, ...        
class Tensor3(Generic[E, F, T, U, V]):
    def __init__(self, b: Backend[E], f: Type[F], t: T, u: U, v: V):
        pass
    def get_backend(self) -> Backend[E]:
        pass
    def get_handle(self) -> E:
        pass
    def set_handle(self, e: E):
        pass
    def get_order(self) -> int:
        return 3
    def get_ftype(self) -> Type[F]:
        pass
    def get_dim1_type(self) -> T:
        pass
    def get_dim2_type(self) -> U:
        pass
    def get_dim3_type(self) -> V:
        pass        
#class Tensor4(Generic[E, F, T, U, V, W]):
#    pass

# abstract dimensions A,B,C,D
class A:
    def __repr__(self) -> str:
        return "A"
class B:
    def __repr__(self) -> str:
        return "B"
class C:
    def __repr__(self) -> str:
        return "C"    
class D:
    def __repr__(self) -> str:
        return "D"


# classes that represent fixed dims (for use in dim formulas)
class _0:
    def __repr__(self)->str:
        return "0"
class _1:
    def __repr__(self)->str:
        return "1"
class _2:
    def __repr__(self)->str:
        return "2"            
class _3:
    def __repr__(self)->str:
        return "3"          
class _4:
    def __repr__(self)->str:
        return "4"          

class DimAdd(Generic[U,V]):
    def __init__(self, u:U, v: V):
        self._u = u
        self._v = v
    def __repr__(self) -> str:
        return "(" + self._u.__repr__() + "+" + self._v.__repr__() + ")"

class DimMul(Generic[U,V]):
    def __init__(self, u:U, v: V):
        self._u = u
        self._v = v
    def __repr__(self) -> str:
        return "(" + self._u.__repr__() + "*" + self._v.__repr__() + ")"

class DimSub(Generic[U,V]):
    def __init__(self, u:U, v: V):
        self._u = u
        self._v = v
    def __repr__(self) -> str:
        return "(" + self._u.__repr__() + "-" + self._v.__repr__() + ")"

class DimDiv(Generic[U,V]):
    def __init__(self, u:U, v: V):
        self._u = u
        self._v = v
    def __repr__(self) -> str:
        return "(" + self._u.__repr__() + "/" + self._v.__repr__() + ")"

class EqCert(Generic[T,U]):
    pass
class Inclusion1Cert(Generic[T,U,V]):
    pass

@dataclass
class Equality:
    left: Any
    right: Any

@dataclass
class Inclusion:
    left: Any
    right: Any

class Session(Generic[E]):
    def __init__(self):
        self._backend = Backend[E]
        dims : Dict[any, int] = {}
        self._dims = {}
    ## set dim
    def set_dim(self, a: Any, d: int):
        self._dims[a] = d
    ## get dim
    def get_dim(self, a: Any) -> int:
        return self._dims[a]
    def get_backend(self) -> Backend[E]:
        return self._backend
    def set_backend(self, b: Backend[E]) -> 'Session[E]':
        self._backend = b
        return self

class Context:
    def __init__(self):
        self.assumptions = []
        self.tocheck = []
    def assume_equal(self, d1, d2):
        self.assumptions.append(Equality(d1,d2))
    def check_equal_(self, d1: T, d2: U) -> EqCert[T,U]:
        print("check equal", d1, d2)
        self.tocheck.append(Equality(d1,d2))
        # TODO check
        return EqCert[T,U]()
    def check_in2_(self, d12: Tuple[T, U], d2: V):
        print("check is in collection", d12, d2)
        # TODO check
        self.tocheck.append(Inclusion(d12, d2))
        return Inclusion1Cert[T,U,V]()

def update1(t1: Tensor1[E,F,T], t2: Tensor1[E,F,T]) -> Tensor1[E,F,T]:
    backend = t1.get_backend()
    t1_h = t1.get_handle()
    t2_h = t2.get_handle()
    rv_h = backend.update(t1_h, t2_h)
    o = t1.get_order()
    f = t1.get_ftype()
    rv = Tensor1(backend, f, t1.get_dim1())
    rv.set_handle(rv_h)
    return rv

# similarly we would add 2D, 3D, 4D versions

def ones(s: Session[E], f: Type[F], x: T) -> Tensor1[E, F, T]:
    pass 
def ones2(s: Session[E], f: Type[F], x: T, y:U) -> Tensor2[E, F, T, U]:
    pass
def ones3(s: Session[E], f: Type[F], x: T, y:U, z: Z) -> Tensor3[E, F, T, U, Z]:
    pass

def einop_2_1(t: Tensor2[E, F, T, U], tgt: V) -> Tensor1[E,F,V]:
    pass

def einop_2_1_(c: Context, src: Tuple[T, U], tgt: V) -> Callable[[Tensor2[E, F, T, U], V], Tensor1[E,F,V]]:
    c.check_in2_(src, tgt)
    return lambda t, tg: einop_2_1(t, tg)

def cast1_(c: Context, src: T, tgt: V) -> Callable[[Tensor1[E,F,T]], Tensor1[E,F,V]]:
    c.check_equal_(src, tgt)
    def rv(t1: Tensor1[E,F,T]):
        backend = t1.get_backend()
        r = Tensor1[E,F,V](backend, t1.get_ftype(), tgt)
        t1_h = t1.get_handle()
        r.set_handle(t1_h)
        return r
    return rv

    # https://pytorch.org/docs/stable/nn.html#torch.nn.Conv1d
    # input (N, C in, L in)
    # output (N, C out, L out)
    # where L out = round lower( (l in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1 )
def conv1d_dim_(
    tgt: R,
    ctx: Context,
    l_in : S,
    in_channels : T, 
    out_channels : U, 
    kernel_size : V, 
    stride : W, 
    padding : X, 
    dilation : Y, 
    groups : Z, 
    bias=True, 
    padding_mode='zeros') -> EqCert[R, DimDiv[DimSub[DimSub[DimAdd[S, DimMul[_2,X]], DimMul[Y, DimSub[V, _1]]],_1],DimAdd[W, _1]]]:

    return ctx.check_equal_(tgt, DimDiv(
        DimSub(
            DimSub(DimAdd(l_in, DimMul(_2(),padding)), 
            DimMul(dilation, DimSub(kernel_size, _1()))),_1()),DimAdd(stride, _1())))

def conv1d_(tgt: Q,
    f: Type[F],
    ctx: Context,
    l_in : S,
    in_channels : T, 
    out_channels : U, 
    kernel_size : V, 
    stride : W, 
    padding : X, 
    dilation : Y, 
    groups : Z, 
    bias=True, 
    padding_mode='zeros') -> Callable[[Tensor3[E,F, R,T, S]], Tensor3[E,F, R, U, Q]]:
    conv1d_dim_(tgt, ctx, l_in, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
    def g(x):
        pass
    return g # type: ignore
