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

class AbstractTensor:
    pass

# see https://keras.io/backend/#backend for inspiration
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
    def variable(self, value, dtype=None, name=None, constraint=None) -> E:
        pass
    def constant(self, value, dtype=None, shape=None, name=None) -> E:
        pass
    def placeholder(self, shape=None, ndim=None, dtype=None, sparse=False, name=None) -> E:
        pass
    def zeros(self, shape, dtype=None, name=None) -> E:
        pass
    def ones(self, shape, dtype=None, name=None) -> E:
        pass
    def random_uniform_variable(self, shape, low, high, dtype=None, name=None, seed=None) -> E:
        pass
    def random_normal_variable(self, shape, mean, scale, dtype=None, name=None, seed=None) -> E:
        pass
    
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
    def get_dim1_type(self) -> Type[T]:
        pass
    def get_dim1(self) -> T:
        pass        
class Tensor2(Generic[E, F, T, U]):
    def __init__(self, b: Backend[E], f: Type[F], t: Type[T], u: Type[U]):
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
    def get_dim1_type(self) -> Type[T]:
        pass
    def get_dim2_type(self) -> Type[U]:
        pass
class Tensor3(Generic[E, F, T, U, V]):
    pass
class Tensor4(Generic[E, F, T, U, V, W]):
    pass

class A:
    pass
class B:
    pass
class C:
    pass
class D:
    pass

class Succ(Generic[T]):
    pass

class _0:
    pass
_1 = Succ[_0]
_2 = Succ[_1]
_3 = Succ[_2]
_4 = Succ[_3]
_5 = Succ[_4]

class DimAdd(Generic[U,V]):
    def __init__(self, u:U, v: V):
        self._u = u
        self._v = v

class DimMul(Generic[U,V]):
    def __init__(self, u:U, v: V):
        self._u = u
        self._v = v

class DimSub(Generic[U,V]):
    def __init__(self, u:U, v: V):
        self._u = u
        self._v = v

class DimDiv(Generic[U,V]):
    def __init__(self, u:U, v: V):
        self._u = u
        self._v = v


@dataclass
class Equality:
    left: Any
    right: Any

@dataclass
class DimSumAST:
    left: Any
    right: Any    

class EqCert(Generic[T,U]):
    pass

class Session(Generic[E]):
    def __init__(self):
        self._backend = Backend[E]
    ## set dim
    ## get dim
    def get_dim(a: Any) -> int:
        return 0
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
    def check_equal(self, d1: Type[T], d2: Type[U]) -> EqCert[T,U]:
        self.tocheck.append(Equality(d1,d2))
        # TODO check
        return EqCert[T,U]()
    # no
    def cast(self, s: Session, t1: Tensor1[E, F, U], ttx:EqCert[U, T]) -> Tensor1[E, F, T]:
        #self.check_equal(tx, t2)
        pass # Tensor1[E, F,T]()    

## def conv1d(self, x:E, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros') -> E:
## size type creator...

#@overload
#def update(t1: Tensor1[E,F,T], t2: Tensor1[E,F,T]) -> Tensor1[E,F,T]: ...
#@overload
#def update(t1: Tensor2[E,F,T,U], t2: Tensor2[E,F,T,U]) -> Tensor2[E,F,T,U]: ...

#todo update 1/2/3/4
def update(t1: Tensor1[E,F,T], t2: Tensor1[E,F,T]) -> Tensor1[E,F,T]:
    backend = t1.get_backend()
    t1_h = t1.get_handle()
    t2_h = t2.get_handle()
    rv_h = backend.update(t1_h, t2_h)
    o = t1.get_order()
    f = t1.get_ftype()
    #if o == 1:
    rv = Tensor1(backend, f, t1.get_dim1())
    rv.set_handle(rv_h)
    return rv
    #if o == 2:
    #    rv = Tensor1(backend, f, t1.get_dim1_type(), t1.get_dim2_type())
    #    rv.set_handle(rv_h)
    #    return rv
    #error("update / tensor size")


def ones(s: Session[E], f: Type[F], x: T) -> Tensor1[E, F, T]:
    pass 
def ones2(s: Session[E], f: Type[F], x: T, y:U) -> Tensor2[E, F, T, U]:
    pass
def ones3(s: Session[E], f: Type[F], x: T, y:U, z: V) -> Tensor3[E, F, T, U, V]:
    pass

## crap
@overload
def ones_(s: Session[E], f: Type[F], x: Tuple[Type[T]], ty: Type[Tensor1[E, F, T]]) -> Tensor1[E, F, T]: ...
@overload
def ones_(s: Session[E], f: Type[F], x: Tuple[Type[T], Type[U]], ty: Type[Tensor2[E, F, T, U]]) -> Tensor2[E, F, T, U]: ...

def ones_(s, f, x, ty):
    pass

def ze_ones(s, f, x):
    print(x)
    backend = s.get_backend()
    if (len(x) == 1):
        rv = Tensor1(backend, f, x[0])
        h = backend.ones(shape = (s.get_dim(x[0])), dtype = f)
        rv.set_handle(h)
        return rv
    if (len(x) == 2):
        rv = Tensor2(backend, f, x[0], x[1])
        h = backend.ones(shape = (s.get_dim(x[0]), s.get_dim(x[1])), dtype = f)
        rv.set_handle(h)
        return rv
    error("ones / shape size")

def einop_2_1(t: Tensor2[E, F, T, U], tgt: V) -> Tensor1[E,F,V]:
    pass

#non
class Einop_2_1(Generic[T, V, U]):
    pass

def einop_2_1_(src: Tuple[T, U], tgt: V) -> Callable[[Tensor2[E, F, T, U], V], Tensor1[E,F,V]]:
    print("todo check:", src, tgt)
    return lambda t, tg: einop_2_1(t, tg)

def cast1_(c: Context, src: T, tgt: V) -> Callable[[Tensor1[E,F,T]], Tensor1[E,F,V]]:
    print("todo check:", src, tgt)
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
    pass

#no
def conv1d(t1: Tensor3[E,F, R,T, S],
    in_channels : T, 
    out_channels : U, 
    kernel_size : V, 
    stride : W, 
    padding : X, 
    dilation : Y, 
    groups : Z, 
    bias=True, 
    padding_mode='zeros') -> Tensor3[E,F, R, U, DimDiv[DimSub[DimSub[DimAdd[S, DimMul[_2,X]], DimMul[Y, DimSub[V, _1]]],_1],DimAdd[W, _1]]] :
    pass

"""
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
    def variable(self, value, dtype=None, name=None, constraint=None) -> E:
        return E()
    def constant(self, value, dtype=None, shape=None, name=None) -> E:
        return E()
    def placeholder(self, shape=None, ndim=None, dtype=None, sparse=False, name=None) -> E:
        return E()
    def zeros(self, shape, dtype=None, name=None) -> E:
        return E()
    def ones(self, shape, dtype=None, name=None) -> E:
        return E()
    def random_uniform_variable(self, shape, low, high, dtype=None, name=None, seed=None) -> E:
        return E()
    def random_normal_variable(self, shape, mean, scale, dtype=None, name=None, seed=None) -> E:
        return E()
"""

"""
class EE:
    pass

def example01(s: Session[EE]):
    t1: Tensor1[EE, float, A] = ones(s, float, A())
    t2: Tensor1[EE, float, B] = ones(s, float, B())
    t3: Tensor1[EE, float, A] = ones_(s, float, (B,), Tensor1[EE, float, B])
    update(t1, t2)
    reveal_type(t1)
    reveal_type(t2)
    reveal_type(t3)

"""
