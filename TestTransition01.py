import Transition
from typing import Type, TypeVar, Generic, NewType, TYPE_CHECKING, overload, Any, Tuple, Union, Callable

ctx: Transition.Context = Transition.Context()

class A:
    pass
class B:
    pass
class C:
    pass
class D:
    pass
class E:
    pass
class Q:
    pass

class TM(type):
    @classmethod
    def __class_getitem__(cls, item):
        print("TM", str(cls), str(item))    
        return str(cls)
    @classmethod
    def __getitem__(cls, item):
        print("TM2", str(cls), str(item))  
        def emu():
            return str(cls)
        return emu

T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')
W = TypeVar('W')

class TR(Generic[T], metaclass = TM):
    def id(self, x):
        return x

class XY(Generic[T,U]):
    pass

einop_AB_B: Callable[
    [Transition.Tensor2[E, float, A, B], B], Transition.Tensor1[E, float, B]] = Transition.einop_2_1_((A(),B()), B())

#conv1dim = Transition.conv1d_dim_(A(), B(), D(), Transition._4(), Transition._3(), Transition._3(), Transition._3(), Transition._3()) 
#reveal_type(conv1dim)
conv1d : Callable[[Transition.Tensor3[E,float, A,B, C]], Transition.Tensor3[E,float, A, B, Q]] = Transition.conv1d_(Q(), float, ctx, C(), B(), B(), Transition._4(), Transition._3(), Transition._3(), Transition._3(), Transition._3()) 
# reveal_type(conv1d)

def example01(s: Transition.Session[E]):
    t1: Transition.Tensor1[E, float, A] = Transition.ones(s, float, A())
    t2: Transition.Tensor1[E, float, A] = Transition.ones(s, float, A())
    #t3: Transition.Tensor1[E, float, A] = Transition.ones_(s, float, (B,), Transition.Tensor1[E, float, B])
    Transition.update(t1, t2)
    reveal_type(t1)
    reveal_type(t2)
    #reveal_type(t3)
    t3: Transition.Tensor2[E, float, A, B] = Transition.ones2(s, float, A(), B())
    t4: Transition.Tensor1[E, float, B] = Transition.einop_2_1(t3, B())
    reveal_type(t4)
    t5: Transition.Tensor1[E, float, B] = einop_AB_B(t3, C())
    reveal_type(t5)
    tr: TR[XY[A,B]] = TR[XY[A,B]]()
    t6: Transition.Tensor3[E, float, A, B, C] = Transition.ones3(s, float, A(), B(), C())
    t7: Transition.Tensor3[E, float, A, C, Q] = conv1d(t6)
    reveal_type(t7)

tr2: TR[XY[A,C]] = TR[XY[A,C]]()

### TODO: try to catch bad dim extraction BEFORE running the code
"""
class AddDim0(type):
    @classmethod
    def __class_getitem__(cls, item):
        print(type(cls), cls.__dict__)
        print(type(item), item.__dict__)
        print("***getitem " + str(cls) + " " + str(item))
        return f"{str(cls)}[int]"
    @classmethod
    def __getitem__(cls, item):
        print(type(cls), cls.__dict__)
        print(type(item), item.__dict__)
        print("***getitem " + str(cls) + " " + str(item))
        return f"{str(cls)}[float]"        

class AddDim(Generic[T], metaclass = AddDim0):
    pass
    """