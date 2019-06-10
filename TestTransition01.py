import Transition
from typing import Type, TypeVar, Generic, NewType, TYPE_CHECKING, overload, Any, Tuple, Union, Callable

ctx: Transition.Context = Transition.Context()

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
class E:
    def __repr__(self) -> str:
        return "E"
class Q:
    def __repr__(self) -> str:
        return "Q"

T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')
W = TypeVar('W')

einop_AB_B: Callable[
    [Transition.Tensor2[E, float, A, B], B], Transition.Tensor1[E, float, B]] = Transition.einop_2_1_(ctx, (A(),B()), B())

conv1d : Callable[[Transition.Tensor3[E,float, A,B, C]], Transition.Tensor3[E,float, A, B, Q]] = Transition.conv1d_(Q(), float, ctx, C(), B(), B(), Transition._4(), Transition._3(), Transition._3(), Transition._3(), Transition._3()) 

def example01(s: Transition.Session[E]):
    t1: Transition.Tensor1[E, float, A] = Transition.ones(s, float, A())
    t2: Transition.Tensor1[E, float, A] = Transition.ones(s, float, A())
    Transition.update1(t1, t2)

    t3: Transition.Tensor2[E, float, A, B] = Transition.ones2(s, float, A(), B())
    t4: Transition.Tensor1[E, float, B] = Transition.einop_2_1(t3, B())

    t5: Transition.Tensor1[E, float, B] = einop_AB_B(t3, B())

    t6: Transition.Tensor3[E, float, A, B, C] = Transition.ones3(s, float, A(), B(), C())
    t7: Transition.Tensor3[E, float, A, B, Q] = conv1d(t6)

