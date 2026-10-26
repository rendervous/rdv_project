"""Node.cast() - rebuilds an instance as a re-specialized type using its
original constructor arguments. Every test here constructs a real instance
(allocates a small host-visible buffer), so needs a Vulkan device."""
import pytest
import rdv

pytestmark = pytest.mark.gpu

T = rdv.Generics.T
X = rdv.Generics.X


class A(rdv.Struct, rdv.Generic[T, X]):
    pass


class B(A):
    tag: int

    def __init__(self, tag):
        self.tag = tag


class C(B):
    pass


class Unrelated(rdv.Struct, rdv.Generic[T]):
    pass


def test_cast_binds_shared_generic_and_replays_constructor():
    b = B(42)
    assert type(b).__generics__ == {"T": None, "X": None}

    b2 = b.cast(A[int, float])
    assert type(b2).__generics__ == {"T": int, "X": float}
    assert type(b2).__name__.startswith("B[")
    assert b2.tag == 42


def test_partial_cast_can_stay_generic():
    b = B(1).cast(A[int])
    assert type(b).__generics__ == {"T": int, "X": None}
    assert type(b).__is_generic__


def test_cast_works_through_deeper_subclass_chain():
    c = C(7).cast(A[str, str])
    assert type(c).__generics__ == {"T": str, "X": str}
    assert type(c).__name__.startswith("C[")
    assert c.tag == 7


def test_cast_to_unrelated_type_raises():
    b = B(1)
    with pytest.raises(TypeError):
        b.cast(Unrelated[int])


def test_recast_to_same_value_is_a_noop():
    b = B(1).cast(A[int])
    b2 = b.cast(A[int])
    assert type(b2).__generics__ == type(b).__generics__


def test_recast_to_conflicting_value_raises():
    b = B(1).cast(A[int])
    with pytest.raises(TypeError):
        b.cast(A[float])
