"""Generic[...] declaration/binding - pure class-creation logic, no GPU needed."""
import pytest
import rdv


def test_generic_declares_open_parameter():
    class Struct(rdv.Node):
        pass

    T = rdv.Generics.T

    class A(Struct, rdv.Generic[T]):
        pass

    assert A.__generics__ == {"T": None}
    assert A.__is_generic__


def test_old_declare_via_subscript_is_rejected():
    class Struct(rdv.Node):
        pass

    T = rdv.Generics.T

    with pytest.raises(TypeError):
        class Bad(Struct[T]):
            pass


def test_unresolved_parameter_still_generic_in_subclass():
    IN, OUT = rdv.Generics.IN, rdv.Generics.OUT

    class Map(rdv.Component, rdv.Generic[IN, OUT]):
        pass

    class Constant(Map):
        pass

    assert Constant.__generics__ == Map.__generics__ == {"IN": None, "OUT": None}
    assert Constant.__is_generic__


def test_redeclaring_inherited_generic_raises():
    IN, OUT = rdv.Generics.IN, rdv.Generics.OUT

    class Map(rdv.Component, rdv.Generic[IN, OUT]):
        pass

    with pytest.raises(TypeError):
        class Bad(Map, rdv.Generic[IN]):
            pass


def test_binding_via_subscript():
    IN, OUT = rdv.Generics.IN, rdv.Generics.OUT

    class Map(rdv.Component, rdv.Generic[IN, OUT]):
        pass

    Bound = Map[int, float]
    assert Bound.__generics__ == {"IN": int, "OUT": float}
    assert not Bound.__is_generic__


def test_binding_unknown_name_raises():
    T = rdv.Generics.T

    class Struct(rdv.Node):
        pass

    class A(Struct, rdv.Generic[T]):
        pass

    with pytest.raises(TypeError):
        A[rdv.Generics.OTHER:int]


def test_specialization_name_follows_declaration_order_not_bind_order():
    T, N = rdv.Generics.T, rdv.Generics.N

    class TensorLike(rdv.Struct, rdv.Generic[T, N]):
        pass

    Bound = TensorLike[T:float, N:3]
    names = list(Bound.__generics__.keys())
    assert names == ["T", "N"]
