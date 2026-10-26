"""__aliases__ declaration/resolution - pure class-creation logic, no GPU needed."""
import pytest
import rdv


def test_alias_fixed_across_specializations():
    class Image(rdv.Component):
        __aliases__ = {rdv.Generics.VEC3PTR: rdv.Pointer[rdv.vec3]}

    assert Image.__aliases__["VEC3PTR"] is rdv.Pointer[rdv.vec3]


def test_subclass_cannot_redefine_alias():
    class Image(rdv.Component):
        __aliases__ = {rdv.Generics.VEC3PTR: rdv.Pointer[rdv.vec3]}

    with pytest.raises(TypeError):
        class Bad(Image):
            __aliases__ = {rdv.Generics.VEC3PTR: rdv.Pointer[rdv.vec4]}


def test_alias_cannot_double_as_generic():
    T = rdv.Generics.T

    with pytest.raises(TypeError):
        class Bad(rdv.Component, rdv.Generic[T]):
            __aliases__ = {T: int}


def test_alias_resolves_fresh_per_specialization():
    T = rdv.Generics.T

    class Wrapper(rdv.Struct, rdv.Generic[T]):
        __aliases__ = {rdv.Generics.PTR: rdv.Deferred[rdv.Buffer[T]]}

    float_ptr = Wrapper[rdv.Type.FLOAT32].__aliases__["PTR"]
    int_ptr = Wrapper[rdv.Type.INT32].__aliases__["PTR"]
    assert float_ptr is rdv.Deferred[rdv.Buffer[rdv.Type.FLOAT32]]
    assert int_ptr is rdv.Deferred[rdv.Buffer[rdv.Type.INT32]]
    assert float_ptr is not int_ptr


def test_alias_not_usable_as_generic_binding_target():
    class Image(rdv.Component):
        __aliases__ = {rdv.Generics.VEC3PTR: rdv.Pointer[rdv.vec3]}

    with pytest.raises(TypeError):
        Image[rdv.Generics.VEC3PTR:rdv.Pointer[rdv.vec4]]
