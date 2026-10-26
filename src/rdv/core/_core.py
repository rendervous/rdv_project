from functools import cached_property
import typing as _typing
import re as _re
import itertools as _itertools
import os as _os
import inspect as _inspect
import time as _time
import math as _math
import weakref as _weakref

from vk.vk import EngineType

from vk import (
    caps,
    Type,
    MemoryLocation,
    DescriptorType,
    vec2,
    vec3,
    vec4,
    mat2,
    mat3,
    mat3x4,
    mat4x3,
    mat4,
    ivec2,
    ivec3,
    ivec4,
    uvec2,
    uvec3,
    uvec4,
    bvec2,
    bvec3,
    bvec4,
    device,
    device_index,
    Format,
    Window,
    window,
    FloatState,
    IntState,
    BoolState,
    ColorState,
    StringState,
    CanvasState,
    MouseState,
)

from vk.math3d import *

import vk as _vk


__PROFILING__ = False


def profiling(enable=True):
    global __PROFILING__
    __PROFILING__ = enable


_HLSL_DIR = _os.path.dirname(_os.path.dirname(__file__))
"""
rdv package root (containing shaders/ - this file itself lives one level
deeper, in rdv/core/), passed as from_glsl's include_dirs so
`#include "shaders/core/tensor.h"`-style directives (e.g. Tensor._generate_code)
resolve relative to it.
"""


def _as_include(code, owner):
    """
    __code__/__precode__/__poscode__ may be given as either literal GLSL
    text, or a path (e.g. "shaders/core/tensor.h") to a file under _HLSL_DIR
    holding it instead, or a path to a file sitting right next to `owner`'s
    own declaring .py file (e.g. a Map subclassed in some_package/foo.py,
    with __code__ = "foo.h" placed in that same directory - lets a node
    defined outside the rdv package itself, e.g. in a downstream project
    like rdv_nr, ship its own shader file alongside its own Python module
    instead of needing it under _HLSL_DIR) - in either path case this
    returns the #include directive for it (resolved via include_dirs - see
    _build_pipeline, which collects owner.__include_dir__ from every node
    in the dependency order this got recorded against, below) rather than
    the bare path itself, which would otherwise be emitted verbatim into
    the generated shader and fail as a nonsense identifier. Literal GLSL
    text is returned unchanged - it doesn't exist as a file under either
    root, so neither check below matches.
    """
    if not code:
        return code
    if _os.path.exists(_os.path.join(_HLSL_DIR, code)):
        return f'#include "{code}"'
    try:
        own_dir = _os.path.dirname(_inspect.getfile(owner))
    except TypeError:
        own_dir = None
    if own_dir and _os.path.exists(_os.path.join(own_dir, code)):
        owner.__include_dir__ = own_dir
        return f'#include "{code}"'
    return code


# ===== TOOL CLASSES =====

class _Indexer:
    """
    Tool class to assign unique ids to signatures or objects.
    Notice the implication is s1 != s2 => i1 != i2, not the other way around.
    """
    def __init__(self, all_time_different: bool = False):
        self._ids = { }
        self._signatures = { }
        self._all_time_different = all_time_different

    def contains(self, signature):
        """
        Determines whether this signature has already assigned an id.
        """
        return signature in self._signatures

    def idx(self, signature: _typing.Any):
        """
        Assigns unique ids to signatures.
        """
        if not self._all_time_different:
            id = self._ids.setdefault(signature, len(self._ids))
        else:
            id = len(self._ids)
        self._signatures[id] = signature
        return id

    def signature(self, id: int):
        """
        Gets the object/signature for which this id was assigned.
        """
        return self._signatures[id]


class _classproperty:
    """
    Like functools.cached_property, but for a value derived purely from the
    class itself (e.g. is_reference, computed only from
    cls._fields_layout_decription()) - accessible, and cached, via the class
    directly (SomeClass.attr), not just through an instance.
    functools.cached_property alone doesn't work for this: its __get__
    special-cases instance=None (i.e. class-level access) by returning the
    descriptor object itself, unchanged - always truthy, regardless of the
    wrapped computation - rather than actually invoking the function; stacking
    @classmethod underneath doesn't fix that; it's cached_property's own
    class-level branch that never runs the function.
    """
    def __init__(self, func):
        self.func = func
        self.attrname = func.__name__
        self.__doc__ = func.__doc__

    def __set_name__(self, owner, name):
        self.attrname = name

    def __get__(self, instance, owner=None):
        if owner is None:
            owner = type(instance)
        value = self.func(owner)
        # Cached directly on owner's own dict (shadowing this descriptor,
        # inherited from wherever it was first defined) - every concrete
        # subclass gets its own cached value the first time *it* is
        # accessed, since the computation depends on the subclass's own
        # _fields_layout_decription(), not whichever class first defined
        # this property.
        setattr(owner, self.attrname, value)
        return value


class RDVType:
    """
    Type used to describe a mechanism to integrate any external
    type into the RDV node parameter types.
    """
    @classmethod
    def _fields_layout_decription(cls) -> Type | list | dict:
        """
        When implemented gives the description of the type
        of this object in the form of a dict, list or simple type.
        Allows to convert field declaration into the expanded version
        for layout and accessing, e.g., v: MyStruct -> v: dict(fields)
        """
        raise NotImplementedError()

    @_classproperty
    def is_reference(cls):
        """
        Returns true only if the construction T(ptr) is valid for this type T.
        """
        layout_description = cls._fields_layout_decription()
        if layout_description == Type.UINT64:
            return True
        if not isinstance(layout_description, dict):
            return False
        if len(layout_description) != 1:
            return False
        v, = layout_description.values()
        return v == Type.UINT64

    def _fields_value(self) -> int | float | dict | list:
        """
        When implemented gives the value or dict of value to
        be assigned if this object is set. For instance,
        map.my_field = my_struct
        """
        raise NotImplementedError()

    @_classproperty
    def is_inline_rdvtype(cls):
        """
        Cached (see _classproperty) equivalent of the old _is_inline_rdvtype(field_type)
        free function - True for a non-Struct RDVType (e.g. Tensor) whose own
        _fields_layout_decription() is a composite, inlined into its
        container's own Accessor the same way a Struct is (see
        Accessor._sub/_set_inline_rdvtype), as opposed to a plain pointer
        reference (is_reference, e.g. any Code - Component/Main/... - or
        Pointer/Buffer/Deferred). Only meaningful once a caller has already
        established cls is actually a Node/RDVType (isinstance(field_type,
        _NodeMeta) and issubclass(field_type, RDVType)) - a field can be a
        plain dict/list/Type/int too, none of which have this property.
        """
        return not issubclass(cls, Struct) and not cls.is_reference

    @_classproperty
    def is_compound(cls):
        """
        Cached (see _classproperty) equivalent of the old
        Accessor._is_compound(field_type) free function, for when
        field_type is already known to be a Node/RDVType (see
        is_inline_rdvtype above) - True for a Struct or any other inlined
        RDVType (is_inline_rdvtype, e.g. Tensor), False for a plain
        pointer reference. Accessor._is_compound itself still handles the
        non-RDVType cases (a plain dict/list field description).
        """
        return issubclass(cls, Struct) or cls.is_inline_rdvtype


def resolve_ptr(obj: _typing.Any):
    if obj is None:
        return 0
    if isinstance(obj, int):
        return obj
    try:
        return obj._fields_value()
    except:
        if getattr(type(obj), '__is_generic__', False):
            raise RuntimeError(
                f"Can not retrieve a ptr for a generic node with type {type(obj)} - "
                f"one or more of its generic parameters (its own IN/OUT, or a still-"
                f"generic child argument buried inside it) were never resolved to a "
                f"concrete value, so it has no concrete GPU field layout to point at. "
                f"Look for a constructor argument built purely from other generic "
                f"nodes (e.g. an X[...] expression) that never got cast()/pushed down "
                f"to a concrete IN/OUT before this instance was used.")
        raise RuntimeError(f"Can not retrieve a ptr from type {type(obj)}")
    # if isinstance(obj, RDVType):
    #     assert type(obj).is_reference, "Only references can be converted to a ptr"
    #     value = obj._fields_value()
    #     if isinstance(value, int):
    #         return value
    #     value, = value.values()
    #     return value
    # raise RuntimeError(f"Can not retrieve a ptr from type {type(obj)}")


# ===== DATA WRAPPER CLASS =====


class RDVObject(RDVType):
    """
    Defines rdv own representation for host data: tensors, vk tensors, buffers, etc.
    Internally wraps any dlpack object into the current device copying to a vk memory if neeeded.
    """
    def __init__(self, data):
        self._data = data
        self._wrap = _vk.wrap(data)
        if not self._wrap.is_direct:
            self._wrap.update_gpu() # updates the content to the gpu
        self._device_index = _vk.device_index()
        self._tensor_cache = {}

    @cached_property
    def memory(self):
        return self._wrap

    @cached_property
    def device_ptr(self):
        return self._wrap.device_ptr

    @cached_property
    def device_index(self):
        return self._device_index

    @cached_property
    def shape(self):
        return self._wrap.shape

    @cached_property
    def scalar_type(self):
        return self._wrap.scalar_type

    @cached_property
    def is_direct(self):
        """
        Returns whether the tensor is directly accessible on the GPU or not.
        If it is not direct, it means that the tensor is a copy of a CPU tensor
        and needs to be updated before use.
        """
        return self._wrap.is_direct

    def as_tensor(self, backend):
        """
        Returns the directly accessible tensor in the specified backend.
        The backend can be 'torch', 'jax', 'numpy', 'cupy' or 'tensorflow'.
        If this Tensor wraps our own native vk.Tensor/vk.Buffer, its DLPack
        export is used directly (e.g. a device-only tensor still exports via
        vulky's own CUDA interop) rather than going through WrappedMemory,
        whose __dlpack__ requires a host/CUDA-visible pointer of its own and
        would otherwise reject a plain device-local allocation.

        Cached per backend: _data/_wrap never change once this instance
        exists, so the returned tensor is a live view over the same memory
        regardless of how many times this is called - re-exporting via
        DLPack (torch.from_dlpack et al - a measurable per-call cost, e.g.
        in a tight render loop calling this every frame) every time would
        just rebuild an equivalent view, not pick up anything new.
        """
        if backend not in ('torch', 'jax', 'numpy', 'cupy', 'tensorflow'):
            raise NotImplementedError(f'Unknown backend {backend}. Supported backends are: torch, jax, numpy, cupy, tensorflow.')
        cached = self._tensor_cache.get(backend)
        if cached is not None:
            return cached
        if isinstance(self._data, (_vk.Tensor, _vk.Buffer)):
            result = getattr(self._data, backend)()
        else:
            result = getattr(self._wrap, f'as_{backend}')()
        self._tensor_cache[backend] = result
        return result

    def make_host_dirty(self):
        self._wrap.make_cpu_dirty()

    def make_device_dirty(self):
        self._wrap.make_gpu_dirty()

    def update_host(self):
        self._wrap.update_cpu()

    def update_device(self):
        self._wrap.update_gpu()

    def host(self) -> 'RDVObject':
        """A copy of this object's current data on host memory."""
        self._wrap.update_cpu()
        host_tensor = _vk.tensor(self.shape, self._wrap.scalar_type, location=_vk.MemoryLocation.HOST)
        host_tensor.load(self._wrap)
        return RDVObject(host_tensor)

    def device(self) -> 'RDVObject':
        """A copy of this tensor's current data on device memory."""
        device_tensor = _vk.tensor(self.shape, self._wrap.scalar_type, location=_vk.MemoryLocation.DEVICE)
        device_tensor.load(self._wrap)
        return RDVObject(device_tensor)

    @classmethod
    def _fields_layout_decription(cls):
        return Type.UINT64

    def _fields_value(self):
        return self.device_ptr


def wrap(data: _typing.Any) -> RDVObject:
    if isinstance(data, RDVObject):
        return data
    return RDVObject(data)


# ===== CORE NODE DEFINITION =====

class GenericParam:
    """
    Base marking a type as a generic parameter:
        class T(GenericParam): pass
    or, via a Generics container (preferred - see Generics below):
        T = Generics.T
    (A bare `GenericParam('T')` factory shorthand was tried too, but PyCharm/
    mypy don't use a metaclass __call__'s return type to infer what
    ClassName(...) produces - T ended up inferred as Any regardless of
    annotation, so that form was dropped.)
    Never instantiated - GenericParam subclasses are only ever used as type
    markers (in annotations and __generics__), so no instance behavior (get/
    set/etc.) is defined; a: T / instance.a = anything is plain, unrelated
    attribute access on the Node instance, not on T.
    """
    pass


_GP = _typing.TypeVar('_GP', bound='GenericParam')


class GenericsMeta(type):
    """Metaclass making a class act as a container of generic parameters (see Generics)."""
    __cache__ = {}

    def __getattr__(cls: type[_GP], item) -> type[_GP]:
        # Generics.T lazily creates a fresh GenericParam subclass named T the
        # first time it's accessed; every later access (Generics.T again, from
        # anywhere) returns that exact same cached type, never a new one.
        key = (cls, item)
        if key not in GenericsMeta.__cache__:
            GenericsMeta.__cache__[key] = type(item, (cls,), {})
        return GenericsMeta.__cache__[key]

    def __eq__(cls, other):
        # Generics.T == some_value (e.g. Type.UINT64) - used as an
        # __aliases__ value (see _GenericComparison/Node.__aliases__) to
        # generate different code depending on what this type's own T
        # eventually resolves to - is deliberately NOT plain identity
        # comparison here. But Generics.T == <another param, or a Node
        # type> must stay ordinary identity comparison: that's what the
        # rest of _NodeMeta's own generic-binding/resolution machinery
        # (_merge_binding, _specialize_type, ...) actually relies on when
        # a generic happens to be bound to (or aliased to) another type.
        if _is_param(other) or isinstance(other, _NodeMeta):
            return cls is other
        return _GenericComparison(cls, '==', other)

    def __ne__(cls, other):
        if _is_param(other) or isinstance(other, _NodeMeta):
            return cls is not other
        return _GenericComparison(cls, '!=', other)

    # type.__eq__ being overridden above would otherwise silently make every
    # GenericParam subclass unhashable (defining __eq__ without __hash__
    # sets it to None) - fatal, since these are used as dict/frozenset
    # members throughout _NodeMeta (e.g. _specialization_cache's key).
    # Identity-based, matching the identity-based __eq__ branch above.
    __hash__ = type.__hash__

    # Arithmetic (Generics.IN + 1, 2 * Generics.N, ...) - unlike ==/!=,
    # there's no identity-comparison special case to preserve here (nothing
    # else in _NodeMeta relies on +/-/*/ meaning identity), so this always
    # just builds a deferred _GenericArithmetic node, the same way == always
    # would if _GenericComparison didn't need that carve-out.
    def __add__(cls, other):
        return _GenericArithmetic(cls, '+', other)

    def __radd__(cls, other):
        return _GenericArithmetic(other, '+', cls)

    def __sub__(cls, other):
        return _GenericArithmetic(cls, '-', other)

    def __rsub__(cls, other):
        return _GenericArithmetic(other, '-', cls)

    def __mul__(cls, other):
        return _GenericArithmetic(cls, '*', other)

    def __rmul__(cls, other):
        return _GenericArithmetic(other, '*', cls)

    def __truediv__(cls, other):
        return _GenericArithmetic(cls, '/', other)

    def __rtruediv__(cls, other):
        return _GenericArithmetic(other, '/', cls)

    def __call__(cls, x):
        """
        T(x): the value bound to this generic parameter T in x's own type
        (x a Node type, e.g. T(Map[2, 3])) or in type(x) (x a Node
        instance, e.g. T(some_map)) - shorthand for x.generic_value(T) /
        type(x).generic_value(T). GenericParam subclasses are never meant
        to be instantiated (see GenericParam's own docstring) - this
        replaces normal type-call/instantiation semantics for them
        entirely, rather than adding to it.

        T(Generics.G) (x itself still an unresolved generic parameter,
        e.g. inside an __aliases__ = {...} class-body dict, before G is
        bound to anything) can't resolve immediately - there's no owner
        type yet to ask - so this instead returns a deferred
        _GenericValueOf, resolved by _specialize_type once G is bound to a
        concrete Node type, the same way _GenericComparison/
        _GenericArithmetic defer Generics.T == value / Generics.T + 1.
        """
        if _is_param(x):
            return _GenericValueOf(cls, x)
        owner = x if isinstance(x, _NodeMeta) else type(x)
        return owner.generic_value(cls)


class Generics(GenericParam, metaclass=GenericsMeta):
    """
    Container of generic parameters: `Generics.T` is equivalent to
    `class T(GenericParam): pass`, but declared on demand as an attribute
    access instead of needing an explicit class statement per parameter.

    A Generics.T name can also be used to declare a fixed __aliases__ entry
    (`__aliases__ = {Generics.T: SomeType}`) instead of a genuine open
    __generics__ parameter - see Node.__aliases__.
    """
    pass


def _is_param(arg) -> bool:
    return isinstance(arg, type) and issubclass(arg, GenericParam)


def _is_unresolved(value) -> bool:
    """
    True for a generic binding that isn't really settled yet, so should
    still be freely (re)bindable to a more-resolved value instead of
    raising a conflict the way a genuinely fixed value would: None (never
    bound), a bare still-open GenericParam (see _is_param), or a Node type
    that's itself still generic (e.g. one of Select's own generics bound
    to Identity - the whole, still fully-generic class - before whatever
    Identity gets cast to becomes known; see Select.__init__/
    Composite.__init__ pushing a known IN down into a still-generic
    sub-map). Used by _merge_binding/Node.match/__is_generic__ so a class
    with a generic bound this way is itself still treated as generic
    (deferring _check_concrete_field_types) instead of being mistaken for
    finished just because every name in __generics__ happens to have SOME
    value.
    """
    return value is None or _is_param(value) or (isinstance(value, _NodeMeta) and value.__is_generic__)


class _GenericComparison:
    """
    Result of Generics.T == value / Generics.T != value (see
    GenericsMeta.__eq__/__ne__) - a deferred comparison, not a bool: T isn't
    bound to a concrete value yet at the point this gets built (e.g. right
    inside an __aliases__ = {...} class-body dict, before the owning class
    even exists). Only meaningful as an __aliases__ value (see
    Node.__aliases__) - resolved by _specialize_type, the same pass that
    already substitutes a Node-typed alias's own generic references, to 1 or
    0 once the owning class's own T is known; still unresolved (returned
    as-is), same as any other still-open alias value, if it isn't yet.
    """
    def __init__(self, param, op, value):
        self.param = param
        self.op = op
        self.value = value

    def __repr__(self):
        return f'{self.param.__name__} {self.op} {self.value!r}'

    def __bool__(self):
        # Using it directly as a plain bool (e.g. in an `if`) before it's
        # gone through _specialize_type's resolution would silently produce
        # a meaningless-but-always-truthy result instead - fail loudly.
        raise TypeError(f'{self!r} is a deferred comparison (only resolved as an __aliases__ value, '
                         f'once bound) - it is not itself a bool yet.')


class _GenericArithmetic:
    """
    Result of an arithmetic expression (Generics.IN + 1, 2 * Generics.N, ...)
    between a generic parameter and an int - or another such expression, so
    these compose freely (Generics.IN - 1 + 2) - see GenericsMeta.__add__ and
    friends. Like _GenericComparison, a deferred value, not an int: resolved
    by _specialize_type (and, while still unresolved, tolerated the same way
    a bare still-open param is - see _normalize_field) once every parameter
    it references is bound. '/' always means integer division (//) - a
    generic parameter (an array count, a tensor rank, ...) is always an int,
    never a float.
    """
    _OPS = {
        '+': lambda a, b: a + b,
        '-': lambda a, b: a - b,
        '*': lambda a, b: a * b,
        '/': lambda a, b: a // b,
    }

    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

    def __repr__(self):
        def name(v):
            return v.__name__ if _is_param(v) else repr(v)
        return f'({name(self.left)} {self.op} {name(self.right)})'

    def __bool__(self):
        raise TypeError(f'{self!r} is a deferred arithmetic expression (only resolved once every '
                         f'parameter it references is bound) - it is not itself a number yet.')

    def __add__(self, other):
        return _GenericArithmetic(self, '+', other)

    def __radd__(self, other):
        return _GenericArithmetic(other, '+', self)

    def __sub__(self, other):
        return _GenericArithmetic(self, '-', other)

    def __rsub__(self, other):
        return _GenericArithmetic(other, '-', self)

    def __mul__(self, other):
        return _GenericArithmetic(self, '*', other)

    def __rmul__(self, other):
        return _GenericArithmetic(other, '*', self)

    def __truediv__(self, other):
        return _GenericArithmetic(self, '/', other)

    def __rtruediv__(self, other):
        return _GenericArithmetic(other, '/', self)


class _GenericValueOf:
    """
    Result of value_param(target_param) (e.g. OUT(Generics.G)) when
    target_param is itself still an unresolved generic parameter, not yet a
    concrete Node type/instance - see GenericsMeta.__call__. A deferred
    lookup, not a value yet, like _GenericComparison/_GenericArithmetic:
    only meaningful as an __aliases__ value (see Node.__aliases__),
    resolved by _specialize_type once target_param's own binding becomes a
    concrete Node type - by asking THAT type for its own value_param
    generic (e.g. once G resolves to some Map[3, 2] subtype, OUT(G)
    resolves to that type's own OUT, i.e. 2). Still open (returned as-is),
    same as any other still-unresolved alias value, if target_param isn't
    bound to a concrete type yet.
    """
    def __init__(self, value_param, target_param):
        self.value_param = value_param
        self.target_param = target_param

    def __repr__(self):
        return f'{self.value_param.__name__}({self.target_param.__name__})'

    def __bool__(self):
        raise TypeError(f'{self!r} is a deferred generic-value lookup (only resolved once '
                         f'{self.target_param.__name__} is bound to a concrete type) - not a value yet.')


class _GenericDeclaration:
    """
    Result of Generic[...] (see Generic below). Carries the declared params
    but contributes nothing to the actual MRO (__mro_entries__ -> ()) - a
    class statement using Generic[...] as one of its bases still ends up with
    only its "real" bases in bases/__mro__; _NodeMeta.__new__ instead finds
    this object directly in __orig_bases__ (set automatically by Python
    itself - see PEP 560 - whenever a base defines __mro_entries__) to learn
    which new generic names cls declares.
    """
    def __init__(self, params: tuple):
        self.params = params

    def __mro_entries__(self, bases):
        return ()


class Generic:
    """
    Declares the generic parameters a Node subclass introduces, independent
    of its actual base(s):

        class A(Struct, Generic[T]): ...

    declares T as a new (open) generic parameter of A - regardless of
    whether Struct is itself a root, already generic, or shared with other
    bases. This replaces the older, more confusing `class A(Struct[T]): ...`
    spelling, which conflated "declare a new generic here" with "which base
    happens to be this family's root" (only a root could declare that way).
    An unresolved parameter declared this way is still a generic parameter of
    every subclass that doesn't itself bind it, e.g.:

        class Map(Component, Generic[IN, OUT]): ...
        class Constant(Map): ...  # Constant is still generic in IN and OUT

    Cls[T: value] (binding a generic that's already declared, either here or
    inherited) is unaffected - Generic[...] is only ever about declaring new
    names, never about binding them.
    """
    def __class_getitem__(cls, params):
        if not isinstance(params, tuple):
            params = (params,)
        for p in params:
            if not _is_param(p):
                raise TypeError(f'Generic[...] arguments must be generic parameters (e.g. Generics.T), got {p!r}')
        return _GenericDeclaration(params)


def _binding(arg):
    """
    Turns one Root[...] subscript argument into a (name, value) pair. A bare
    `T` means unbound (None). `N: 4` is parsed by Python as slice(N, 4).
    """
    if _is_param(arg):
        return arg.__name__, None
    if isinstance(arg, slice):
        assert _is_param(arg.start), f'Expected a generic parameter, got {arg.start!r}'
        assert arg.step is None, f'Invalid generics binding {arg!r}'
        return arg.start.__name__, arg.stop
    raise TypeError(f'Invalid generics argument {arg!r}, expected T or T: value')


def _merge_binding(generics: dict, name: str, value):
    """
    A generic can be declared, then set from None to a value, or re-set to the
    same value / None (both idempotent) - but never rebound to a different,
    already-fixed value. A generic still aliased to another (unresolved) param
    is treated the same as None - still open, freely replaceable - since a
    param is never itself a legitimate final/concrete value.
    """
    if name not in generics or _is_unresolved(generics[name]):
        generics[name] = value
        return
    current = generics[name]
    if value is not None and value != current:
        raise TypeError(f'Generic {name!r} is already bound to {current!r}, can not rebind to {value!r}')


def _full_resolve_map(generics: dict):
    """
    generics (cls's own final, fully-merged __generics__ - inherited,
    declared via Generic[...], and bound, all already folded in by the
    caller), chased through alias chains (a value that is itself a param,
    e.g. T bound to H) down to its terminal value (a concrete value, or None
    if genuinely still open). This is what field/annotation resolution
    should use - unlike the caller's own __generics__ isn't pruned to drop
    e.g. names inherited-but-shadowed, so it's guaranteed to have every name
    a field could reference.
    """
    raw = dict(generics)

    def resolve(v, seen=()):
        if _is_param(v) and v.__name__ in raw and v.__name__ not in seen:
            return resolve(raw[v.__name__], seen + (v.__name__,))
        return v

    return {k: resolve(v) for k, v in raw.items()}


def _specialize_type(t, resolve_map):
    """
    Returns t (a param, a _GenericComparison, or a Node type referencing a
    param through its generics) with any param it references substituted
    via resolve_map - a _GenericComparison resolves to 1 or 0 once its own
    param does (see _GenericComparison), a Node type re-specializes with its
    resolved generics (see _specialize). Returns t itself, unchanged, if
    nothing in it is resolvable yet.
    """
    if _is_param(t):
        r = resolve_map.get(t.__name__)
        return r if r is not None and not _is_param(r) else t

    if isinstance(t, _GenericComparison):
        resolved = _specialize_type(t.param, resolve_map)
        if _is_param(resolved):
            return t  # still open - nothing to compare yet
        is_equal = resolved == t.value
        return int(is_equal if t.op == '==' else not is_equal)

    if isinstance(t, _GenericArithmetic):
        left = _specialize_type(t.left, resolve_map)
        right = _specialize_type(t.right, resolve_map)
        if _is_param(left) or isinstance(left, _GenericArithmetic) or _is_param(right) or isinstance(right, _GenericArithmetic):
            return t if left is t.left and right is t.right else _GenericArithmetic(left, t.op, right)
        return _GenericArithmetic._OPS[t.op](left, right)

    if isinstance(t, _GenericValueOf):
        resolved_target = _specialize_type(t.target_param, resolve_map)
        if not isinstance(resolved_target, _NodeMeta):
            return t  # target_param still open - nothing to look up yet
        return resolved_target.generic_value(t.value_param)

    if isinstance(t, _NodeMeta):
        new_generics = {k: _specialize_type(v, resolve_map) for k, v in t.__generics__.items()}
        # NOT new_generics == t.__generics__: _specialize_type always either
        # returns a value completely unchanged (the exact same object) or a
        # newly resolved one, never an equal-but-different-object copy, so
        # identity here is both sufficient and (unlike ==, which would
        # otherwise dispatch to GenericsMeta.__eq__ any time a value is
        # itself still an unresolved param) safe from _GenericComparison.
        if all(new_generics[k] is v for k, v in t.__generics__.items()):
            return t
        # Re-derive from t's true (non-specialization) root and go through the
        # same cached _specialize(...) path as Cls[...], instead of building a
        # one-off type by hand - so e.g. resolving B[H=T]'s T to int yields the
        # exact same B[int] object as writing B[int] directly, not a new,
        # uncached, still-named-"B[H=T]" type every time this gets called.
        root = t
        while root.__is_specialization__:
            root = root.__bases__[0]
        return _specialize(root, new_generics)

    return t


class Single(RDVType):
    """
    Canonical internal representation of a primitive scalar/vector/matrix
    field, e.g. Single[Type.FLOAT32]. Annotations still accept the friendly
    `float`/`int`/`vec3`/... spelling; _normalize_field converts them to this
    uniformly, one (cached, so comparable by identity) instance per Type value.
    """
    _cache = {}

    def __class_getitem__(cls, t):
        if t not in cls._cache:
            cls._cache[t] = type('Single({})'.format(t), (Single,), dict(__type__=t))
        return cls._cache[t]

    def __init__(self, value):
        assert hasattr(type(self), '__type__'), "Can not instantiate an untyped Sinlge object. Use Single[type] instead"
        self.__value = value

    def __repr__(self):
        return f'Single[{self.__type__}]({self.__value})'

    @classmethod
    def _fields_layout_decription(cls):
        return cls.__type__

    def _fields_value(self):
        return self.__value


def _build_primitives() -> dict:
    """Maps the friendly python-level spellings (int, float, _vk.vec3, ...) to their Type enum value."""
    names = {
        'int': Type.INT32, 'float': Type.FLOAT32, 'bool': Type.BOOL,
        'vec2': Type.VEC2, 'vec3': Type.VEC3, 'vec4': Type.VEC4,
        'ivec2': Type.IVEC2, 'ivec3': Type.IVEC3, 'ivec4': Type.IVEC4,
        'uvec2': Type.UVEC2, 'uvec3': Type.UVEC3, 'uvec4': Type.UVEC4,
        'mat2': Type.MAT2, 'mat3': Type.MAT3, 'mat4': Type.MAT4,
        'mat2x3': Type.MAT2x3, 'mat2x4': Type.MAT2x4, 'mat3x2': Type.MAT3x2,
        'mat3x4': Type.MAT3x4, 'mat4x2': Type.MAT4x2, 'mat4x3': Type.MAT4x3,
    }
    builtins = {'int': int, 'float': float, 'bool': bool}
    primitives = {}
    for py_name, type_enum in names.items():
        py_type = builtins.get(py_name) or getattr(_vk, py_name, None)
        if py_type is not None:
            primitives[py_type] = type_enum
    return primitives


_PRIMITIVES = _build_primitives()


class Array:
    """Array[Size[count], Type]: declares a fixed/generic-sized array field."""
    def __class_getitem__(cls, item):
        size, element = item
        return _ArrayType(size, element)


class _ArrayType:
    def __init__(self, size, element):
        self.size = size
        self.element = element

    def __eq__(self, other):
        return isinstance(other, _ArrayType) and self.size is other.size and self.element == other.element

    def __hash__(self):
        return hash((self.size, self.element))

    def __repr__(self):
        return f'Array[{self.size!r}, {self.element!r}]'


def _merge_field_type(field_name, existing, incoming):
    """
    Two fields collected under the same name are accepted only if their
    (already generics-resolved) types match - except structs (dicts), whose
    fields are instead combined (union), field by field, recursively.
    """
    if isinstance(existing, dict) and isinstance(incoming, dict):
        merged = dict(existing)
        for k, v in incoming.items():
            merged[k] = _merge_field_type(k, merged[k], v) if k in merged else v
        return merged
    if existing == incoming:
        return existing
    raise TypeError(f'Conflicting type for field {field_name!r}: {existing!r} vs {incoming!r}')


class _InnerStruct:
    """
    Marks a class actually nested inside a Node's own body (`class light:
    ...`) as an inner-struct field. A bare annotation referencing some
    unrelated, already-existing class (`x: dict`) is not wrapped, and so is
    rejected by _normalize_field - only a genuinely nested definition counts
    as "class (to define an inner struct)".
    """
    __slots__ = ('klass',)

    def __init__(self, klass):
        self.klass = klass


def _own_fields_of(klass) -> dict:
    """
    The raw (name -> type) fields declared directly on klass: via bare
    annotations, an explicit __fields__ dict, or a nested plain class (treated
    as a struct-typed field). Declaring both annotations and __fields__
    together is an error. Reads klass.__annotations__ (own only, never
    inherited) rather than a pre-creation namespace dict, since annotations
    are evaluated lazily (PEP 649) and are not yet resolvable before the class
    exists. Dunder names (__foo__) are never fields, even if annotated (e.g.
    Node's own __accessor__ : 'Accessor') - only single/double-leading-
    underscore-or-plain names are.
    """
    annotations = getattr(klass, '__annotations__', None) or {}
    own_fields = klass.__dict__.get('__fields__')
    if annotations and own_fields:
        raise TypeError('fields declared via both annotations and __fields__')
    raw = dict(own_fields) if own_fields is not None else dict(annotations)
    raw = {k: v for k, v in raw.items() if not k.startswith('__')}
    for k, v in vars(klass).items():
        if k.startswith('__') or k in raw:
            continue
        if isinstance(v, type) and not _is_param(v) and not isinstance(v, _NodeMeta):
            raw[k] = _InnerStruct(v)  # a class nested here: a struct-typed field
    return raw


def _normalize_field(t, resolve_map):
    """
    Canonicalizes one field type, restricted to: int/float/a _vk vector or
    matrix type (-> Single[Type.X]), a Node type (kept as-is, generics
    specialized), any other RDVType (e.g. Tensor - resolved via its own
    _fields_layout_decription()), a class nested inside the declaring Node
    (-> its own collected struct), Array[Size[n], T] (-> [n, normalize(T)]),
    Single[...] or a generic parameter (substituted via resolve_map, once
    known). Anything else is rejected.
    """
    if isinstance(t, list):
        return [t[0], _normalize_field(t[1], resolve_map)]
    if isinstance(t, dict):
        return {k: _normalize_field(v, resolve_map) for k, v in t.items()}
    if isinstance(t, _InnerStruct):
        return _collect_fields(t.klass, resolve_map)
    if _is_param(t):
        if t.__name__ not in resolve_map:
            raise TypeError(f'Field uses generic {t.__name__!r}, which is not declared by this type '
                             f'(e.g. class Struct(..., Generic[{t.__name__}]): ...).')
        r = resolve_map[t.__name__]
        # r may itself still be a bare param (not yet bound) or a deferred
        # _GenericArithmetic/_GenericComparison expression (e.g. an array
        # count of Generics.IN + 1, on a type - like Tensor - specialized
        # before the outer class' own IN is known) - not resolvable yet
        # either way, so leave t as-is rather than recursing into it as if
        # it were itself a field type (see _specialize_type, which is what
        # actually resolves _GenericArithmetic once the real value is
        # known).
        if r is None or _is_param(r) or isinstance(r, (_GenericArithmetic, _GenericComparison)):
            return t
        return _normalize_field(r, resolve_map)
    if isinstance(t, _ArrayType):
        size = _normalize_field(t.size, resolve_map)
        return [size, _normalize_field(t.element, resolve_map)]
    if isinstance(t, int):
        # only reachable for an Array[...]'s size (recursed into just above
        # once a generic parameter resolves to a concrete count) - a plain
        # int is already a valid, terminal count, not itself a "field type"
        # to canonicalize any further.
        return t
    if isinstance(t, Type):
        # a generic bound directly to a raw Type value (e.g. T=Type.FLOAT32,
        # as opposed to the friendly float/int/vec3 spelling _PRIMITIVES
        # below maps, or an already-canonical Single[Type.FLOAT32]) - same
        # canonical wrapping, just a different raw spelling to normalize.
        return Single[t]
    if t in _PRIMITIVES:
        return Single[_PRIMITIVES[t]]
    if isinstance(t, type):
        if issubclass(t, Single):
            return t
        if isinstance(t, _NodeMeta):
            return _specialize_type(t, resolve_map)
        if issubclass(t, RDVType):
            return t._fields_layout_decription()
    raise TypeError(f'Invalid field type {t!r}: expected Tensor, int, float, a vk vector/matrix type, '
                     f'a Node type, a class nested here (inner struct), Array[...], Single[...], or a generic parameter.')


def _collect_fields(cls, resolve_map):
    """
    A class's own fields, plus every base's (inherited, own or nested-struct)
    fields, all resolved via resolve_map. Same-name fields are accepted if
    their resolved type matches, or (for structs) combined field-by-field.
    """
    fields = {}
    for klass in reversed(cls.__mro__):
        # a _NodeMeta class already has its own declaration captured (at
        # __new__ time, before __fields__ gets overwritten with the collected
        # result below); anything else (e.g. a plain nested struct class,
        # never processed by _NodeMeta) is scanned fresh, own attributes only.
        declared = klass.__dict__.get('__declared_fields__')
        if declared is None:
            declared = _own_fields_of(klass)
        for k, v in declared.items():
            nv = _normalize_field(v, resolve_map)
            fields[k] = _merge_field_type(k, fields[k], nv) if k in fields else nv
    return fields


def _is_root_declaration(base, root) -> bool:
    """True if base is root itself (bare), or a direct root[...] specialization
    of it (its only base is root) - i.e. an actual root[...] declaration
    point, as opposed to a further-removed specialization of some other class
    (A[H]), or a plain `class A(root): pass` (a normal subclass, not a
    binding - so root can still be combined with other root-derived classes,
    e.g. `class C(A, B): pass`)."""
    return base is root or (
        isinstance(base, _NodeMeta) and base.__is_specialization__ and base.__bases__ == (root,)
    )


def _node_bases(bases):
    """The subset of bases that are actually part of the Node/_NodeMeta family
    (plain, non-Node mixins are ignored for root/generics purposes; Python's
    normal MRO still brings their methods in)."""
    return [b for b in bases if isinstance(b, _NodeMeta)]


def _validate_bases_and_root(name, bases):
    """
    Directly inheriting bare Node (possibly alongside plain, non-Node mixins)
    creates a new root node: returns None
    here, the caller sets cls.__root__ = cls itself afterwards. Otherwise
    every Node-family base must already belong to some root (Node itself does
    not - Node can not be subscripted/used directly other than bare), all
    must belong to the *same* root, and at most one of them may be a root
    declaration (root itself, or root[...]) - which, if present, must be the
    first Node-family base.
    """
    node_bases = _node_bases(bases)
    if not node_bases or node_bases == [Node]:
        return None

    roots = []
    for b in node_bases:
        r = getattr(b, '__root__', None)
        if r is None:
            raise TypeError(f'{name}: {b.__name__} does not belong to a root node '
                             f'(only Node itself can be inherited bare to create one)')
        roots.append(r)
    if len(set(roots)) > 1:
        raise TypeError(f"{name}: bases belong to different root nodes "
                         f"({', '.join(sorted({r.__name__ for r in roots}))})")

    root = roots[0]
    if len(node_bases) > 1:
        declaring = [i for i, b in enumerate(node_bases) if _is_root_declaration(b, root)]
        if len(declaring) > 1:
            raise TypeError(f'{name}: only one root declaration ({root.__name__}[...]) allowed among bases')
        if declaring and declaring[0] != 0:
            raise TypeError(f'{name}: {root.__name__} must be the first base')
    return root


def _merge_bases_generics(bases):
    """
    Every (Node-family) base's own generics (fixed or not) are inherited -
    merged across all of them (same root family, validated by
    _validate_bases_and_root; plain non-Node mixins contribute no generics
    of their own, only fields/behavior - see _node_bases), via
    _merge_binding: two bases naming the *same* generic must agree (an
    already-fixed value can't be silently overridden by a conflicting one -
    still-open/None from one side and fixed from another is fine, the fixed
    value wins), but two bases naming *different* generics (e.g. several
    independently-generic traits, each declaring their own) both end up
    exposed - a class combining them is generic in all of them at once,
    with no need to redeclare any via its own Generic[...].
    """
    generics = {}
    for base in _node_bases(bases):
        for k, v in base.__generics__.items():
            _merge_binding(generics, k, v)
    return generics


def _declared_generics_of(dct, bases):
    """
    New generic parameter names this class declares via Generic[...] among
    its bases (see Generic/_GenericDeclaration) - e.g. class A(Struct,
    Generic[T]): ... declares T here. Python's own class-creation machinery
    (PEP 560) already stashes the ORIGINAL, unresolved bases tuple (the one
    that still contains the Generic[...] object, before __mro_entries__
    dropped it) as dct['__orig_bases__'] whenever any base defines
    __mro_entries__; bases itself (already resolved, what _NodeMeta.__new__
    otherwise works with) never contains it. Declaring the same name twice
    (two Generic[...] bases, or duplicate params in one) is redundant but
    harmless (a dict, later one wins - both are always None anyway); it's
    _NodeMeta.__new__'s job to reject a name that collides with one already
    inherited from a base.
    """
    declared = {}
    for base in dct.get('__orig_bases__', bases):
        if isinstance(base, _GenericDeclaration):
            for p in base.params:
                declared[p.__name__] = None
    return declared


def _own_aliases_of(dct):
    """
    Normalizes this class' own raw __aliases__ declaration (e.g.
    {Generics.T: Pointer[vec3]}, keyed by the generic parameter itself) into
    {name: Type}, keyed by name like __generics__. Unlike a plain generic
    parameter, an alias has no "unbound" state: it must already name a
    concrete type at declaration time (never None, never another param left
    to resolve later) - that fixed value is exactly what lets it stand in as
    a real GLSL type name in this node's own code, but also why it can never
    be re-bound/cast the way a genuine open generic can.
    """
    raw = dct.get('__aliases__')
    if not raw:
        return {}
    own = {}
    for k, v in raw.items():
        if not _is_param(k):
            raise TypeError(f'__aliases__ keys must be a generic parameter (e.g. Generics.T), got {k!r}')
        if v is None or _is_param(v):
            raise TypeError(f'Alias {k.__name__!r} must be bound to a concrete type, got {v!r}')
        own[k.__name__] = v
    return own


def _merge_bases_aliases(name, bases, own_aliases):
    """
    __aliases__ accumulates from every Node-family base (unlike __generics__,
    which only exposes the first base's generics) - an alias is a plain,
    fixed name -> concrete type fact, not specialization state, so there is
    no "which base owns it" ambiguity; two bases agreeing on the same name is
    fine, but two bases (or a base and this class' own declaration)
    disagreeing about the same name is an error: once a name is an alias for
    a type, no descendant may redefine it, since other code (in this class
    or another subclass) may already rely on it meaning that original type.
    """
    aliases = {}
    for base in _node_bases(bases):
        for k, v in getattr(base, '__aliases__', {}).items():
            if k in aliases and aliases[k] != v:
                raise TypeError(f"{name}: bases disagree on alias {k!r} ({aliases[k]!r} vs {v!r})")
            aliases[k] = v
    for k, v in own_aliases.items():
        if k in aliases:
            raise TypeError(f"{name}: alias {k!r} is already defined (as {aliases[k]!r}) by a base; "
                             f"aliases can not be redefined")
        aliases[k] = v
    return aliases


def _node_types_in(value):
    """Yields every Node type referenced anywhere inside a (possibly nested, already-collected) field value."""
    if isinstance(value, dict):
        for v in value.values():
            yield from _node_types_in(v)
    elif isinstance(value, list):
        yield from _node_types_in(value[1])
    elif isinstance(value, _NodeMeta):
        yield value


def _check_concrete_field_types(name, fields):
    """
    Once a type is itself fully concrete (no more generics left to resolve),
    every Node type referenced by one of its fields must be concrete too - not
    still generic, and not __abstract__.
    """
    for field_type in _node_types_in(fields):
        if field_type.__is_generic__ or field_type.__abstract__:
            why = 'generic' if field_type.__is_generic__ else 'abstract'
            raise TypeError(f'{name}: field type {field_type.__name__} is {why}; '
                             f'fields must reference concrete, non-abstract Node types')


def _collect_requires(name, bases, declared_requires, generics, fields, aliases=None):
    """
    __requires__ lists Node types that must "appear before" this one (e.g. for
    code generation order). Two kinds of source, handled differently:
    - explicit __requires__ declarations are stable (__declared_requires__
      itself, kept raw/unresolved so it's still correctly resolvable by an
      even-more-derived subclass later), so they simply accumulate: this
      class' own + every base's own declared __requires__ (not their final
      one) - but, when building the final, transitive __requires__ below,
      each one is resolved against *this* class' own (already fully
      resolved) generics first - e.g. a declaration like `__requires__ =
      [SomeLibrary[N]]` referencing this same type's own N generic
      parameter must mean SomeLibrary[3] once N is bound to 3, not stay
      pinned to the still-open parameter forever.
    - Node types bound as a generic parameter value, referenced (directly or
      nested) by a field, or used as an alias value (e.g. Deferred's own
      IB/OB-style aliases - the alias's #define is otherwise the only trace
      of it, its own code/struct declaration would never get emitted
      without this), are re-derived fresh from *this* class' own (already
      fully resolved) generics/fields/aliases every time - never inherited
      as a blob from a base's final __requires__, since a still-generic
      base's own requires can reference now-superseded (unresolved) types
      that a further specialization has since replaced.
    Either way, each requirement is added together with *its own* (already
    transitive) __requires__, so the result stays a complete ordering, not
    just direct edges. If that union ends up containing one of this class'
    own ancestors, that is a cyclic dependency and raises immediately.
    """
    own_declared = set(declared_requires)
    for base in bases:
        own_declared |= set(getattr(base, '__declared_requires__', ()))

    resolve_map = _full_resolve_map(generics)
    resolved_declared = [_specialize_type(req, resolve_map) for req in own_declared]

    candidates = (resolved_declared
                  + [v for v in generics.values() if isinstance(v, _NodeMeta)]
                  + list(_node_types_in(fields))
                  + [v for v in (aliases or {}).values() if isinstance(v, _NodeMeta)])
    requires = set()
    for req in candidates:
        if not isinstance(req, _NodeMeta):
            raise TypeError(f'{name}: __requires__ entries must be Node types, got {req!r}')
        requires.add(req)
        requires |= set(req.__requires__)

    for base in bases:
        for ancestor in base.__mro__:
            if ancestor in requires:
                raise TypeError(f"{name}: cyclic dependency - {ancestor.__name__} is required "
                                 f"(directly or transitively) but is also one of this type's own ancestors")
    return frozenset(own_declared), frozenset(requires)


_codename_ids = {}

_signature_ids = _itertools.count()
"""Global counter handing out each concrete (non-generic, non-abstract) type's own unique __signature__."""


def _codename_for(cls) -> str:
    """
    A unique, valid GLSL identifier for cls. Two statically equivalent types
    are already the exact same (cached, see _specialize) type object, so
    assigning one id per distinct type object here automatically gives them
    the same codename too.

    The readable part stays deliberately short: a class-valued generic
    (e.g. a BinaryOperation's own M1/M2) is embedded by its own
    already-computed __codename__, never its full __name__ - by the time
    cls itself is being named, every class-valued generic it carries was
    already built (and so already has its own short __codename__), so
    this never needs to recurse into *that* type's own signature. Embedding
    the full __name__ instead (the previous behavior) recurses through
    however many levels of nesting that type itself has, compounding with
    every further level of composition - harmless for a shallow type
    (Cameras[IN=3, OUT=6]) but, for a deeply nested one (radiance-style
    Sensor(Sensor(..., ConcatMap(...)), Add(Select, Mul(Promote(...),
    Composite(...)))) chains), grows well past glslang's identifier length
    limit ('name too long', a hard compile failure).

    Only rendered at all when it stays unambiguous and bounded - every
    generic value is either a plain int/bool, or (at most one of them) a
    class, the same "single type parameter" shape as e.g. C++'s
    Pointer<T>. Anything wider (multiple class-valued generics - e.g. a
    BinaryOperation's own M1 *and* M2 both being types) falls back to a
    bare Type_<id>: rendering *that* readably would need to expose
    every operand's own signature to stay unambiguous, right back to the
    same blowup this exists to avoid.
    """
    if cls not in _codename_ids:
        _codename_ids[cls] = len(_codename_ids)
    cls_id = _codename_ids[cls]
    root_name = _re.sub(r'\W', '', (getattr(cls, '__root__', None) or cls).__name__)
    generics = {k: v for k, v in (getattr(cls, '__generics__', None) or {}).items() if v is not None}
    class_valued = [v for v in generics.values() if isinstance(v, _NodeMeta)]
    parts = []
    if len(class_valued) <= 1:
        for v in generics.values():
            if isinstance(v, _NodeMeta):
                parts.append(v.__codename__)
            elif isinstance(v, (int, bool)):
                parts.append(str(v))
            else:
                parts = None
                break
    else:
        parts = None
    if parts:
        return f'{root_name}_{"_".join(parts)}_{cls_id:x}'
    return f'{root_name}_{cls_id:x}'


def _build_glsl_type_names() -> dict:
    pairs = [
        ('BOOL', 'bool'), ('INT8', 'int8_t'), ('UINT8', 'uint8_t'), ('UINT16', 'uint16_t'),
        ('INT16', 'int16_t'), ('FLOAT16', 'float16_t'), ('UINT32', 'uint'), ('UINT64', 'uint64_t'),
        ('INT32', 'int'), ('INT64', 'int64_t'), ('FLOAT32', 'float'), ('FLOAT64', 'double'),
        ('UVEC4', 'uvec4'), ('UVEC3', 'uvec3'), ('UVEC2', 'uvec2'),
        ('VEC2', 'vec2'), ('VEC3', 'vec3'), ('VEC4', 'vec4'),
        ('IVEC2', 'ivec2'), ('IVEC3', 'ivec3'), ('IVEC4', 'ivec4'),
        ('MAT2', 'mat2'), ('MAT3', 'mat3'), ('MAT4', 'mat4'),
        ('MAT2x3', 'mat2x3'), ('MAT2x4', 'mat2x4'), ('MAT3x2', 'mat3x2'),
        ('MAT3x4', 'mat3x4'), ('MAT4x2', 'mat4x2'), ('MAT4x3', 'mat4x3'),
    ]
    names = {}
    for attr, glsl in pairs:
        enum_val = getattr(Type, attr, None)
        if enum_val is not None:
            names[enum_val] = glsl
    return names


_GLSL_TYPE_NAMES = _build_glsl_type_names()


def _glsl_type_of(field_type) -> str:
    """The GLSL type spelling for one (already-collected) field type: Single -> its
    Type's GLSL name, Tensor -> uint64_t, a Node type -> its own codename, a
    struct (dict) -> an anonymous inline struct of its own fields."""
    if isinstance(field_type, type) and issubclass(field_type, Single):
        return _GLSL_TYPE_NAMES[field_type.__type__]
    if isinstance(field_type, type) and issubclass(field_type, RDVType):
        return field_type.__codename__
    if isinstance(field_type, dict):
        body = ''.join(f'    {_glsl_field_decl(k, v)}\n' for k, v in field_type.items())
        return f'struct {{\n{body}}}'
    raise TypeError(f'Can not translate field type {field_type!r} to GLSL.')


def _glsl_field_decl(field_name, field_type) -> str:
    """One GLSL field declaration line (array-ness, [count], goes after the name, not the type)."""
    if isinstance(field_type, list):
        count, element = field_type
        return f'{_glsl_type_of(element)} {field_name}[{count}];'
    return f'{_glsl_type_of(field_type)} {field_name};'


_NodeT = _typing.TypeVar('_NodeT', bound='Node')


_specialization_cache = {}


def _specialize(cls, bindings: dict):
    """
    Builds (or returns, from cache) the unique specialization of cls for the
    given generic bindings merged on top of cls's own generics - the exact
    same (cls, resulting generics) always produces the exact same type object,
    whether reached through Cls[...], constructor inference, or cast().
    """
    generics = dict(cls.__generics__)
    for k, v in bindings.items():
        _merge_binding(generics, k, v)
    # Always keyed/based off cls's true (non-specialization) root, not cls
    # itself - otherwise specializing an already-specialized cls (e.g. cls is
    # itself Constant[OUT=3], reached via a partial constructor inference,
    # then further bound by cast()) would build a new type nested on top of
    # it (named "Constant[OUT=3][IN=0, OUT=3]") instead of collapsing to the
    # exact same Constant[IN=0, OUT=3] a single, direct specialization would
    # produce (see _specialize_type, which already relies on this same
    # collapsing behavior).
    root = cls
    while root.__is_specialization__:
        root = root.__bases__[0]
    key = (root, frozenset(generics.items()))
    if key not in _specialization_cache:
        # Named/bound in root.__generics__'s own (declared) order, not
        # bindings' (bindings only holds this call's delta, in whatever order
        # it happened to be bound in - e.g. constructor-inferred generics are
        # ordered by whichever self.match() calls __init__ happens to make
        # first, which need not match declaration order at all). The full
        # merged generics dict (not just this call's delta) is passed as
        # __bind__ since it's applied on top of root's own (fully unbound)
        # generics, not cls's (possibly already partially bound) ones.
        specialization = ', '.join(f'{k}={v}' for k, v in generics.items())
        _specialization_cache[key] = _NodeMeta(f'{root.__name__}[{specialization}]', (root,), {'__bind__': generics})
    return _specialization_cache[key]


class _NodeMeta(type):
    """
    Metaclass that accumulates __generics__/__fields__ along the inheritance
    chain and turns Cls[...] into a further specialization (binding) of Cls's
    generics.
    """

    def __new__(mcls, name, bases, dct):
        root = _validate_bases_and_root(name, bases)
        is_specialization = '__bind__' in dct  # true only for a Cls[...] result, not `class X(Cls): ...`
        bindings = dct.pop('__bind__', {})
        generics = _merge_bases_generics(bases)
        for new_name in _declared_generics_of(dct, bases):
            if new_name in generics:
                raise TypeError(f"{name}: Generic[{new_name}] redeclares {new_name!r}, already a generic "
                                 f"parameter inherited from a base")
            generics[new_name] = None
        for k, v in bindings.items():
            _merge_binding(generics, k, v)
        own_aliases = _own_aliases_of(dct)
        aliases = _merge_bases_aliases(name, bases, own_aliases)
        # An alias declared in terms of one of this type's own generics (e.g.
        # Tensor's D: Deferred[Buffer[T]]) is inherited fixed/unresolved from
        # its declaring class, but must resolve fresh per specialization -
        # Tensor[T=float].D and Tensor[T=int].D are different concrete types,
        # even though neither ever "redefines" D (that's still checked, on
        # the raw/unresolved value, inside _merge_bases_aliases above - two
        # specializations naturally resolving differently isn't a conflict).
        if aliases:
            resolve_map = _full_resolve_map(generics)
            aliases = {k: _specialize_type(v, resolve_map) for k, v in aliases.items()}
        if overlap := (generics.keys() & aliases.keys()):
            raise TypeError(f"{name}: {sorted(overlap)!r} declared as both a generic parameter and an alias")
        cls = super().__new__(mcls, name, bases, dct)
        try:
            is_bare_node = _node_bases(bases) == [Node]  # NameError while Node itself is being created
        except NameError:
            is_bare_node = False
        cls.__root__ = cls if is_bare_node else root
        cls.__is_specialization__ = is_specialization
        cls.__generics__ = generics
        cls.__is_generic__ = any(_is_unresolved(v) for v in generics.values())
        if not cls.__is_generic__:
            # A plain function/lambda alias value (e.g. Generics.IS_BOO:
            # lambda cls: hasattr(cls, "is_boo") and cls.is_boo) is an
            # escape hatch for whatever _GenericComparison/_GenericArithmetic/
            # _GenericValueOf can't express legibly - resolved here, called
            # with cls itself, once cls is a genuine concrete (non-generic)
            # type (so the lambda can inspect its finished attributes/
            # methods freely); still generic classes leave it unevaluated
            # (inherited as-is, same as any other still-open alias) until
            # some later specialization is concrete. `not isinstance(v, type)`
            # excludes an alias already resolved to a Node type or a still-
            # open GenericParam (both callable, but never meant to be
            # invoked here).
            aliases = {k: (v(cls) if callable(v) and not isinstance(v, type) else v) for k, v in aliases.items()}
        cls.__aliases__ = aliases
        try:
            cls.__declared_fields__ = _own_fields_of(cls)  # needs cls to exist (PEP 649 annotations)
        except TypeError as e:
            raise TypeError(f'{name}: {e}') from None
        cls.__fields__ = _collect_fields(cls, _full_resolve_map(generics))
        if not cls.__is_generic__:
            _check_concrete_field_types(name, cls.__fields__)
        cls.__declared_requires__, cls.__requires__ = _collect_requires(
            name, bases, dct.get('__requires__', None) or [], generics, cls.__fields__, aliases)
        try:
            cls._custom_definitions(bases, dct)
        except NotImplementedError:
            pass  # this root family (e.g. bare Node itself) has nothing to collect
        if '__codename__' not in cls.__dict__:
            # own-dict check, not hasattr: hasattr also finds an *inherited*
            # __codename__ (every class has one, inherited transitively from
            # Node if nothing else), which would wrongly skip assigning this
            # class its own and leave it silently sharing some ancestor's.
            cls.__codename__ = _codename_for(cls)
        # __abstract__ is never inherited: only a class that sets it itself is abstract,
        # a subclass of an abstract type is concrete again unless it says so itself too.
        cls.__abstract__ = bool(dct.get('__abstract__', False))
        cls.__signature__ = next(_signature_ids) if not cls.__is_generic__ and not cls.__abstract__ else None
        return cls

    def __getitem__(cls: type[_NodeT], args) -> type[_NodeT]:
        # returning cls's own type (not _NodeMeta), so PyCharm/mypy infer
        # Foo[...] as (a subtype of) Foo itself, not some unrelated metaclass
        # instance - this is also what makes Foo[...]() a recognized,
        # callable constructor of Foo.
        if not isinstance(args, tuple):
            args = (args,)
        # Bare values (e.g. Foo[int], but also a bare param used as an alias,
        # e.g. Foo[H]) bind positionally, in declaration order, to whichever
        # generics are still unbound. Cls[...] only ever *binds* generics cls
        # already has (declared by cls itself via Generic[...], or inherited)
        # - it never declares new ones; see Generic for that.
        unbound = iter(k for k, v in cls.__generics__.items() if v is None)
        bindings = {}
        for a in args:
            if isinstance(a, slice):
                name, value = _binding(a)
                if name not in cls.__generics__:
                    raise TypeError(f"{cls.__name__} has no generic parameter {name!r}; "
                                     f"declare it via class X({cls.__name__}, Generic[{name}]): ...")
            elif _is_param(a):
                try:
                    name, value = next(unbound), a
                except StopIteration:
                    raise TypeError(f"{cls.__name__} has no unbound generic parameter left for {a.__name__!r}; "
                                     f"declare it via class X({cls.__name__}, Generic[{a.__name__}]): ...")
            else:
                try:
                    name = next(unbound)
                except StopIteration:
                    raise TypeError(f'Too many generic arguments for {cls.__name__}: {args!r}')
                value = a
            bindings[name] = value
        return _specialize(cls, bindings)

    def __call__(cls: type[_NodeT], *args, **kwargs) -> _NodeT:
        # _device_index/__accessor__ are set up before __init__ runs (unlike
        # plain type.__call__, which would only do so after) so that field
        # assignment during __init__ (e.g. self.x = x in a Struct/Component/
        # Main subclass) has a real, properly buffer-backed accessor to land
        # in from its very first statement - not a second, separate one
        # that then gets silently replaced/discarded once __init__ returns.
        instance = cls.__new__(cls)
        instance._device_index = _vk.device_index()
        instance.__accessor__ = _build_accessor(cls, instance._device_index)
        # remembered so cast() can later rebuild an equivalent instance of a
        # further-specialized type from scratch, the same way the
        # pending-generics rebuild just below does.
        instance.__init_args__ = args
        instance.__init_kwargs__ = kwargs
        instance.__init__(*args, **kwargs)
        # self.match(...) may have inferred some of cls's still-unbound
        # generics from the constructor arguments; if it did, the unique
        # concrete (or further-specialized) type for those inferred values
        # is looked up/created and the instance is rebuilt with it from
        # scratch - so a type completed this way is the exact same type
        # object as if it had been named explicitly, e.g. type(A(5)) ==
        # type(A[int](4)).
        pending = instance.__dict__.pop('__pending_generics__', None)
        if pending:
            concrete_cls = _specialize(cls, pending)
            if concrete_cls is not cls:
                return concrete_cls(*args, **kwargs)
        return instance


class Node(metaclass=_NodeMeta):
    """
    Inherit bare (`class Struct(Node): pass`) to create a new root node: it
    gets its own identity (__root__ is itself). Node itself can only ever be
    inherited bare, never subscripted directly. Any Node subclass (root or
    not) declares new generic parameters via Generic[...] among its bases
    (`class A(Struct, Generic[T]): ...`), independent of which base it is;
    Cls[...] itself only ever binds generics a class already has.
    """
    __root__ = None
    __generics__ = {}
    __is_generic__ = False
    __aliases__ = {}
    """
    Fixed, non-generic type aliases: {Generics.T: SomeConcreteType}. Unlike a
    __generics__ entry, an alias is bound to a concrete type the moment it is
    declared and stays that way forever - it is inherited by every subclass
    but can never be reassigned by one (even to the same value again is
    rejected, see _merge_bases_aliases), since other code (this class' own,
    or another subclass') may already depend on that name meaning that exact
    type. Also unlike a generic parameter, an alias never appears in
    __generics__, so it can't be bound/rebound via Cls[...] or inferred via
    self.match() - it exists purely so generated code (__code__/__precode__)
    can refer to a fixed type by a short, generic-looking name (see
    _build_code, which #define's each alias next to each generic).
    """
    __accessor__ : 'Accessor' = None
    """
    Set on every instance (see _build_accessor) to its own Accessor: staged in
    a plain dict if this type is generic or __abstract__, otherwise backed by
    a fresh vk buffer laid out from __fields__. Access/set fields through it,
    e.g. `instance.__accessor__.some_field = value`.
    """
    __declared_requires__ = frozenset()
    __requires__ = frozenset()
    """
    Node types that must "appear before" this one (e.g. for code generation
    order): declare via __requires__ = [OtherNode, ...]. Inherited (unioned)
    from bases, and automatically extended with any Node type bound as one of
    this type's generic parameters, or referenced by a field - both cases pull
    in that Node's own (already transitive) __requires__ too, and are always
    re-derived fresh from this type's own resolved generics/fields (never
    inherited as-is from a still-generic base). A cyclic dependency raises at
    the first class where it becomes detectable.
    """

    @_classproperty
    def dependency_order(cls):
        """
        cls plus every dependency (__requires__), transitively, in postorder:
        a node is only appended once every one of its own requirements has
        already been fully visited (and thus already appended) - so each
        node is emitted only after all of its own requirements, as code
        generation and binding-slot assignment both need. Marking a node
        "seen" before recursing (rather than only once appended) still
        prevents infinite recursion on a cycle, but does NOT append it
        early - so a node shared by two requirements (e.g. cls requires
        both Deferred[Pointer[vec3]] and, once __requires__ is flattened,
        Pointer[vec3] itself) always ends up after everything that
        (transitively) needs it, regardless of which sibling's subtree
        happens to reach it first. A naive preorder-then-reverse traversal
        does NOT have this property: whichever sibling is visited first
        "claims" the shared node's position, which can leave it on the
        wrong side of something that actually depends on it.

        Cached (see _classproperty) - this traversal is otherwise repeated
        on every dispatch() call (cls.__requires__ never changes once the
        class exists), which showed up as measurable per-frame overhead in
        a tight render loop.
        """
        order = []
        seen = set()

        def visit(n):
            if n in seen:
                return
            seen.add(n)
            for r in n.__requires__:
                visit(r)
            order.append(n)

        visit(cls)
        return order

    @_classproperty
    def libraries(cls):
        """The Library subclasses in cls's own dependency_order, in that
        same order - cached the same way (see _build_code's init_codes,
        which calls each one's own RDV_NODE(global_id) init)."""
        return [n for n in cls.dependency_order if issubclass(n, Library)]

    append_computation = None
    """
    None on a plain node. A node type that needs some other Main dispatched
    first as part of any graph containing it (e.g. a barrier instance whose
    own data is only filled by a grid's own evaluator - see maths.
    TensorGridBarrier) overrides this with an instance method:

        def append_computation(self, cmd) -> None: ...

    responsible for recording its own compute work (typically via
    append_dispatch(cmd, some_main_instance, threads)) into cmd. Whether a
    type has one at all is a pure type-level fact (overridden or not,
    checked once - see has_compute_dependency/contains_compute_dependencies),
    but *what* it does is instance-level, since e.g. two different barrier
    instances of the same type wrap two different grids/tensors - see
    dispatch(), which walks main_node's own field graph (only where
    contains_compute_dependencies says it's worth it) calling this on every
    instance found with one set.
    """

    @classmethod
    def has_compute_dependency(cls) -> bool:
        """
        True iff cls itself overrides append_computation (not any other
        node's, even one it (transitively) requires - see
        contains_compute_dependencies for that). A single-type check - the
        fast primitive contains_compute_dependencies/graph exploration is
        built from, letting a big dependency tree be scanned by testing each
        node type directly instead of walking actual instances.
        """
        return cls.append_computation is not None

    @_classproperty
    def contains_compute_dependencies(cls):
        """
        True if cls itself, or anything in its own dependency_order, has
        append_computation overridden - i.e. dispatching cls may require
        first dispatching one or more other Main nodes (see dispatch(),
        which still has to walk actual instances to find out which, since
        this is only a type-level yes/no). Cached (see _classproperty), like
        dependency_order/libraries above: only ever depends on cls's own
        dependency_order (dependency *node types*, fixed once cls exists)
        and each of their has_compute_dependency() - i.e. only ever needs to
        know is-it-overridden-or-not, so it stays correct regardless of what
        any instance's own append_computation actually does, and only needs
        recomputing (impossible for a caller to trigger, short of rebuilding
        cls) if a node with none right now started with one.
        """
        return any(n.has_compute_dependency() for n in cls.dependency_order)
    __abstract__ = False
    """
    Set __abstract__ = True on a (possibly fully concrete, non-generic) type
    to mark it as having no concrete representation of its own - e.g. a base
    meant only to be subclassed. Never inherited: only a class that sets
    __abstract__ = True itself is abstract - a subclass of an abstract type is
    concrete again unless it sets __abstract__ = True itself too (see
    _NodeMeta.__new__). Enforced the same way generic types are: an abstract
    type can not be referenced by another type's field (see
    _check_concrete_field_types).
    """
    __descriptors__ = []
    """
    The descriptors this node declares on its own code. Each entry is one
    binding slot: either a plain descriptor type, or [count, descriptor_type]
    for an array of count descriptors at that one slot. In the node's own
    generated code, slot x (0 <= x < len(__descriptors__)) is written
    BINDING_SLOT(x) - a macro, #define'd around this node's own code segment
    only while it is being assembled into a dispatch (see
    _binding_slot_assignment/Main.generate_code), resolving to this node's own
    dispatch-time starting slot plus x. e.g.:
    __descriptors__ = [
        _vk.DescriptorType.SAMPLED_IMAGE,
        [4, _vk.DescriptorType.SAMPLED_IMAGE],
    ]
    equivalent in code to:
    layout(binding=BINDING_SLOT(0)) image2D RDV_NODE##_image;
    layout(binding=BINDING_SLOT(1)) image2D RDV_NODE##_images[4];
    """
    __signature__ = None
    """A unique id assigned (see _NodeMeta.__new__) to every concrete (non-generic, non-abstract) type - usable as a cache key for compiled code/pipelines."""
    _device_index = None
    """
    The device index active when this instance was created (see
    _NodeMeta.__call__). All child nodes referenced from it (inlined struct
    fields or reference/ptr fields alike, see Accessor) must share it.
    """

    @cached_property
    def device_index(self):
        return self._device_index

    def __getattr__(self, item):
        if item in type(self).__fields__:
            return getattr(self.__accessor__, item)
        raise AttributeError(item)

    def __setattr__(self, key, value):
        # only delegate actual rdv fields to __accessor__ (already built,
        # buffer-backed, before __init__ runs - see _NodeMeta.__call__) -
        # anything else (_device_index, ...) is a plain instance attribute.
        # Deliberately the SAME __accessor__ every other node uses (e.g. for
        # its own _fields_value()/device_ptr), not a second, separate one -
        # two independent accessors backed by two different buffers would
        # mean writes here never show up wherever the other one is read.
        # Common to every Node with an __accessor__ (Struct included - a
        # Struct instance's fields live in it exactly like any other node's,
        # despite older docs here claiming otherwise).
        if key in type(self).__fields__:
            setattr(self.__accessor__, key, value)
        else:
            object.__setattr__(self, key, value)

    def match(self, generic_type, value):
        """
        Call from __init__ as `self.match(argument, T)` to tie a generic
        parameter to a constructor argument. T must be a generic parameter of
        this instance's type. If T is already concrete on it, argument must be
        of that same type; otherwise T is inferred from argument's type,
        raising if that conflicts with an already-fixed value or a previous
        (different) inference this call.
        """
        if value is None:
            return
        name = generic_type.__name__
        generics = type(self).__generics__
        if name not in generics:
            raise TypeError(f'{name!r} is not a generic parameter of {type(self).__name__}.')
        current = generics.get(name)
        # A current value that's itself still unresolved (e.g. bound to a
        # still-generic Node type - see _is_unresolved) isn't really fixed
        # yet either: fall through to the pending path below the same way
        # a bare None current would, so a later, more-resolved value can
        # still take its place instead of being rejected as a conflict.
        if current is not None and not _is_unresolved(current):
            if value != current:
                raise TypeError(f'{name} is fixed to {current!r}, new value is {value!r}')
            return
        pending = self.__dict__.setdefault('__pending_generics__', {})
        existing = pending.get(name)
        if existing is not None and existing != value:
            raise TypeError(f'Conflicting inference for generic {name!r}: {existing!r} vs {value!r}')
        pending[name] = value

    @classmethod
    def generic_value(cls, generic_type):
        """
        The value currently assigned to generic_type (a generic parameter of
        this type, e.g. Generics.T) - None if it's still open/unbound.
        Raises if generic_type isn't actually a generic parameter of cls.
        """
        name = generic_type.__name__
        if name not in cls.__generics__:
            raise TypeError(f'{name!r} is not a generic parameter of {cls.__name__}.')
        return cls.__generics__[name]

    def cast(self, cast_type):
        """
        Rebuilds this instance as its own type (type(self)), re-specialized
        with cast_type's own generic bindings merged on top of type(self)'s
        current ones (by name - so this instance's own generics not shared
        with cast_type are left untouched), using the exact same constructor
        arguments this instance was originally created with.

        cast_type does not need to be type(self) itself: it can be any
        ancestor of type(self) - possibly a specialization of one - that
        shares (some of) its generic parameter names, e.g. for `b: B[T]`
        with `class B(A, Generic[T])` (so B[T] shares T with A[T]),
        `b.cast(A[int])` rebuilds b as B[int]. If cast_type leaves some of
        those shared names unbound (or doesn't mention all of type(self)'s
        own generics at all), the result can still end up generic itself -
        cast only ever applies whatever bindings cast_type actually names.
        """
        cast_root = cast_type
        while cast_root.__is_specialization__:
            cast_root = cast_root.__bases__[0]
        if not issubclass(type(self), cast_root):
            raise TypeError(f'{type(self).__name__} can not be cast to {cast_type.__name__}: '
                             f'{cast_root.__name__} is not one of its bases')
        bindings = {}
        for name, value in cast_type.__generics__.items():
            if name not in type(self).__generics__:
                raise TypeError(f'{type(self).__name__} has no generic parameter {name!r} to cast')
            bindings[name] = value
        new_cls = _specialize(type(self), bindings)
        if new_cls is type(self):
            # cast_type's bindings didn't actually narrow anything beyond
            # what type(self) already was - same type, so this exact
            # instance already satisfies it; no need to replay the
            # constructor (e.g. re-uploading a Constant's data to the GPU)
            # just to build an equal copy of itself.
            return self
        return new_cls(*self.__init_args__, **self.__init_kwargs__)

    @classmethod
    def static_resources(cls):
        """
        The actual resource (or, for an array slot, list of resources)
        bound at each of __descriptors__' slots, in order.
        Resources are independent for each device_index.
        Static resources are always the same and bound in the creation of the pipeline.
        Array element should be None for dynamic resources.
        """
        return []

    @classmethod
    def dynamic_resources(cls):
        """
        The actual resource (or, for an array slot, list of resources)
        bound at each of __descriptors__' slots, in order.
        Resources are independent for each device_index.
        Dynamic resources are bound in every dispatch.
        Array element should be None for static resources.
        """
        return None

    @classmethod
    def _custom_definitions(cls, bases, dct):
        """
        When implemented collects additional definitions from bases and
        current dct to conform the final dct.
        """
        raise NotImplementedError()

    @classmethod
    def _generate_code(cls):
        """
        When implemented generates the code for this node. Dependencies
        code are not
        """
        raise NotImplementedError()

    @classmethod
    def generated_code(cls) -> str:
        """
        This type's own generated code (see _generate_code), computed once
        per (concrete) class and cached on it - cached_property can't be
        stacked under classmethod (it calls self.func(instance), and a
        classmethod object isn't directly callable), so the cache is kept
        by hand in the class' own __dict__ instead.
        """
        if '__generated_code__' not in cls.__dict__:
            cls.__generated_code__ = cls._generate_code()
        return cls.__generated_code__


class Struct(Node, RDVType):
    """Root node for structured data types: turns its fields into a GLSL struct definition."""

    @classmethod
    def _custom_definitions(cls, bases, dct):
        pass

    @classmethod
    def _generate_code(cls) -> str:
        body = ''.join(f'    {_glsl_field_decl(k, v)}\n' for k, v in cls.__fields__.items())
        return f'struct {cls.__codename__} {{\n{body}}};'

    def values(self) -> dict:
        """
        This struct's own field values, read through its own __accessor__
        (via the usual self.x = x in __init__ - see Node.__setattr__). Used
        by another node's Accessor to copy this struct's values into one of
        its own struct-typed fields (see Accessor._set_struct).
        """
        return {k: getattr(self, k) for k in type(self).__fields__}

    def _fields_value(self):
        return self.__accessor__.rdv_values()


def _layout_of(field_type):
    """
    Converts an already-resolved field type (as stored in __fields__) into a
    plain vk layout description: a Single unwraps to its raw Type; any other
    RDVType (a Code - e.g. Component/Main - or Pointer/Buffer/Deferred, all
    backed by a plain ptr; a Struct or Tensor-like type, inlined via its own
    composite _fields_layout_decription()/__fields__) resolves through its
    own _fields_layout_decription(); [count, element] and {name: field, ...}
    recurse the same way __fields__ itself is already shaped.
    """
    if isinstance(field_type, type) and issubclass(field_type, Single):
        return field_type.__type__
    if isinstance(field_type, list):
        count, element = field_type
        return [count, _layout_of(element)]
    if isinstance(field_type, dict):
        return {k: _layout_of(v) for k, v in field_type.items()}
    if issubclass(field_type, RDVType):
        return field_type._fields_layout_decription()
    raise TypeError(f'Can not build a layout for field type {field_type!r}.')


_MEMORYVIEW_FORMATS = {
    Type.BOOL: 'B', Type.INT8: 'b', Type.UINT8: 'B', Type.INT16: 'h', Type.UINT16: 'H',
    Type.INT32: 'i', Type.UINT32: 'I', Type.INT64: 'q', Type.UINT64: 'Q',
    Type.FLOAT32: 'f', Type.FLOAT64: 'd',
}
"""
memoryview/struct format character for each scalar component Type a leaf
field can bottom out at (see Accessor._view_for) - deliberately omits
Type.FLOAT16, which has no native memoryview format and isn't used as a
field type anywhere in this codebase.
"""


def _flatten_components(value):
    """Yields value's own raw scalar components, in order - value itself for
    a plain number, or (recursively, for a matrix's own column vectors) every
    number reachable by iterating value (see Accessor._set_leaf)."""
    if isinstance(value, (int, float)):
        yield value
        return
    for item in value:
        yield from _flatten_components(item)


def _is_inline_rdvtype(field_type):
    """
    True for a non-Struct RDVType (e.g. Tensor) whose own
    _fields_layout_decription() is a composite (dict/list), inlined the
    same way a Struct is (see Accessor._set_inline_rdvtype) - as opposed to
    a plain pointer reference (is_reference, e.g. any Code - a
    Component/Main/... - or Pointer/Buffer/Deferred). The isinstance/
    issubclass guards here stay uncached (field_type may not even be a
    Node/RDVType at all, e.g. a plain dict/list/Type) - only the
    RDVType-specific part (RDVType.is_inline_rdvtype) is cached.
    """
    return (isinstance(field_type, _NodeMeta)
            and issubclass(field_type, RDVType) and field_type.is_inline_rdvtype)


class Accessor:
    """
    Handles field access, with dot notation (struct) or indexing (array), for
    an object described by a fields description shaped the same way
    __fields__ already is: a dict (struct - attribute access) or a
    [count, element] list (array - item access). If backed by a vk buffer
    (buff/layout given), every leaf write also lands in it, following the
    same inlining rule as GLSL codegen and _layout_of: a Struct-typed field,
    or any other non-reference RDVType (e.g. Tensor - see
    _is_inline_rdvtype), is inlined, recursively, into a child Accessor
    sharing this same buffer at its own sub-layout; anything is_reference
    (a Code - Component/Main/... - or Pointer/Buffer/Deferred) is a
    reference and is written as a plain device ptr, never inlined. Without a
    buffer (abstract or still-generic instances), values are only staged in
    a plain dict/list. A Struct-typed (or other inlined RDVType) field can
    only be assigned a value that is itself a matching instance, copied
    in field-by-field; an array field accepts a plain list or another
    (array-shaped) Accessor, copied in element-by-element.
    """

    _values = None
    _buff = None

    def __init__(self, fields, buff: '_vk.Buffer' = None, layout=None, device_index=None,
                 base_offset=0, views=None, fast_views=None):
        if device_index is None:
            device_index = _vk.device_index()
        assert isinstance(fields, (dict, list)), \
            'fields description must be a dict (struct) or a [count, element] list (array).'
        assert (buff is None) == (layout is None), 'buff and layout must be given together, or not at all.'
        object.__setattr__(self, '_fields', fields)
        object.__setattr__(self, '_buff', buff)
        object.__setattr__(self, '_layout', layout)
        object.__setattr__(self, '_device_index', device_index)
        # This accessor's own byte offset, cumulative from self._buff's true
        # start - a LayoutField obtained by chaining .field()/.element() off
        # a *nested* Layout (self._layout, from a compound field's own
        # sub-layout) has an .offset local to that nested Layout, not
        # cumulative from the buffer start, so a leaf write (see _set_leaf)
        # needs base_offset + the local offset to land at the right byte.
        # Computed once, by _sub, when a child accessor for a compound field
        # is first built (see _sub) - not re-walked on every write.
        object.__setattr__(self, '_base_offset', base_offset)
        # {component Type: memoryview}, shared by reference with every other
        # accessor in this same tree (root and every _sub'd child alike) -
        # see _view_for. Only the root actually owns self._buff, but any
        # accessor at any depth can populate/reuse an entry in it.
        object.__setattr__(self, '_views', views if views is not None else {})
        # Fixed-index fast path for the 3 by far most common component
        # types (int32/float32/uint64 - every pointer field is uint64, and
        # most scalar fields are int32/float32) - a 3-slot list, shared the
        # same way _views is, so write_int/write_float/write_uint64 (and
        # _view_for, for the same 3 types reached the general way) index
        # straight into it instead of a _views dict lookup.
        object.__setattr__(self, '_fast_views', fast_views if fast_views is not None else [None, None, None])
        # {key: (view, base_element_index, component_count)} for a leaf
        # field already written at least once - private to this accessor
        # (unlike _views), since a leaf key's own layout/offset are fixed
        # once and for all by this accessor's own (fixed) layout - see
        # _set_leaf. Avoids re-deriving them (a field_handle() call plus two
        # native type_size() calls) on every write to the same field.
        object.__setattr__(self, '_leaf_cache', {})
        if isinstance(fields, dict):
            values = {k: None for k in fields}
        else:
            # fields[0] (the array count) can still be a symbolic generic
            # parameter here - e.g. a still-generic instance's staged
            # accessor (no buff yet - see _build_accessor), built before
            # __init__'s own self.match(...) calls resolve it and trigger a
            # rebuild against the now-concrete class (_NodeMeta.__call__).
            # Whatever gets written to *this* accessor in the meantime is
            # discarded once that rebuild happens, so an unresolved count
            # just means "unknown yet", not an error.
            count = fields[0] if isinstance(fields[0], int) else 0
            values = [None] * (count or 0)
        object.__setattr__(self, '_values', values)

    def rdv_values(self):
        if isinstance(self._fields, dict):
            return {k: v if not isinstance(v, Accessor) else v.rdv_values() for k, v in self._values.items()}
        if isinstance(self._fields, list):
            return [v if not isinstance(v, Accessor) else v.rdv_values() for v in self._values]
        return None

    @cached_property
    def rdv_device_ptr(self):
        return self._buff.device_ptr

    @staticmethod
    def _is_compound(field_type):
        if isinstance(field_type, (dict, list)):
            return True
        if isinstance(field_type, _NodeMeta) and issubclass(field_type, RDVType):
            return field_type.is_compound
        return False

    def _field_handle(self, key):
        """The LayoutField/element handle for key within this accessor's own
        (possibly nested) layout - its own .offset is local to that layout,
        not cumulative from the buffer start (see _set_leaf/_base_offset)."""
        if self._layout is None:
            return None
        return self._layout.field(key) if isinstance(self._fields, dict) else self._layout.element(key)

    def _fast_view(self, index: int, component_type):
        """The (possibly newly built, then cached in self._fast_views[index]
        - shared by every accessor in this tree) memoryview for one of the
        3 pre-sized fast slots - see _view_for/write_int/write_float/
        write_uint64."""
        view = self._fast_views[index]
        if view is None:
            view = self._buff.memoryview().cast(_MEMORYVIEW_FORMATS[component_type])
            self._fast_views[index] = view
        return view

    @cached_property
    def as_ints(self):
        return self._fast_view(0, Type.INT32)

    @cached_property
    def as_floats(self):
        return self._fast_view(1, Type.FLOAT32)

    @cached_property
    def as_uint64s(self):
        return self._fast_view(2, Type.UINT64)

    def _view_for(self, component_type):
        """The (possibly newly built, then cached - shared by every accessor
        in this tree) memoryview over self._buff's own bytes, cast to
        component_type - see _MEMORYVIEW_FORMATS/_set_leaf. The 3 most
        common component types (see _fast_view) are a fixed-index list
        lookup instead of a _views dict one."""
        if component_type is Type.INT32:
            return self._fast_view(0, component_type)
        if component_type is Type.FLOAT32:
            return self._fast_view(1, component_type)
        if component_type is Type.UINT64:
            return self._fast_view(2, component_type)
        view = self._views.get(component_type)
        if view is None:
            view = self._buff.memoryview().cast(_MEMORYVIEW_FORMATS[component_type])
            self._views[component_type] = view
        return view

    def write_int(self, byte_offset: int, value: int) -> None:
        """Writes a raw int32 directly at byte_offset (relative to this
        accessor's own base, e.g. a field's own _field_handle(...).offset)
        - entirely bypasses field lookup/dispatch (__setattr__/_set/
        _set_leaf: dict lookups, isinstance checks, ...), unlike a plain
        `instance.some_int_field = value`. See set_direct on Main-derived
        evaluator types (MapEvaluator, Capture, ...), whose own fields are
        all fixed uint64 pointers/int32 scalars rewritten every dispatch."""
        self._fast_view(0, Type.INT32)[(self._base_offset + byte_offset) // 4] = value

    def write_float(self, byte_offset: int, value: float) -> None:
        """Writes a raw float32 directly at byte_offset - see write_int."""
        self._fast_view(1, Type.FLOAT32)[(self._base_offset + byte_offset) // 4] = value

    def write_uint64(self, byte_offset: int, value: int) -> None:
        """Writes a raw uint64 directly at byte_offset - see write_int."""
        self._fast_view(2, Type.UINT64)[(self._base_offset + byte_offset) // 8] = value

    def _sub(self, key, field_type):
        """The (possibly cached) child Accessor for a compound (struct or array) field."""
        cached = self._values[key]
        if isinstance(cached, Accessor):
            return cached
        if _is_inline_rdvtype(field_type):
            # a non-Struct RDVType's own sub-field names/shape come from its
            # _fields_layout_decription(), not __fields__ (see
            # Accessor._set_inline_rdvtype).
            desc = field_type._fields_layout_decription()
        else:
            desc = field_type.__fields__ if isinstance(field_type, _NodeMeta) else field_type
        handle = self._field_handle(key)
        sub_layout = handle.layout if handle is not None else None
        child_offset = self._base_offset + handle.offset if handle is not None else 0
        child = Accessor(desc, self._buff if sub_layout is not None else None, sub_layout, self._device_index,
                          child_offset, self._views, self._fast_views)
        self._values[key] = child
        return child

    def _check_device(self, value):
        """A child Node (inlined struct or ptr reference) must share its container's device_index."""
        if isinstance(value, Node) and self._device_index is not None and value.device_index is not None \
                and value.device_index != self._device_index:
            raise ValueError(f'{value!r} was created on device {value.device_index!r}, '
                              f'expected {self._device_index!r} (its container\'s device).')

    def _set_leaf(self, key, field_type, value):
        if isinstance(self._values, list) and isinstance(key, int) and key >= len(self._values):
            # a staged (still-generic) array accessor may have been sized 0
            # for an unresolved count (see Accessor.__init__) - just grow to
            # fit; it's discarded once the pending-generics rebuild happens.
            self._values.extend([None] * (key + 1 - len(self._values)))
        self._values[key] = value
        if self._buff is None:
            return
        write_value = value
        if isinstance(field_type, _NodeMeta) or isinstance(value, RDVType):
            write_value = resolve_ptr(value)
        # Straight to this leaf's own bytes via an indexed write on a
        # whole-buffer memoryview already cast to its component scalar type
        # (see _view_for) - no vk.Buffer.write(...) call (field-handle
        # validation, dotted-path parsing, or - for a nested field - a
        # torch/DLPack round trip through _named_access.py) on the hot path.
        cached = self._leaf_cache.get(key)
        if cached is None:
            handle = self._field_handle(key)
            layout = handle.layout
            component_type = layout.component_type
            itemsize = _vk.type_size(component_type)
            component_count = _vk.type_size(layout.type) // itemsize
            view = self._view_for(component_type)
            base = (self._base_offset + handle.offset) // itemsize
            cached = (view, base, component_count)
            self._leaf_cache[key] = cached
        view, base, component_count = cached
        if component_count == 1:
            view[base] = write_value
        else:
            for i, component in enumerate(_flatten_components(write_value)):
                view[base + i] = component

    def _set_struct(self, key, field_type, value):
        """A struct-typed field accepts either a dict of its fields, or an
        instance of that same Struct (read through its own values(), which
        itself reads through that instance's own __accessor__)."""
        if isinstance(value, dict):
            fields_dict = value
        elif isinstance(value, field_type):
            self._check_device(value)
            fields_dict = value.values()
        else:
            raise TypeError(f'{key!r} expects a {field_type.__name__} instance or a dict of its fields, got {value!r}.')
        child = self._sub(key, field_type)
        for name in field_type.__fields__:
            setattr(child, name, fields_dict[name])

    def _set_array(self, key, field_type, value):
        count, element = field_type
        child = self._sub(key, field_type)
        source = [value[i] for i in range(len(value._values))] if isinstance(value, Accessor) else list(value)
        # count can still be a symbolic generic parameter for a still-generic
        # instance's staged accessor (see Accessor.__init__) - nothing to
        # validate against yet in that case.
        if isinstance(count, int) and count and len(source) != count:
            raise ValueError(f'{key!r} expects {count} elements, got {len(source)}.')
        for i, v in enumerate(source):
            child[i] = v

    def _set_inline_rdvtype(self, key, field_type, value):
        """
        Like _set_struct, but for a non-Struct RDVType (e.g. Tensor) whose
        own _fields_layout_decription() is a composite, inlined the same way
        a Struct's __fields__ is (see _is_inline_rdvtype) - since it isn't a
        Struct, its sub-field names/values come from its own
        _fields_layout_decription()/_fields_value() (the RDVType protocol),
        not __fields__/values() (the Struct-specific one). value's type is
        checked against field_type's own root rather than strict isinstance:
        field_type can still be symbolically tied to an outer generic that's
        only pending, not yet bound, while this runs (self.match(...)'s
        rebuild happens only after __init__ returns - see
        _NodeMeta.__call__), so it may not literally be value's own type
        even when value is exactly the right kind of thing.
        """
        if isinstance(value, dict):
            desc = field_type._fields_layout_decription()
            fields_dict = value
        elif isinstance(value, RDVType) and isinstance(value, Node) \
                and getattr(type(value), '__root__', None) is getattr(field_type, '__root__', None):
            self._check_device(value)
            desc = type(value)._fields_layout_decription()
            fields_dict = value._fields_value()
        else:
            raise TypeError(f'{key!r} expects a {field_type.__name__} instance or a dict of its fields, got {value!r}.')
        child = self._sub(key, field_type)
        for name in desc:
            setattr(child, name, fields_dict[name])

    def _set(self, key, field_type, value):
        if isinstance(field_type, _NodeMeta) and issubclass(field_type, Struct):
            self._set_struct(key, field_type, value)
        elif _is_inline_rdvtype(field_type):
            self._set_inline_rdvtype(key, field_type, value)
        elif isinstance(field_type, list):
            self._set_array(key, field_type, value)
        elif isinstance(field_type, dict):
            raise TypeError(f'{key!r} is an inner struct; set its own fields individually.')
        else:
            self._set_leaf(key, field_type, value)

    def __getattr__(self, name):
        if not isinstance(self._fields, dict) or name not in self._fields:
            raise AttributeError(name)
        field_type = self._fields[name]
        return self._sub(name, field_type) if self._is_compound(field_type) else self._values[name]

    def __setattr__(self, name, value):
        if not isinstance(self._fields, dict) or name not in self._fields:
            raise AttributeError(f'{name!r} is not a field of this object.')
        self._set(name, self._fields[name], value)

    def __getitem__(self, index):
        if not isinstance(self._fields, list):
            raise TypeError('This accessor does not represent an array.')
        field_type = self._fields[1]
        return self._sub(index, field_type) if self._is_compound(field_type) else self._values[index]

    def __setitem__(self, index, value):
        if not isinstance(self._fields, list):
            raise TypeError('This accessor does not represent an array.')
        count = self._fields[0]
        # count can still be a symbolic generic parameter (see
        # Accessor.__init__/_set_array) - nothing to bounds-check yet then.
        if isinstance(count, int) and count and not (0 <= index < count):
            raise IndexError(index)
        self._set(index, self._fields[1], value)


_DUMMY_LAYOUT = {'__dummy__': Type.FLOAT32}


def _build_accessor(cls, device_index=None):
    """
    This (already final) type's own Accessor: staged in a plain dict if cls
    is __abstract__ or still generic (no concrete layout exists yet for it),
    otherwise backed by a fresh vk buffer laid out from cls.__fields__ - a
    buffer can't be created empty, so a type with no fields gets a dummy
    single-float layout instead (the Accessor itself still only exposes
    cls.__fields__, i.e. none - the dummy layout only exists to back the
    buffer's own internal representation).
    """
    if cls.__is_generic__ or cls.__abstract__:
        return Accessor(cls.__fields__, device_index=device_index)
    description = _layout_of(cls.__fields__) or _DUMMY_LAYOUT
    buff = _vk.buffer(description, location=_vk.MemoryLocation.HOST)
    return Accessor(cls.__fields__, buff, buff.element_layout, device_index)


def _collect_snippets(cls, attr_name):
    """
    Accumulation logic shared by the "like headers" concepts of specialized
    node kinds (Main's __header__, Code's __precode__/__poscode__): every
    attr_name (a code snippet string) declared by cls's own ancestry or by one
    of its dependencies (__requires__), keyed by the declaring type - so a
    diamond-repeated declaration only ever contributes once. Order is
    dependencies first (sorted by codename, for a stable order), then this
    type's own ancestry, base to derived.

    A Library dependency is skipped here: unlike a plain forward-declared
    prototype (e.g. Image.__precode__ - harmless to repeat), a Library's own
    precode/poscode (e.g. DeferredCore's binding-slot-specific SSBO
    declaration) is only valid emitted exactly once, and it already is -
    every __requires__ member (Library or not) gets its own dedicated,
    singular entry in dependency_order, which _build_code uses to
    emit that member's own full generated_code() (precode included) already,
    in the correct, topologically-sound position. Re-embedding a Library's
    precode here as well would duplicate a stateful resource declaration
    (wrong/conflicting BINDING_SLOT(...) the second time, since that macro
    is bound to *this* node's own slot range while cls's own snippets are
    being assembled, not the Library's).
    """
    snippets = {}
    for dep in sorted(cls.__requires__, key=lambda d: d.__codename__):
        if issubclass(dep, Library):
            continue
        snippet = dep.__dict__.get(attr_name)
        if snippet:
            snippets[dep] = _as_include(snippet, dep)
    for klass in reversed(cls.__mro__):
        snippet = klass.__dict__.get(attr_name)
        if snippet:
            snippets[klass] = _as_include(snippet, klass)
    return snippets


def _resolve_code(cls):
    """
    This type's own __code__ if it declares one; otherwise exactly one of its
    direct bases must provide one (differing values from more than one base
    raise a TypeError - an ambiguous __code__). Direct bases already hold
    their own resolved (not raw-declared) __code__, since every type gets
    this same resolution applied at its own creation (see
    Code._custom_definitions), so a single-level lookup is enough to walk
    the whole chain. A base's own __include_dir__ (see _as_include) is
    copied onto cls too when its __code__ is inherited this way, so
    _build_pipeline still finds it via cls's own dependency_order even if
    that base itself isn't a __requires__ member (dependency_order walks
    __requires__, not __mro__).
    """
    own = cls.__dict__.get('__code__')
    if own is not None:
        return _as_include(own, cls)
    code = None
    for base in cls.__bases__:
        base_code = getattr(base, '__code__', None)
        if base_code is not None:
            if code is not None and code != base_code:
                raise TypeError(f'{cls.__name__}: more than one base defines a conflicting __code__')
            code = base_code
            base_include_dir = getattr(base, '__include_dir__', None)
            if base_include_dir:
                cls.__include_dir__ = base_include_dir
    return code


class Code(Node, RDVType):
    """Root node for customized pieces of megakernel code."""

    __code__ = None
    """
    This node's own GLSL code body, resolved (see _resolve_code) once at
    class creation. Either literal GLSL text, or a path (e.g.
    "shaders/core/tensor.h") to a file holding it under the package's own shaders/
    directory, or under this node's own declaring .py file's directory -
    see _as_include - auto-turned into the #include directive for it
    instead of being emitted (and misparsed) as-is.
    """
    __include_dir__ = None
    """Set by _as_include when __code__/__precode__/__poscode__ resolved
    to a file next to this node's own declaring .py module (rather than
    under the package's own _HLSL_DIR) - the directory _build_pipeline
    adds to the shader compiler's own include_dirs so that #include
    directive can actually be found."""
    __precode__ = None
    """Code (or an shaders/... path - see __code__ above) this node's own
    declaration adds prior to __code__ for every subclass of it."""
    __poscode__ = None
    """Code (or an shaders/... path - see __code__ above) this node's own
    declaration adds posterior to __code__ for every subclass of it."""
    __extended_code__ = None
    """Extra code added posterior to __code__, for this node only (never inherited)."""

    @classmethod
    def _custom_definitions(cls, bases, dct):
        cls.__code__ = _resolve_code(cls)
        cls.__extended_code__ = dct.get('__extended_code__') or ''

    @classmethod
    def _generate_code(cls):
        precode = '\n'.join(_collect_snippets(cls, '__precode__').values())
        poscode = '\n'.join(reversed(list(_collect_snippets(cls, '__poscode__').values())))
        return '\n'.join(part for part in (precode, cls.__code__, cls.__extended_code__, poscode) if part)

    @classmethod
    def _fields_layout_decription(cls):
        # A Code instance is always referenced by pointer wherever it's used
        # as a field elsewhere (see _generate_code's own {codename}_Ref
        # pattern - the "real" fields live in a separately-allocated Ref
        # buffer, this struct itself just holds a pointer to it) - never
        # inlined, unlike Struct.
        return Type.UINT64

    def _fields_value(self):
        return self.__accessor__.rdv_device_ptr


class Library(Code):
    """Root node for collections of code/struct definitions shared across computes."""
    __precode__ = "shaders/core/library_precode.h"
    __code__ = "shaders/core/library.h"

    @classmethod
    def _custom_definitions(cls, bases, dct):
        assert not cls.__fields__, "Libraries are stateless types."
        super()._custom_definitions(bases, dct)


class Component(Code):
    """
    Base class for describing reusable behavior with associated state.
    """

    @classmethod
    def _generate_code(cls):
        if cls.__is_generic__:
            raise RuntimeError(f'Cannot generate code for a generic type {cls.__name__}.')
        # lowercase _ref (not _Ref) to match the convention every reference
        # type (Pointer/Buffer/FixArray) uses, and that Deferred's resolve()
        # macro (RDV_CONCAT(T,_ref)) relies on to cast T's buffer_reference
        # from a raw pointer, regardless of which one T actually is.
        ref_codename = f'{cls.__codename__}_ref'
        data_codename = f'{cls.__codename__}_data'
        # A GLSL struct body can't be empty - a Component with no real
        # fields (e.g. Identity) silently gets one hidden dummy float here,
        # matching _DUMMY_LAYOUT's own fallback for its host-side buffer
        # (_build_accessor) - never added to __fields__ itself, so it's
        # invisible to users/the accessor/field introspection, only ever
        # exists in the generated text.
        fields_body = ''.join(f'    {_glsl_field_decl(k, v)}\n' for k, v in cls.__fields__.items()) or '    float __pad__;\n'
        params_code = f"""
struct {data_codename} {{\n{fields_body}}};
layout(buffer_reference, scalar) buffer {ref_codename} {{\n {data_codename} data; }};
struct {cls.__codename__} {{
    {ref_codename} rdv_ref;
}};
"""
        return params_code + super()._generate_code()


def _binding_slot_assignment(order):
    """
    Assigns each node in order (see Node.dependency_order) its own contiguous
    range of BINDING_SLOT(x) values, one after another in that same order - a
    node with N descriptors (len(__descriptors__)) owns slots [start, start+N).
    Returns (slot_by_node, total_slot_count).
    """
    slots = {}
    current = 1
    for n in order:
        if not isinstance(n, _NodeMeta):
            continue
        slots[n] = current
        current += len(n.__descriptors__)
    return slots, current


class Main(Component):
    """
    Root node for a compute entry point. __header__ is this type's own header
    snippet (a string), accumulated (see _collect_snippets) from this type's
    own ancestry and its dependencies, each contributing at most once.
    __code__ is this type's own GLSL body, appended last.
    """

    __precode__ = "shaders/core/main_precode.h"

    @classmethod
    def megakernel_code(cls, webgpu: bool = False) -> str:
        return _build_code(cls, webgpu)

    def export_webgpu(self, path: str | None = None, **kwargs) -> str:
        """
        Exports this node as a self-contained JavaScript module (WGSL
        megakernel + the WebGPU machinery to run it) - see
        rdv.core._webgpu_export.export_webgpu for the details and options.
        """
        from ._webgpu_export import export_webgpu
        return export_webgpu(self, path, **kwargs)

__dispatch_pipelines__ = {}  # (main.__signature__, device_index) -> (pipeline, descriptor_set)
"""Cached pipelines and descriptor sets with bound resources."""
__dispatch_commands__ = _weakref.WeakKeyDictionary()  # main_node instance -> {threads: command_buffer}
"""
Cached command buffers, per main_node *instance* (not type/signature, like
__dispatch_pipelines__ above) - now that main_node's own device pointer is
baked into the command buffer as a push constant at record time (see
_build_pipeline/dispatch), rather than read from a separate buffer whose
content could be rewritten between dispatches, a single recording can only
ever serve that exact instance. Weak-keyed so a main_node's own cached
command buffers (and the GPU command-pool memory they hold) are freed once
nothing else references it anymore, instead of pinning every main_node
that's ever been dispatched for the lifetime of the process.
"""
__update_after_bind_caps__ = {}  # device_index -> bool
"""Per-device cache of _vk.caps().can_update_after_bind (see dispatch())."""


def _build_code(cls: _NodeMeta, webgpu: bool = False) -> str:
    """
    The full megakernel GLSL for cls. webgpu=True builds the variant fed to
    the WebGPU export (see _webgpu.py): RDV_WEBGPU is defined (shaders can
    #ifdef around anything that can't survive 64-bit narrowing), and
    cooperative matrices/float16 are never enabled, whatever the current
    device supports.
    """
    if cls.__is_generic__:
        raise RuntimeError("Cannot build code for generic main node.")
    assert issubclass(cls, Main)
    if hasattr(cls, "__local_size__"):
        local_size = cls.__local_size__
    else:
        local_size = (32, 1, 1)
    order = cls.dependency_order
    slots, _ = _binding_slot_assignment(order)
    # Conditionally unlocked, not unconditionally #extension'd like the
    # rest of the header below: GL_KHR_cooperative_matrix is only usable
    # (and only requestable at device-creation time in the first place -
    # see Device::vk_init()) when the current device's own Caps says so.
    # RDV_COOPERATIVE_MATRIX_SUPPORT is how a shader (e.g.
    # shaders/maps/linear_fast.h) tells the two cases apart - never
    # assume it, always #ifdef. Never enabled for the WebGPU target, which
    # gets RDV_WEBGPU instead.
    target_defines = "#define RDV_WEBGPU\n" if webgpu else ""
    if not webgpu and _vk.caps().supports_cooperative_matrix:
        target_defines = (
            "#extension GL_KHR_cooperative_matrix : require\n"
            "#define RDV_COOPERATIVE_MATRIX_SUPPORT\n"
            # Declared here (global/early), not just inside whichever
            # coopmat-using node's own __code__ happens to need float16_t -
            # a Buffer[Type.FLOAT16] field (e.g. LinearTensorOpFast's own
            # pre-tiled weight buffer) can be a *dependency*, emitted
            # earlier in dependency order than the node that actually
            # requires the extension, so declaring it only there is too
            # late (float16_t isn't a recognized type yet at that point).
            "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n"
        )
    code = f"""
// =============== BEGIN MEGAKERNEL HEADER ================
#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_debug_printf : require
#extension GL_EXT_control_flow_attributes : require
{target_defines}layout (local_size_x = {local_size[0]}, local_size_y = {local_size[1]}, local_size_z = {local_size[2]}) in;
#define GPUPtr uint64_t
#define PRINT debugPrintfEXT
#define RDV_CONCAT_(a, b) a##b
#define RDV_CONCAT(a, b) RDV_CONCAT_(a, b)
#define deref(x) ((x).rdv_ref).data
#define cast(T, ptr) T(RDV_CONCAT(T,_ref)(ptr))
#define address(ptr) uint64_t((ptr).rdv_ref)
#define is_null(ptr) (((ptr).rdv_ref)==0)
// =============== END MEGAKERNEL HEADER ================

"""
    for n in order:
        start_slot = slots[n]
        code += f"/* Generated code for {n.__codename__} */\n"
        code += "\n"
        code += f"#define RDV_NODE {n.__codename__}\n"
        if len(n.__descriptors__) > 0:
            code += f"#define BINDING_SLOT(n) ((n)+{start_slot})\n"
        for gk, gv in n.__generics__.items():
            # a generic bound to a "builtin" (int/float/vec3/... or a raw
            # Type value like Type.FLOAT32) is still in its raw, friendly
            # form here - canonicalize to Single[Type.X] first, same as a
            # field annotation would (see _normalize_field), since
            # _glsl_type_of only understands Single/RDVType/Node types, not
            # raw builtins or bare Type values.
            if isinstance(gv, Type):
                gv = _glsl_type_of(Single[gv])
            elif isinstance(gv, type):
                if gv in _PRIMITIVES:
                    gv = Single[_PRIMITIVES[gv]]
                gv = _glsl_type_of(gv)
            code += f"#define {gk} {gv} \n"
        for ak, av in n.__aliases__.items():
            # aliases are always already-concrete (see _own_aliases_of), but
            # may still be in "builtin" raw form, same canonicalization as
            # generics just above - including a plain int (e.g. 1/0, from a
            # resolved _GenericComparison - see Deferred.USE_RAW_POINTER),
            # which #define's directly, the same way a plain-int *generic*
            # already does above (there's no type to translate via
            # _glsl_type_of for those).
            if isinstance(av, Type):
                av = _glsl_type_of(Single[av])
            elif isinstance(av, type):
                if av in _PRIMITIVES:
                    av = Single[_PRIMITIVES[av]]
                av = _glsl_type_of(av)
            code += f"#define {ak} {av} \n"
        code += n.generated_code() + "\n"
        for gk in n.__generics__:
            code += f"#undef {gk}\n"
        for ak in n.__aliases__:
            code += f"#undef {ak}\n"
        if len(n.__descriptors__) > 0:
            code += f"#undef BINDING_SLOT\n"
        code += f"#undef RDV_NODE\n"
        code += "\n\n"

    init_codes = ""
    for n in cls.libraries:
        init_codes += f"    {n.__codename__}(global_id);\n"
    code += f"""
layout(push_constant) uniform rdv_dispatcher {{
    {cls.__codename__} rdv_main_node;
    // total dimensions (not necessary the dispatch size if batched)
    int rdv_dim_x;
    int rdv_dim_y;
    int rdv_dim_z;
    // current batch offset
    int rdv_start_x;
    int rdv_start_y;
    int rdv_start_z;
}};

void main() {{
    uvec3 global_id = gl_GlobalInvocationID;
    global_id += uvec3(rdv_start_x, rdv_start_y, rdv_start_z);
    if (any(greaterThanEqual(global_id, uvec3(rdv_dim_x, rdv_dim_y, rdv_dim_z))))
        return; // Silently skip overflow indices 
    {init_codes}
    rdv_main(rdv_main_node, global_id);
}}
    """
    return code


def _build_pipeline(main_node: Main):
    """
    Best-effort pipeline + descriptor set for main_node: sizes the descriptor
    set from the same dependency order/BINDING_SLOT counting generate_code()
    itself uses (see Node.dependency_order/_binding_slot_assignment), then
    writes main_node's own resources at its own assigned slots. Dependency
    node types found only through the requires graph (not as a live
    instance reachable from main_node) contribute slots but no resource
    writes here.
    """
    with _vk.device(main_node.device_index):
        cls = type(main_node)
        order = cls.dependency_order
        slots, total_slots = _binding_slot_assignment(order)
        pipeline = _vk.pipeline(_vk.PipelineType.COMPUTE)
        pipeline.local_size(*main_node.__local_size__)
        for n in order:
            start_slot = slots[n]
            for i, descriptor in enumerate(n.__descriptors__):
                if isinstance(descriptor, list):
                    count, descriptor_type = descriptor
                else:
                    count = None
                    descriptor_type = descriptor
                pipeline.layout(0, start_slot + i, **{f'b{start_slot + i}': descriptor_type if count is None else (descriptor_type, count)})
        # rdv_main_node (main_node's own device pointer) plus the dispatch
        # dims/offset ints, pushed fresh into the command buffer being
        # recorded every time dispatch() (re)records one (see there) -
        # field order must match the rdv_dispatcher struct in _build_code
        # exactly (both computed under LayoutRule.Std430 - see
        # Pipeline.layout_constants). Replaces the old "dispatcher" uniform
        # buffer, one pipeline-wide binding whose *content* got rewritten
        # before every dispatch() call so one pre-recorded command buffer
        # could be reused across every instance of this same Main type -
        # a push constant is baked into the command buffer at record time
        # instead, so that trick no longer applies: the command buffer is
        # now cached per main_node instance (see __dispatch_commands__).
        pipeline.layout_constants(
            rdv_main_node=Type.UINT64,
            rdv_dim_x=Type.INT32, rdv_dim_y=Type.INT32, rdv_dim_z=Type.INT32,
            rdv_start_x=Type.INT32, rdv_start_y=Type.INT32, rdv_start_z=Type.INT32,
        )
        code = _build_code(cls)
        # extra per-node include dirs (see _as_include/Code.__include_dir__) -
        # order already covers cls plus every transitive __requires__
        # dependency; cls.__mro__ on top of that catches a __code__/
        # __precode__/__poscode__ inherited from (or contributed by) a base
        # that isn't itself a __requires__ member.
        extra_dirs = {d for n in order if (d := getattr(n, '__include_dir__', None))}
        extra_dirs.update(d for base in cls.__mro__ if (d := base.__dict__.get('__include_dir__')))
        try:
            shader_source = _vk.ShaderSource.from_glsl(code, _vk.ShaderStageType.COMPUTE, include_dirs=[_HLSL_DIR, *extra_dirs])
        except Exception as e:
            lines = code.split('\n')
            numbered = [str(line_number + 1) + "\t" + line for line_number, line in enumerate(lines) ]
            print("\n".join(numbered))
            raise e
        pipeline.stage(_vk.ShaderStageType.COMPUTE, shader_source)
        pipeline.close()
        try:
            descriptor_set: _vk.DescriptorSet = pipeline.descriptor_set(0)
        except:
            descriptor_set = None  # No descriptor set
        i = 0
        for n in order:
            for resource in n.static_resources():
                if resource is not None:
                    descriptor_set.bind(**{f'b{i+1}': resource})
                i += 1
    return pipeline, descriptor_set


def _ensure_pipeline(node):
    key = (type(node).__signature__, node.device_index)
    if key not in __dispatch_pipelines__:
        __dispatch_pipelines__[key] = _build_pipeline(node)
    return __dispatch_pipelines__[key]


def _bind_dynamic_resources(node, dset):
    # Shared by dispatch() (main_node's own descriptor set) and
    # append_dispatch (any other node's - a compute_dependency's, typically).
    has_dynamic = False
    j = 0
    for n in type(node).libraries:
        d_resources = n.dynamic_resources()
        if d_resources is None:
            j += len(n.__descriptors__)
        else:
            for resource in d_resources:
                if resource is not None:
                    if isinstance(resource, list) and resource and isinstance(resource[0], tuple):
                        # A sparse registry (e.g. ImagingLibrary, whose slots
                        # can be freed/reused - see ResourceHandle) can't
                        # always bind its occupied slots as one contiguous
                        # array from 0: it's a list of (resource, index)
                        # pairs instead, one explicit-slot bind() call each.
                        for item in resource:
                            dset.bind(**{f'b{j + 1}': item})
                    else:
                        dset.bind(**{f'b{j + 1}': resource})
                    has_dynamic = True
                j += 1
    return has_dynamic


def append_dispatch(cmd, node: Main, threads: tuple[int, int, int]):
    """
    Records node's own set_pipeline/bind/bind_constants/dispatch_threads
    group into cmd - the one recording primitive dispatch() and every
    Node.append_computation() override (see has_compute_dependency) share,
    so a barrier-like node's own dispatch is recorded identically whether
    it's the thing being dispatched directly or a compute_dependency of
    something else.
    """
    pipeline, dset = _ensure_pipeline(node)
    _bind_dynamic_resources(node, dset)
    threads_x, threads_y, threads_z = threads
    cmd.set_pipeline(pipeline)
    if dset is not None:
        cmd.bind(0, [dset])
    cmd.bind_constants(
        rdv_main_node=node.__accessor__.rdv_device_ptr,
        rdv_dim_x=threads_x, rdv_dim_y=threads_y, rdv_dim_z=threads_z,
        rdv_start_x=0, rdv_start_y=0, rdv_start_z=0,
    )
    cmd.dispatch_threads(threads_x, threads_y, threads_z)


def _collect_compute_dependencies(node, seen_ids: set, out: list):
    """
    Depth-first walk of node's own Node-typed fields (postorder, so a
    dependency's own dependencies land before it), collecting every distinct
    (by identity) instance whose type has_compute_dependency() - see
    dispatch(). Only descends into a field at all where its declared type's
    own contains_compute_dependencies says it's worth it, so a graph with no
    barriers anywhere costs one cached bool check per field, not an attribute
    read.
    """
    for name, field_type in type(node).__fields__.items():
        if not isinstance(field_type, _NodeMeta) or not field_type.contains_compute_dependencies:
            continue
        value = getattr(node, name)
        if not isinstance(value, Node):
            continue
        _collect_compute_dependencies(value, seen_ids, out)
        if type(value).has_compute_dependency() and id(value) not in seen_ids:
            seen_ids.add(id(value))
            out.append(value)


def dispatch(main_node: Main, threads_x: int, threads_y: int = 1, threads_z: int = 1):
    """
    Dispatches a grid of threads for a main node. Pipelines/descriptor sets
    are cached per (type, device_index); command buffers additionally per
    thread-group size.
    """
    pipeline, descriptor_set = _ensure_pipeline(main_node)
    has_dynamic_resources = _bind_dynamic_resources(main_node, descriptor_set)

    threads = (threads_x, threads_y, threads_z)
    # A node with truly dynamic resources (e.g. ImagingLibrary's image/
    # sampler arrays, whose bound contents can grow/change from one
    # dispatch to the next - see GPUSampler) can't share a cached, already-
    # recorded command buffer across calls on a device that doesn't support
    # UPDATE_AFTER_BIND: resubmitting a previously closed command buffer
    # after descriptor_set.bind() adds new array elements does NOT pick up
    # those new elements on such a driver (only a freshly recorded
    # vkCmdBindDescriptorSets does), even after waiting for the prior
    # submission to fully complete. Pipeline.close() already opts every
    # binding into UPDATE_AFTER_BIND wherever the device supports it, so
    # this only forces a re-record as a fallback where it doesn't.
    device_idx = main_node.device_index
    can_update_after_bind = __update_after_bind_caps__.get(device_idx)
    if can_update_after_bind is None:
        with _vk.device(device_idx):
            can_update_after_bind = _vk.caps().can_update_after_bind
        __update_after_bind_caps__[device_idx] = can_update_after_bind

    # Every instance found across main_node's own field graph (only where
    # contains_compute_dependencies says it's worth looking - see
    # _collect_compute_dependencies) whose type overrides append_computation,
    # in postorder (a dependency's own dependencies first), deduplicated by
    # identity (a given instance can be reachable through more than one
    # field) - each records its own compute work into the same command
    # buffer, before main_node's own group, below. A recording is never
    # cacheable once any of these are present: unlike main_node's own
    # push-constant-embedded pointer, nothing here re-validates that a
    # *cached* recording's dependencies are still the right ones from one
    # dispatch() call to the next.
    compute_dependencies = []
    if type(main_node).contains_compute_dependencies:
        _collect_compute_dependencies(main_node, set(), compute_dependencies)
    needs_rerecord = (has_dynamic_resources and not can_update_after_bind) or bool(compute_dependencies)

    # Keyed by main_node instance (weakly - see __dispatch_commands__),
    # then by threads: main_node's own device pointer is now baked into
    # the command buffer as a push constant at record time (see
    # _build_pipeline), so unlike the old dispatcher-buffer scheme, a
    # given recording is only ever valid for this exact instance.
    node_commands = __dispatch_commands__.get(main_node)
    if node_commands is None:
        node_commands = {}
        __dispatch_commands__[main_node] = node_commands
    command_buffer = None if needs_rerecord else node_commands.get(threads)
    if command_buffer is None:
        # Command buffers are allocated from a pool tied to whatever engine
        # is current at creation time - must be the same explicit
        # EngineType.COMPUTE engine used at submit time below, or an
        # intervening call that lazily activates some other default engine
        # (e.g. a plain buffer/tensor allocation) can leave this command
        # buffer's pool on a different queue family than the one it's later
        # submitted to (a validation-layer-catchable Vulkan error).
        with _vk.device(main_node.device_index):
            with _vk.engine(EngineType.COMPUTE):
                command_buffer = _vk.command_buffer()
                for dep in compute_dependencies:
                    dep.append_computation(command_buffer)
                    # Global memory barrier between this compute_dependency's
                    # own dispatch and whatever reads its output next (the
                    # following dependency, or main_node's own group below) -
                    # a compute-to-compute hazard otherwise has no implicit
                    # ordering within one command buffer.
                    command_buffer.barrier()
                append_dispatch(command_buffer, main_node, threads)
                command_buffer.close()
        if not needs_rerecord:
            node_commands[threads] = command_buffer

    if __PROFILING__:
        start_t = _time.perf_counter()
    with _vk.device(main_node.device_index):
        with _vk.engine(EngineType.COMPUTE):
            _vk.submit_and_wait(command_buffer)
            # _vk.submit([command_buffer]).wait()
    if __PROFILING__:
        start_t = _time.perf_counter() - start_t
        global __submit_times__
        __submit_times__ += start_t


def _build_code_saved(cls: _NodeMeta) -> str:
    if cls.__is_generic__:
        raise RuntimeError("Cannot build code for generic main node.")
    assert issubclass(cls, Main)
    if hasattr(cls, "__local_size__"):
        local_size = cls.__local_size__
    else:
        local_size = (32, 1, 1)
    order = cls.dependency_order
    slots, _ = _binding_slot_assignment(order)
    code = f"""
// =============== BEGIN MEGAKERNEL HEADER ================
#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_debug_printf : require
layout (local_size_x = {local_size[0]}, local_size_y = {local_size[1]}, local_size_z = {local_size[2]}) in;
#define GPUPtr uint64_t
#define PRINT debugPrintfEXT
#define RDV_CONCAT_(a, b) a##b
#define RDV_CONCAT(a, b) RDV_CONCAT_(a, b)
#define deref(x) ((x).rdv_ref).data
#define cast(T, ptr) T(RDV_CONCAT(T,_ref)(ptr))
#define address(ptr) uint64_t((ptr).rdv_ref)
#define is_null(ptr) (((ptr).rdv_ref)==0)
// =============== END MEGAKERNEL HEADER ================

"""
    for n in order:
        start_slot = slots[n]
        code += f"/* Generated code for {n.__codename__} */\n"
        code += "\n"
        code += f"#define RDV_NODE {n.__codename__}\n"
        if len(n.__descriptors__) > 0:
            code += f"#define BINDING_SLOT(n) ((n)+{start_slot})\n"
        for gk, gv in n.__generics__.items():
            # a generic bound to a "builtin" (int/float/vec3/... or a raw
            # Type value like Type.FLOAT32) is still in its raw, friendly
            # form here - canonicalize to Single[Type.X] first, same as a
            # field annotation would (see _normalize_field), since
            # _glsl_type_of only understands Single/RDVType/Node types, not
            # raw builtins or bare Type values.
            if isinstance(gv, Type):
                gv = _glsl_type_of(Single[gv])
            elif isinstance(gv, type):
                if gv in _PRIMITIVES:
                    gv = Single[_PRIMITIVES[gv]]
                gv = _glsl_type_of(gv)
            code += f"#define {gk} {gv} \n"
        for ak, av in n.__aliases__.items():
            # aliases are always already-concrete (see _own_aliases_of), but
            # may still be in "builtin" raw form, same canonicalization as
            # generics just above - including a plain int (e.g. 1/0, from a
            # resolved _GenericComparison - see Deferred.USE_RAW_POINTER),
            # which #define's directly, the same way a plain-int *generic*
            # already does above (there's no type to translate via
            # _glsl_type_of for those).
            if isinstance(av, Type):
                av = _glsl_type_of(Single[av])
            elif isinstance(av, type):
                if av in _PRIMITIVES:
                    av = Single[_PRIMITIVES[av]]
                av = _glsl_type_of(av)
            code += f"#define {ak} {av} \n"
        code += n.generated_code() + "\n"
        for gk in n.__generics__:
            code += f"#undef {gk}\n"
        for ak in n.__aliases__:
            code += f"#undef {ak}\n"
        if len(n.__descriptors__) > 0:
            code += f"#undef BINDING_SLOT\n"
        code += f"#undef RDV_NODE\n"
        code += "\n\n"

    init_codes = ""
    for n in cls.libraries:
        init_codes += f"    {n.__codename__}(global_id);\n"
    code += f"""
layout(set = 0, binding = 0, scalar) uniform rdv_dispatcher {{
    {cls.__codename__} rdv_main_node;
    // total dimensions (not necessary the dispatch size if batched)
    int rdv_dim_x;
    int rdv_dim_y;
    int rdv_dim_z;
    // current batch offset
    int rdv_start_x;
    int rdv_start_y;
    int rdv_start_z;
}};

void main() {{
    uvec3 global_id = gl_GlobalInvocationID;
    global_id += uvec3(rdv_start_x, rdv_start_y, rdv_start_z);
    if (any(greaterThanEqual(global_id, uvec3(rdv_dim_x, rdv_dim_y, rdv_dim_z))))
        return; // Silently skip overflow indices 
    {init_codes}
    rdv_main(rdv_main_node, global_id);
}}
    """
    return code

def _build_pipeline_saved(main_node: Main):
    """
    Best-effort pipeline + descriptor set for main_node: sizes the descriptor
    set from the same dependency order/BINDING_SLOT counting generate_code()
    itself uses (see Node.dependency_order/_binding_slot_assignment), then
    writes main_node's own resources at its own assigned slots. Dependency
    node types found only through the requires graph (not as a live
    instance reachable from main_node) contribute slots but no resource
    writes here.
    """
    with _vk.device(main_node.device_index):
        cls = type(main_node)
        order = cls.dependency_order
        slots, total_slots = _binding_slot_assignment(order)
        pipeline = _vk.pipeline(_vk.PipelineType.COMPUTE)
        pipeline.local_size(*main_node.__local_size__)
        pipeline.layout(set=0, start_binding=0, dispatcher=_vk.DescriptorType.UNIFORM_BUFFER)
        for n in order:
            start_slot = slots[n]
            for i, descriptor in enumerate(n.__descriptors__):
                if isinstance(descriptor, list):
                    count, descriptor_type = descriptor
                else:
                    count = None
                    descriptor_type = descriptor
                pipeline.layout(0, start_slot + i, **{f'b{start_slot + i}': descriptor_type if count is None else (descriptor_type, count)})
        code = _build_code(cls)
        # extra per-node include dirs (see _as_include/Code.__include_dir__) -
        # order already covers cls plus every transitive __requires__
        # dependency; cls.__mro__ on top of that catches a __code__/
        # __precode__/__poscode__ inherited from (or contributed by) a base
        # that isn't itself a __requires__ member.
        extra_dirs = {d for n in order if (d := getattr(n, '__include_dir__', None))}
        extra_dirs.update(d for base in cls.__mro__ if (d := base.__dict__.get('__include_dir__')))
        try:
            shader_source = _vk.ShaderSource.from_glsl(code, _vk.ShaderStageType.COMPUTE, include_dirs=[_HLSL_DIR, *extra_dirs])
        except Exception as e:
            lines = code.split('\n')
            numbered = [str(line_number + 1) + "\t" + line for line_number, line in enumerate(lines) ]
            print("\n".join(numbered))
            raise e
        pipeline.stage(_vk.ShaderStageType.COMPUTE, shader_source)
        pipeline.close()
        descriptor_set: _vk.DescriptorSet = pipeline.descriptor_set(0)
        # binding 0 ("dispatcher") is the rdv_dispatcher uniform every
        # generated shader reads (dispatch dimensions/offset plus the
        # pointer to main_node's own fields buffer, as rdv_main_node) - its
        # actual content is written fresh before every dispatch() call (see
        # below), since it depends on that call's thread counts and on
        # main_node itself (the same pipeline is cached and reused across
        # every instance of this same concrete Main type).
        # field order must match the rdv_dispatcher struct in _build_code
        # exactly (scalar layout assigns offsets purely by declaration
        # order) - rdv_main_node first, then the dims/offset ints.
        dispatcher_buff = _vk.buffer(dict(
            rdv_main_node=Type.UINT64,
            rdv_dim_x=Type.INT32, rdv_dim_y=Type.INT32, rdv_dim_z=Type.INT32,
            rdv_start_x=Type.INT32, rdv_start_y=Type.INT32, rdv_start_z=Type.INT32,
        ), location=_vk.MemoryLocation.HOST)
        descriptor_set.bind(dispatcher=dispatcher_buff)
        i = 0
        for n in order:
            for resource in n.static_resources():
                if resource is not None:
                    descriptor_set.bind(**{f'b{i+1}': resource})
                i += 1
        # rdv_main_node (8 bytes, one uint64) followed immediately by the 6
        # dims/offset ints (4 bytes each, no padding between them - verified
        # against the buffer's own computed layout: offsets 0 and 8..28) -
        # two raw memoryviews (byte format 'B', reinterpreted once here to
        # 'Q'/'i') directly over those two byte ranges let dispatch() write
        # every field with a plain indexed assignment instead of 7 separate
        # buffer.write(field, value) calls each dispatch.
        dispatcher_view = dispatcher_buff.memoryview()
        main_node_view = dispatcher_view[0:8].cast('Q')
        dims_view = dispatcher_view[8:32].cast('i')
    return pipeline, descriptor_set, dispatcher_buff, main_node_view, dims_view

def dispatch_saved(main_node: Main, threads_x: int, threads_y: int = 1, threads_z: int = 1):
    """
    Dispatches a grid of threads for a main node. Pipelines/descriptor sets
    are cached per (type, device_index); command buffers additionally per
    thread-group size.
    """
    pipeline_key = (type(main_node).__signature__, main_node.device_index)
    if pipeline_key not in __dispatch_pipelines__:
        __dispatch_pipelines__[pipeline_key] = _build_pipeline(main_node)
    pipeline, descriptor_set, dispatcher_buff, main_node_view, dims_view = __dispatch_pipelines__[pipeline_key]

    # bind dynamic resources
    order = type(main_node).libraries
    i = 0
    has_dynamic_resources = False
    for n in order:
        d_resources = n.dynamic_resources()
        if d_resources is None:
            i += len(n.__descriptors__)
        else:
            for resource in d_resources:
                if resource is not None:
                    if isinstance(resource, list) and resource and isinstance(resource[0], tuple):
                        # A sparse registry (e.g. ImagingLibrary, whose slots
                        # can be freed/reused - see ResourceHandle) can't
                        # always bind its occupied slots as one contiguous
                        # array from 0: it's a list of (resource, index)
                        # pairs instead, one explicit-slot bind() call each.
                        for item in resource:
                            descriptor_set.bind(**{f'b{i + 1}': item})
                    else:
                        descriptor_set.bind(**{f'b{i + 1}': resource})
                    has_dynamic_resources = True
                i += 1

    # dims_view/main_node_view are raw memoryviews straight over the
    # dispatcher uniform's own host-mapped bytes (see _build_pipeline) -
    # plain indexed assignment, no buffer.write(field, value) call at all.
    dims_view[0] = threads_x
    dims_view[1] = threads_y
    dims_view[2] = threads_z
    dims_view[3] = 0
    dims_view[4] = 0
    dims_view[5] = 0
    main_node_view[0] = main_node.__accessor__.rdv_device_ptr
    # dispatcher_buff.write(
    #     rdv_dim_x=threads_x,
    #     rdv_dim_y=threads_y,
    #     rdv_dim_z=threads_z,
    #     rdv_start_x=0,
    #     rdv_start_y=0,
    #     rdv_start_z=0,
    #     rdv_main_node=main_node.__accessor__.rdv_device_ptr(),
    # )

    command_key = (*pipeline_key, threads_x, threads_y, threads_z)
    # A node with truly dynamic resources (e.g. ImagingLibrary's image/
    # sampler arrays, whose bound contents can grow/change from one
    # dispatch to the next - see GPUSampler) can't share a cached, already-
    # recorded command buffer across calls on a device that doesn't support
    # UPDATE_AFTER_BIND: resubmitting a previously closed command buffer
    # after descriptor_set.bind() adds new array elements does NOT pick up
    # those new elements on such a driver (only a freshly recorded
    # vkCmdBindDescriptorSets does), even after waiting for the prior
    # submission to fully complete. Pipeline.close() already opts every
    # binding into UPDATE_AFTER_BIND wherever the device supports it, so
    # this only forces a re-record as a fallback where it doesn't.
    device_idx = main_node.device_index
    can_update_after_bind = __update_after_bind_caps__.get(device_idx)
    if can_update_after_bind is None:
        with _vk.device(device_idx):
            can_update_after_bind = _vk.caps().can_update_after_bind
        __update_after_bind_caps__[device_idx] = can_update_after_bind
    needs_rerecord = has_dynamic_resources and not can_update_after_bind
    if needs_rerecord or command_key not in __dispatch_commands__:
        # Command buffers are allocated from a pool tied to whatever engine
        # is current at creation time - must be the same explicit
        # EngineType.COMPUTE engine used at submit time below, or an
        # intervening call that lazily activates some other default engine
        # (e.g. a plain buffer/tensor allocation) can leave this command
        # buffer's pool on a different queue family than the one it's later
        # submitted to (a validation-layer-catchable Vulkan error).
        with _vk.device(main_node.device_index):
            with _vk.engine(EngineType.COMPUTE):
                command_buffer = _vk.command_buffer()
                command_buffer.set_pipeline(pipeline)
                command_buffer.bind(0, [descriptor_set])
                command_buffer.dispatch_threads(threads_x, threads_y, threads_z)
                command_buffer.close()
        if not needs_rerecord:
            __dispatch_commands__[command_key] = command_buffer
    else:
        command_buffer = __dispatch_commands__[command_key]

    if __PROFILING__:
        start_t = _time.perf_counter()
    with _vk.device(main_node.device_index):
        with _vk.engine(EngineType.COMPUTE):
            _vk.submit_and_wait(command_buffer)
            # _vk.submit([command_buffer]).wait()
    if __PROFILING__:
        start_t = _time.perf_counter() - start_t
        global __submit_times__
        __submit_times__ += start_t





__submit_times__ = 0


def reset_submit_times():
    global __submit_times__
    __submit_times__ = 0


def profiling_submit_times():
    return __submit_times__




# # ===== DEFERRED POINTERS =====
#
# class _DeferredPtrBase:
#     """
#     Defines the base class for pointers that will be known in a future.
#     It refers to it as a name and an offset in bytes to be added automatically.
#     """
#
#     __name_indexer__ = _Indexer()
#
#     __idx_buffer__ = None
#
#     @classmethod
#     def buffer(cls):
#         if cls.__idx_buffer__ is None:
#             cls.__idx_buffer__ = _vk.buffer(16*1024, Type.UINT64, location=_vk.MemoryLocation.HOST)
#         return cls.__idx_buffer__
#
#     @classmethod
#     def bind_deferreds(cls, **deferreds):
#         b = cls.buffer()
#         for k, v in deferreds.items():
#             assert cls.__name_indexer__.contains(k), f"Deferred name {k} was not declared before."
#             idx = cls.__name_indexer__.idx(k)
#             if hasattr(v, "__object__"):
#                 v = v.__object__
#             if hasattr(v, "device_ptr"):
#                 v = v.device_ptr
#             assert isinstance(v, int), f"Could not be resolved a ptr from the bound object {v} to the deferred {k}"
#             b.element(idx).write(b.element_layout, v)
#
#     @classmethod
#     def unbind_deferreds(cls, **deferreds):
#         b = cls.buffer()
#         for k, v in deferreds.items():
#             assert cls.__name_indexer__.contains(k), f"Deferred name {k} was not declared before."
#             idx = cls.__name_indexer__.idx(k)
#             b.element(idx).write(b.element_layout, 0)
#
#     def __init__(self, name: str, offset: int = 0):
#         self._name = name
#         self._offset = offset
#         self._device_index = _vk.device_index()
#         self._deferred_id = _DeferredPtrBase.__name_indexer__.idx(name)
#         assert offset >= 0, "Offset must be non-negative."
#         assert offset < (1 << 47), "Overflow in the supported offset"
#         assert self._deferred_id < (1 << 15), "Overflow in the supported number of different deferred pointer names."
#
#     @cached_property
#     def coded_value(self) -> int:
#         """
#         This is the real value set to the pointer field. It contains coded
#         the id for the given name, the offset and a mark in the least significant
#         bit 1. That will be used to distinguish from a direct pointer because
#         pointers are aligned to at least 4 bytes.
#         """
#         return self._deferred_id << 48 | self._offset << 1 | 1
#
#
# class DeferredPtr(_DeferredPtrBase):
#     """
#     Defines a ptr to be known in the future without any further constraint.
#     """
#     pass
#
#
# class DeferredTensor(_DeferredPtrBase):
#     """
#     Defines a ptr of a tensor to be known in the future.
#     The bound object is constrained to be a tensor with specific shape and type.
#     """
#     def __init__(self, name: str, *shape: int, type: Type = Type.FLOAT32, element_index: int = 0):
#         bytes = _vk.type_size(type) * _math.prod(shape)
#         super().__init__(name, element_index * bytes)
#         self.dimension = len(shape)
#         self.shape = tuple(shape)
#         self.type = type
#
#
# class DeferredComputeNode(_DeferredPtrBase):
#     """
#     Defines a ptr of a compute node to be known in the future.
#     The bound object is constrained to be a compute node with specific signature.
#     """
#     def __init__(self, name: str, type_or_prototype: '_ComputeNodeMeta | ComputeNode'):
#         super().__init__(name, 0)
#         assert hasattr(type_or_prototype.signature), "Can not construct a deferred compute node from a generic type"
#         self.type_or_prototype = type_or_prototype
