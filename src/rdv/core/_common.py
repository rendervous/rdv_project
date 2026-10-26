from ._core import *
from ._core import _Indexer, _layout_of
import vk as _vk
import typing as _typing
import random as _random


T = Generics.T
N = Generics.N


_NUMPY_DTYPE_NAMES = {
    Type.BOOL: 'bool_', Type.INT8: 'int8', Type.UINT8: 'uint8', Type.INT16: 'int16', Type.UINT16: 'uint16',
    Type.INT32: 'int32', Type.UINT32: 'uint32', Type.INT64: 'int64', Type.UINT64: 'uint64',
    Type.FLOAT16: 'float16', Type.FLOAT32: 'float32', Type.FLOAT64: 'float64',
}
"""Maps a raw rdv/vk Type to the numpy-style dtype attribute name every
supported backend (torch/numpy/cupy/jax) exposes under that same name."""


def _reinterpret(raw, backend: str, scalar_type: Type):
    """
    Bit-reinterprets raw (an already backend-native array/tensor, e.g. from
    RDVObject.as_tensor) as scalar_type - same underlying bytes, no value
    conversion - the same operation torch/numpy/cupy/jax call `.view(dtype)`
    and tensorflow calls `tf.bitcast`.
    """
    name = _NUMPY_DTYPE_NAMES[scalar_type]
    if backend == 'tensorflow':
        import tensorflow as tf
        return tf.bitcast(raw, getattr(tf, name))
    if backend == 'jax':
        import jax.numpy as jnp
        return raw.view(getattr(jnp, name))
    import importlib
    module = importlib.import_module(backend)  # torch, numpy or cupy - all expose e.g. .float32 the same way
    return raw.view(getattr(module, name))


def _reshape(raw, backend: str, shape: list[int]):
    """Reshapes raw (an already backend-native array/tensor) to shape, uniformly across backends."""
    if backend == 'tensorflow':
        import tensorflow as tf
        return tf.reshape(raw, shape)
    return raw.reshape(shape)


# =================== COMMON LIBRARIES ===================


class MathLibrary(Library):
    __abstract__ = True
    __code__ = 'shaders/libraries/math_library.h'


class RandomLibrary(Library):
    __abstract__ = True
    __code__ = 'shaders/libraries/random_library.h'
    __requires__ = [
        MathLibrary
    ]

    __descriptors__ = [ DescriptorType.UNIFORM_BUFFER ]
    __resources__ = {    }

    @classmethod
    def static_resources(cls):
        di = device_index()
        if di not in cls.__resources__:
            cls.__resources__[di] = [_vk.buffer(dict(seeds=Type.UVEC4), location=MemoryLocation.HOST)]
        return cls.__resources__[di]

    @classmethod
    def dynamic_resources(cls):
        _vk.noise_int(cls.static_resources()[0], 128, 1<<30)
        # buf = cls.static_resources()[0].memoryview().cast('i')
        # buf[0] = _random.randint(128, 1 << 30)
        # buf[1] = _random.randint(128, 1 << 30)
        # buf[2] = _random.randint(128, 1 << 30)
        # buf[3] = _random.randint(128, 1 << 30)
        # One entry per __descriptors__ slot (here just the one seeds
        # buffer) is required even when nothing needs rebinding - dispatch()
        # counts entries positionally to line up binding slots across every
        # node in dependency order, and an empty list here (unlike a [None])
        # would silently desync every later node's own bindings by one.
        # The buffer's own descriptor binding never changes (bound once via
        # static_resources() at pipeline-build time) - only its content,
        # refreshed in place above - so None (skip rebinding) is correct.
        return None


class CooperativeLibrary(Library):
    __abstract__ = True
    __code__ = 'shaders/libraries/cooperative_library.h'


class GridLibrary(Library, Generic[N]):
    __abstract__ = True
    __code__ = "shaders/libraries/grid_library.h"  # grid_library.h already defines RDV_NODE; suppress Library's default no-op body


class DeferredLibrary(Library):
    __descriptors__ = [
        DescriptorType.STORAGE_BUFFER
    ]
    __code__ = "shaders/libraries/deferred_library.h"
    __resources__ = {}
    __views__ = {}

    __names__ = _Indexer()

    @classmethod
    def encode_deferred(cls, name: str, offset: int | None = None):
        if offset is None:
            offset = 0
        idx = cls.__names__.idx(name)
        return idx << 48 | offset << 1 | 1

    @classmethod
    def _pointers_view(cls):
        """
        A writable memoryview (format 'Q' - one uint64 per deferred-name
        slot) directly over resources()'s host-mapped buffer bytes, via
        Buffer.memoryview() - bind()/unbind() write a raw pointer with a
        plain indexed assignment (view[idx] = ptr), no per-call vk API call
        (Buffer.write, a LayoutField, a cast()...), no torch/numpy detour,
        at all. Buffer.memoryview() itself comes back byte-format ('B'), so
        it's cast once, here, to 'Q' (memoryview.cast(...) only ever works
        from/to a byte format).
        """
        di = device_index()
        if di not in cls.__views__:
            cls.__views__[di] = cls.static_resources()[0].memoryview().cast('Q')
        return cls.__views__[di]

    @classmethod
    def bind(cls, **deferreds):
        view = cls._pointers_view()
        for k, v in deferreds.items():
            view[cls.__names__.idx(k)] = resolve_ptr(v)

    @classmethod
    def unbind(cls, **deferreds):
        view = cls._pointers_view()
        for k in deferreds:
            view[cls.__names__.idx(k)] = 0

    @classmethod
    def static_resources(cls):
        di = device_index()
        if di not in cls.__resources__:
            with _vk.device(di):
                cls.__resources__[di] = [
                    _vk.buffer(1 << 10, Type.UINT64, location=MemoryLocation.HOST),
                ]
        return cls.__resources__[di]

    @classmethod
    def dynamic_resources(cls):
        # One entry per __descriptors__ slot (here just the one pointers
        # buffer) is required even though nothing needs rebinding here - see
        # RandomLibrary.dynamic_resources for why an empty list (rather than
        # [None]) silently desyncs every later node's own binding slots.
        # The buffer itself is bound once via static_resources(); only its
        # content (individual pointer values) changes, via bind()/unbind()'s
        # direct memoryview writes above.
        return None



# =============== POINTERS =======================
# Classes used to represent reference values that can be directly accessed or as a buffer.


class PointerBase(Node, RDVType):
    __abstract__ = True
    _value = None
    _obj = None

    def __init__(self, obj):
        if obj is None:
            self._value = 0
            self._obj = None
        else:
            assert hasattr(obj, "device_ptr")
            self._value = obj.device_ptr
            self._obj = obj

    @classmethod
    def _fields_layout_decription(cls):
        return Type.UINT64

    def _fields_value(self):
        return self._value

    @classmethod
    def _custom_definitions(cls, bases, dct):
        return


class Pointer(PointerBase, Generic[T]):
    @classmethod
    def _generate_code(cls):
        return f"""
layout(buffer_reference, scalar) buffer RDV_CONCAT(RDV_NODE,_ref) {{ T data; }};
struct RDV_NODE {{ RDV_CONCAT(RDV_NODE,_ref) rdv_ref; }};
"""


class Buffer(PointerBase, Generic[T]):
    @classmethod
    def _generate_code(cls):
        return f"""
layout(buffer_reference, scalar) buffer RDV_CONCAT(RDV_NODE,_ref) {{ T data[]; }};
struct RDV_NODE {{ RDV_CONCAT(RDV_NODE,_ref) rdv_ref; }};
"""


class FixArray(PointerBase, Generic[T, N]):
    @classmethod
    def _generate_code(cls):
        return f"""
layout(buffer_reference, scalar) buffer RDV_CONCAT(RDV_NODE,_ref) {{ T data[N]; }};
struct RDV_NODE {{ RDV_CONCAT(RDV_NODE,_ref) rdv_ref; }};
"""


# ============== DEFERREDS ============================
# Classes used to represent pointers that will be bound later relative to a name.




class DeferredBase(Node, RDVType):
    __abstract__ = True
    __requires__ = [
        DeferredLibrary
    ]

    _ptr_or_code = None
    """
    Pointer of the object or coded deferred info.
    """
    _obj = None

    @classmethod
    def _fields_layout_decription(cls):
        return Type.UINT64

    def __init__(self, name: str | None = None, offset: int | None = None, obj: _typing.Any = None):
        assert name is not None or (obj is not None and offset is not None), "Either name or obj must be provided. Offset is valid only if deferred."
        if obj is not None:
            self._ptr_or_code = resolve_ptr(obj)
            self._obj = obj
        else:
            assert name is not None
            self._ptr_or_code = DeferredLibrary.encode_deferred(name, offset)

    def _fields_value(self):
        return self._ptr_or_code


class Deferred(DeferredBase, Generic[T]):
    __aliases__ = {
        Generics.USE_RAW_POINTER: T == Type.UINT64,
    }

    @classmethod
    def _generate_code(cls):
        return """
struct RDV_NODE { uint64_t ptr_or_code; };
T resolve(RDV_NODE obj) {
    // T's own struct (see Pointer/Buffer) wraps a buffer_reference field
    // (T_ref) - it has no implicit constructor from a raw uint64_t, only
    // T_ref itself does, so the ref has to be cast first.
#if USE_RAW_POINTER==1
    return rdv_resolve_ptr(obj.ptr_or_code);
#else
    return cast(T, rdv_resolve_ptr(obj.ptr_or_code));
#endif
}
"""


class _DeferredContext:
    def __init__(self, **deferreds):
        self.__deferreds = deferreds

    def __enter__(self):
        DeferredLibrary.bind(**self.__deferreds)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        DeferredLibrary.unbind(**self.__deferreds)
        # # whatever ran inside this `with` block (typically dispatch(...),
        # # already synchronously waited for by the time __exit__ runs) may
        # # have just written into one of these objects' own memory *through*
        # # the deferred pointer - i.e. behind its own wrapper's back, which
        # # has no other way to know its GPU-side copy is now the freshest
        # # one. Without this, a subsequent update_host()/as_type() on a
        # # non-direct wrap (a staging buffer stands in for the GPU side)
        # # would stay a no-op and silently keep returning stale host data.
        # for v in self.__deferreds.values():
        #     if hasattr(v, 'make_device_dirty'):
        #         v.make_device_dirty()
        #     elif hasattr(v, 'make_gpu_dirty'):
        #         v.make_gpu_dirty()


def deferreds(**deferreds):
    return _DeferredContext(**deferreds)


_BACKEND_MODULES = {
    'torch': 'torch',
    'numpy': 'numpy',
    'cupy': 'cupy',
    'jax': 'jax',
    'jaxlib': 'jax',
    'tensorflow': 'tensorflow',
}
"""Maps a DLPack-compatible object's own type's root module name to the
backend string as_tensor()/as_type() expect - e.g. a jax Array's concrete
type usually lives under jaxlib, not jax itself."""

_CPU_ONLY_BACKENDS = {'numpy'}
"""Backends whose own array type can't represent DEVICE-resident (e.g.
CUDA) memory at all - from_dlpack on one of these for a DEVICE tensor
fails outright (numpy: "Unsupported device in DLTensor"), regardless of
whether the *source* data ensure_on_rdv() was given lived on the host."""


def _detect_backend(data) -> str:
    """The backend name ('torch'/'numpy'/'cupy'/'jax'/'tensorflow') for
    data's own type, inferred from its defining module's root package."""
    root_module = type(data).__module__.split('.')[0]
    backend = _BACKEND_MODULES.get(root_module)
    if backend is None:
        raise TypeError(
            f"ensure_on_rdv: could not infer a supported backend from {type(data)!r} "
            f"(module {root_module!r}) - pass backend explicitly.")
    return backend


_DLPACK_CPU_DEVICE_TYPE = 1  # kDLCPU


def _infer_location(data) -> MemoryLocation:
    """HOST if data's own DLPack device is CPU, DEVICE otherwise (e.g.
    CUDA) - ensure_on_rdv()'s default `location` when not given explicitly,
    so a CPU-only backend (e.g. numpy - from_dlpack on most CPU-only
    backends can't import CUDA-resident memory at all) round-trips
    correctly, while a GPU-resident input still gets a GPU-resident copy."""
    if hasattr(data, '__dlpack_device__'):
        device_type, _ = data.__dlpack_device__()
        if device_type == _DLPACK_CPU_DEVICE_TYPE:
            return MemoryLocation.HOST
    return MemoryLocation.DEVICE


def is_on_rdv(data) -> bool:
    """
    Whether data's own memory is already guaranteed to be genuine
    Vulkan-managed memory - not merely GPU-resident. True for an
    RDVObject/vk.Buffer/vk.Tensor directly, or for any DLPack-compatible
    CUDA tensor (e.g. a torch tensor) whose own pointer already resolves
    to one of vk's own allocations - e.g. it was itself produced by
    .torch()/.numpy()/... on a vk buffer/tensor, or by ensure_on_rdv()
    below. False for anything else (a plain CPU-only tensor/array, an
    ordinary CUDA tensor from some other allocator, ...).

    Delegates to vk.is_own_memory for everything except RDVObject itself,
    whose own is_direct already answers this exact question without
    needing a fresh native-side lookup.
    """
    if isinstance(data, RDVObject):
        return data.is_direct
    return _vk.is_own_memory(data)


def ensure_on_rdv(data, backend: str | None = None, location: MemoryLocation | None = None):
    """
    Guarantees `data`'s own memory is genuine Vulkan-managed memory -
    returns `data` completely unchanged if `is_on_rdv(data)` already (no
    redundant copy/allocation at all), otherwise copies it into a fresh vk
    (Vulkan) tensor and exports that back as the same backend `data`
    itself already is (auto-detected from its type unless `backend` is
    given explicitly).

    e.g. ``t = ensure_on_rdv(torch.rand(3, device='cuda'))``: `t` is
    itself a `torch.Tensor`; if the input wasn't already vk-owned, its own
    memory (`t.data_ptr()`) is now a fresh Vulkan-managed allocation
    (CUDA-visible via interop - see `_infer_location`, `location`'s own
    default), not a view into whatever allocator `data` originally lived
    in (e.g. torch's own caching allocator).

    Unlike plain `wrap(data)`, whose own `device_ptr` *aliases* `data`'s
    existing memory directly whenever it's already GPU-resident (see
    `WrappedMemory.is_direct`) rather than copying it, a non-vk-owned
    `data` always gets a real copy into brand new Vulkan-owned memory here
    - the guarantee this function's name promises.
    """
    if is_on_rdv(data):
        return data
    backend_given = backend is not None
    if backend is None:
        backend = _detect_backend(data)
    if location is None:
        location = _infer_location(data)
    if location == MemoryLocation.DEVICE and backend in _CPU_ONLY_BACKENDS:
        # location says where the *backend* tensor this function creates
        # should live, independent of wherever `data` (the source to copy
        # from) itself happens to be - a CPU-only backend auto-detected
        # from a host-resident `data` (e.g. a plain numpy array) can't
        # honor an explicit location=DEVICE request at all (it would only
        # have worked by accident, via _infer_location's own HOST default,
        # had location been left unspecified) - fall back to a
        # device-capable backend instead of failing deep inside the final
        # DLPack export with an opaque "Unsupported device in DLTensor".
        # Only when backend was auto-detected, not explicitly requested -
        # an explicit backend='numpy' + location=DEVICE is a genuine
        # contradiction the caller should hear about directly.
        if not backend_given:
            backend = 'torch'
        else:
            raise ValueError(
                f"ensure_on_rdv: backend={backend!r} can't represent DEVICE-resident "
                f"memory - pass a device-capable backend (e.g. 'torch') explicitly, "
                f"or leave location unspecified/HOST.")
    probe = wrap(data)
    # data.shape (every supported backend exposes one directly) rather
    # than probe.shape: wrap()'s own buffer-protocol fallback path (taken
    # e.g. for a numpy array that refuses to re-export via DLPack at all -
    # see is_on_rdv/Device::wrap's own fallback) always reports a
    # flattened 1-D shape, elements-only, not data's real one.
    shape = list(data.shape) if hasattr(data, 'shape') else list(probe.shape)
    vk_tensor = _vk.tensor(shape, probe.scalar_type, location=location)
    vk_tensor.load(data)
    return getattr(vk_tensor, backend)()


# def buffer(elements: int, element_description: dict | Type, location: MemoryLocation | None = None) -> Buffer:
#     if location is None:
#         location = _vk.MemoryLocation.HOST
#     return Data(_vk.buffer(elements, _vk.compute_layout(element_description, _vk.LayoutRule.Scalar), location))
#



