"""
Main.export_webgpu: turns a concrete Main instance into one self-contained
JavaScript (ES module) file - the WGSL megakernel (see _webgpu.py for the
GLSL -> WGSL toolchain) plus the WebGPU machinery to evaluate it.

Memory model. The lowered megakernel addresses all node/data memory through a
single `heap` storage buffer (u32 byte offsets instead of device pointers -
see _spirv_lower.py). At export time this module snapshots everything
reachable from the main node - every node's own field buffer and every data
object (tensor/buffer) a pointer leads to - packs it into that heap (offset 0
is never used, so a null pointer stays 0) and relocates every pointer slot,
found by walking the nodes' own accessors (so it's known exactly which
8-byte fields are pointers, never guessed from values).

Named deferreds (Deferred(name=...), e.g. a Tensor over Deferred data) are
what the exported evaluate() binds: each named buffer passed to it is copied
into the heap past the static part, every deferred code found in node memory
is re-encoded as an index into a small table of (name, offset) entries, and
that table is filled with the resolved heap addresses on every call - the
same role DeferredLibrary.bind() plays on the Vulkan side.
Leaf scalar/vector fields of every reachable node are also exposed as named
values (dotted paths from the main node, e.g. 'a' or 'image.c'), writable
from evaluate() the same way.
"""

import base64 as _base64
import json as _json
import os as _os
import re as _re
import struct as _struct

import vk as _vk

from ._core import (Accessor, Code, Main, RDVType, Type, _NodeMeta, _HLSL_DIR, _binding_slot_assignment)
from ._webgpu import glsl_to_wgsl, WebGPUExportError

_HEAP_ALIGN = 16
_DEFERRED_MASK = (1 << 48) - 1
_JS_VALUE_TYPES = {Type.FLOAT32: 'f32', Type.INT32: 'i32', Type.UINT32: 'u32'}
# push block layout, as declared by _build_code's rdv_dispatcher (scalar):
# rdv_main_node (u64) followed by rdv_dim_xyz/rdv_start_xyz (i32 each).
_PUSH_SIZE = 32


def _object_bytes(obj) -> bytes:
    """Current contents of a data object (RDVObject, vk.Buffer/vk.Tensor)."""
    if isinstance(obj, (_vk.Buffer,)):
        return bytes(obj.memoryview())
    from ._core import RDVObject
    if isinstance(obj, RDVObject):
        try:
            obj.update_host()
            array = obj.as_tensor('numpy')
        except Exception:
            # device-only memory has no host-visible pointer to export from
            array = obj.host().as_tensor('numpy')
        import numpy as np
        return np.ascontiguousarray(array).view(np.uint8).tobytes()
    if isinstance(obj, _vk.Tensor):
        import numpy as np
        return np.ascontiguousarray(obj.numpy()).view(np.uint8).tobytes()
    raise WebGPUExportError(f'cannot snapshot the memory of {type(obj).__name__}')


class _Snapshot:
    def __init__(self, main_node):
        self.regions = {}        # device ptr -> bytes (current contents)
        self.labels = {}         # device ptr -> field path it was first reached from
        self.pointer_slots = []  # (region ptr, byte offset, path)
        self.u64_slots = []      # (region ptr, byte offset, path) - raw u64 leaves, maybe pointers
        self.values = {}         # path -> (region ptr, byte offset, js type, count)
        self.visited_nodes = set()
        self._visit_node(main_node, '')
        self.main_ptr = main_node.__accessor__.rdv_device_ptr

    def _add_region(self, ptr, data, path):
        if ptr == 0:
            return
        if len(data) > len(self.regions.get(ptr, b'')):
            self.regions[ptr] = data
        self.labels.setdefault(ptr, path or '<main>')

    def _visit_node(self, node, path):
        if id(node) in self.visited_nodes:
            return
        self.visited_nodes.add(id(node))
        acc = node.__accessor__
        if acc._buff is None:
            raise WebGPUExportError(f'{path or "main node"} ({type(node).__name__}) has no concrete layout '
                                    f'(still generic or abstract).')
        ptr = acc.rdv_device_ptr
        self._walk_accessor(acc, ptr, path)
        self._add_region(ptr, bytes(acc._buff.memoryview()), path)

    def _walk_accessor(self, acc, region_ptr, path):
        fields = acc._fields
        keys = list(fields.keys()) if isinstance(fields, dict) else range(len(acc._values))
        for key in keys:
            field_type = fields[key] if isinstance(fields, dict) else fields[1]
            value = acc._values[key]
            sub_path = (f'{path}.{key}' if path else str(key)) if isinstance(fields, dict) else f'{path}[{key}]'
            if isinstance(value, Accessor):
                self._walk_accessor(value, region_ptr, sub_path)
                continue
            if value is None:
                continue
            handle = acc._field_handle(key)
            offset = acc._base_offset + handle.offset
            if isinstance(field_type, _NodeMeta) or isinstance(value, RDVType):
                self.pointer_slots.append((region_ptr, offset, sub_path))
                self._visit_target(value, sub_path)
                continue
            component_type = handle.layout.component_type
            if component_type in (Type.UINT64, Type.INT64):
                self.u64_slots.append((region_ptr, offset, sub_path))
            elif component_type in _JS_VALUE_TYPES:
                count = _vk.type_size(handle.layout.type) // _vk.type_size(component_type)
                self.values.setdefault(sub_path, (region_ptr, offset, _JS_VALUE_TYPES[component_type], count))

    def _visit_target(self, value, path):
        from ._common import PointerBase, DeferredBase
        from ._core import RDVObject
        if value is None or isinstance(value, int):
            return  # null, or a raw address (relocated by range, see _relocate)
        if isinstance(value, Code):
            self._visit_node(value, path)
        elif isinstance(value, PointerBase):
            self._visit_target(value._obj, path)
        elif isinstance(value, DeferredBase):
            self._visit_target(value._obj, path)  # None for a still-named (deferred) one
        elif isinstance(value, (RDVObject, _vk.Buffer, _vk.Tensor)):
            self._add_region(value.device_ptr, _object_bytes(value), path)
        else:
            raise WebGPUExportError(f'{path}: cannot export a reference to {type(value).__name__}')


class _HeapLayout:
    """Packs snapshot regions (merging any that alias the same memory) into one heap."""

    def __init__(self, snapshot: _Snapshot):
        spans = sorted((ptr, ptr + len(data)) for ptr, data in snapshot.regions.items())
        self.blocks = []  # [start ptr, end ptr, heap offset]
        for start, end in spans:
            if self.blocks and start < self.blocks[-1][1]:
                self.blocks[-1][1] = max(self.blocks[-1][1], end)
            else:
                self.blocks.append([start, end, 0])
        cursor = _HEAP_ALIGN  # heap offset 0 is never handed out: it stays "null"
        for block in self.blocks:
            block[2] = cursor
            cursor = _align(cursor + block[1] - block[0], _HEAP_ALIGN)
        if cursor >= 1 << 32:
            raise WebGPUExportError(f'the exported heap would be {cursor} bytes; addresses are limited to 32 bits.')
        self.size = cursor
        self.heap = bytearray(cursor)
        for ptr, data in snapshot.regions.items():
            at = self.heap_offset(ptr)
            self.heap[at:at + len(data)] = data

    def heap_offset(self, ptr):
        for start, end, offset in self.blocks:
            if start <= ptr < end or (ptr == end and start != end):
                return offset + ptr - start
        return None


def _align(n, a):
    return (n + a - 1) // a * a


def _relocate(snapshot: _Snapshot, layout: _HeapLayout):
    """Rewrites every pointer slot in the packed heap; returns the deferred entries."""
    from ._common import DeferredLibrary
    entries = {}  # (name, offset) -> entry index

    def rewrite(region_ptr, offset, path, strict):
        at = layout.heap_offset(region_ptr) + offset
        value, = _struct.unpack_from('<Q', layout.heap, at)
        if value == 0:
            return
        if value & 1:
            name = DeferredLibrary.__names__.signature(value >> 48)
            key = (name, (value & _DEFERRED_MASK) >> 1)
            new_value = (entries.setdefault(key, len(entries)) << 1) | 1
        else:
            new_value = layout.heap_offset(value)
            if new_value is None:
                if not strict:
                    return  # a plain u64 value that just isn't an address
                raise WebGPUExportError(f'{path}: points to memory ({value:#x}) not reachable from the exported '
                                        f'node graph.')
        _struct.pack_into('<Q', layout.heap, at, new_value)

    for region_ptr, offset, path in snapshot.pointer_slots:
        rewrite(region_ptr, offset, path, strict=True)
    for region_ptr, offset, path in snapshot.u64_slots:
        rewrite(region_ptr, offset, path, strict=False)
    return [dict(name=name, offset=offset) for (name, offset), _ in sorted(entries.items(), key=lambda e: e[1])]


_TEXTURE_FORMATS = {1: 'r32float', 2: 'rg32float', 4: 'rgba32float'}  # 3 channels are stored padded to 4


def _single_image_binding(node_type, b):
    """Texture (slot 0) or sampler (slot 1) of a SingleImageLibrary[ID] - the
    only image resources the WebGPU export supports (one plain binding each;
    ImagingLibrary's bindless arrays have no WebGPU equivalent)."""
    from ..maths import SingleImageLibrary
    if not issubclass(node_type, SingleImageLibrary):
        raise WebGPUExportError(f'{node_type.__name__}: only SingleImageLibrary (SingleGPUSampler/SingleGPUGrid) images '
                                f'can be exported to WebGPU.')
    bound = node_type.bound()
    if bound is None:
        raise WebGPUExportError(f'{node_type.__name__}: nothing is bound to this ID yet.')
    if b['kind'] == 'sampler':
        if not bound['clamp']:
            raise WebGPUExportError(f'{node_type.__name__}: clamp=False (clamp-to-border) has no WebGPU sampler '
                                    f'equivalent - use clamp=True.')
        return dict(binding=b['binding'], kind='sampler', filter='linear' if bound['linear'] else 'nearest',
                    label=f'{node_type.__name__} sampler')
    shape = bound['shape']  # spatial, slowest first
    size = (list(reversed(shape)) + [1, 1])[:3]
    channels = 4 if bound['channels'] == 3 else bound['channels']
    data = _object_bytes(bound['source'])
    return dict(binding=b['binding'], kind='texture', dimension=f'{len(shape)}d', size=size,
                format=_TEXTURE_FORMATS[channels], bytes_per_texel=4 * channels,
                data=_base64.b64encode(data).decode('ascii'),
                label=f"{node_type.__name__} image, {'x'.join(map(str, size[:len(shape)]))} {_TEXTURE_FORMATS[channels]}")


def _descriptor_bindings(cls, lowered_bindings, deferred_entries):
    """Exported metadata for every buffer binding the lowered shader declares."""
    from ._common import DeferredLibrary, RandomLibrary
    slots, _ = _binding_slot_assignment(cls.dependency_order)
    owner = {}
    for node_type, start in slots.items():
        for i in range(len(node_type.__descriptors__)):
            owner[start + i] = (node_type, i)
    result = []
    for b in lowered_bindings:
        if b['kind'] in ('heap', 'heap_static', 'heap_dynamic'):
            result.append(dict(binding=b['binding'], kind=b['kind']))
        elif b['kind'] == 'push':
            result.append(dict(binding=b['binding'], kind='push', size=_PUSH_SIZE))
        elif b['kind'] in ('texture', 'sampler'):
            node_type, index = owner[b['binding']]
            result.append(_single_image_binding(node_type, b))
        else:
            node_type, index = owner[b['binding']]
            if issubclass(node_type, DeferredLibrary):
                result.append(dict(binding=b['binding'], kind='deferred_table', size=max(len(deferred_entries), 1) * 8))
            elif issubclass(node_type, RandomLibrary):
                result.append(dict(binding=b['binding'], kind='random_seeds', size=16))
            else:
                resource = node_type.static_resources()[index]
                if not isinstance(resource, _vk.Buffer):
                    raise WebGPUExportError(f'{node_type.__name__}: descriptor {index} is not a buffer, which the '
                                            f'WebGPU export does not support.')
                data = bytes(resource.memoryview())
                result.append(dict(binding=b['binding'], kind='static', size=len(data),
                                   data=_base64.b64encode(data).decode('ascii'),
                                   label=f'{node_type.__name__} static buffer {index}'))
    return result


def export_webgpu(main_node: Main, path: str | None = None, threads=None, keep_dir: str | None = None,
                  static_writes: bool = False) -> str:
    """
    Exports main_node as one ES module exposing

        evaluate(inputs, { threads, readback, device }) -> Promise<{name: data}>
        createEvaluator(device?) -> Promise<{ evaluate, device, names, values, destroy }>

    where inputs maps deferred names to ArrayBuffers/typed arrays (copied in,
    then read back into the same arrays once the dispatch completes) and
    value paths (see module docstring) to numbers/arrays.

    threads: default dispatch size (an int or a 1-3 tuple), used whenever
    evaluate() isn't given one. keep_dir: keeps every intermediate file of
    the GLSL -> WGSL toolchain there (see glsl_to_wgsl). static_writes:
    whether the kernel writes into memory that isn't bound by name (node
    fields, static tensors) or reads what other threads wrote during the
    same dispatch - by default loads read a read-only heap and stores a
    read-write mirror (much faster loads, see _spirv_lower), a static
    write is reported as an error by evaluate(); True keeps one
    read-write heap.
    Returns the JS source; also writes it to path if given.
    """
    cls = type(main_node)
    if cls.contains_compute_dependencies:
        from ._core import _collect_compute_dependencies
        deps = []
        _collect_compute_dependencies(main_node, set(), deps)
        if deps:
            raise WebGPUExportError('nodes with compute dependencies (barriers) are not supported by the WebGPU '
                                    'export yet.')
    order = cls.dependency_order
    code = cls.megakernel_code(webgpu=True)
    include_dirs = [_HLSL_DIR]
    include_dirs += sorted({d for n in order if (d := getattr(n, '__include_dir__', None))})
    include_dirs += sorted({d for base in cls.__mro__ if (d := base.__dict__.get('__include_dir__'))})
    snapshot = _Snapshot(main_node)
    layout = _HeapLayout(snapshot)
    dynamic_base = _align(layout.size, 256)  # named buffers are placed from here on
    wgsl, lowered_bindings, warnings = glsl_to_wgsl(code, include_dirs, keep_dir=keep_dir,
                                                    split_heap=not static_writes, dynamic_base=dynamic_base)
    entries = _relocate(snapshot, layout)
    # Tint drops a binding the WGSL ends up not using (e.g. the sampler of a
    # texture only ever read with textureLoad); 'auto' pipeline layouts
    # reject bind group entries for those, so only export what survived.
    used = {int(n) for n in _re.findall(r'@binding\((\d+)u?\)', wgsl)}
    bindings = [b for b in _descriptor_bindings(cls, lowered_bindings, entries) if b['binding'] in used]
    names = sorted({e['name'] for e in entries})
    values = {}
    for value_path, (region_ptr, offset, js_type, count) in snapshot.values.items():
        if value_path in names:
            raise WebGPUExportError(f'{value_path!r} is both a deferred name and a field path.')
        values[value_path] = dict(offset=layout.heap_offset(region_ptr) + offset, type=js_type, count=count)

    if threads is not None:
        threads = list(threads) if isinstance(threads, (tuple, list)) else [threads]
        threads += [1] * (3 - len(threads))
    local_size = list(getattr(cls, '__local_size__', (32, 1, 1)))
    # the base64 contents go to BOUND_DATA at the end of the file, the rest
    # of each binding's description stays in BINDINGS
    bound_data = []
    for b in bindings:
        if 'data' in b:
            size = f" ({b['size']} bytes)" if b['kind'] == 'static' else ''
            bound_data.append(f"  // {b['label']}{size}\n  {b['binding']}: {_json.dumps(b.pop('data'))},")
    # a JS template literal: keep backslashes, backticks and ${ literal
    escaped_wgsl = wgsl.replace('\\', '\\\\').replace('`', '\\`').replace('${', '\\${')
    replacements = {
        '__CLASS__': cls.__name__,
        '__WGSL__': escaped_wgsl,
        '__BOUND_DATA__': '\n'.join(bound_data),
        '__LOCAL_SIZE__': _json.dumps(local_size),
        '__BINDINGS__': ('[\n' + ''.join(f'  {_json.dumps(b)},\n' for b in bindings) + ']') if bindings else '[]',
        '__STATIC_HEAP__': _json.dumps(_base64.b64encode(bytes(layout.heap)).decode('ascii')),
        '__STATIC_HEAP_SIZE__': str(layout.size),
        '__DYNAMIC_BASE__': str(dynamic_base),
        '__MAIN_NODE__': str(layout.heap_offset(snapshot.main_ptr)),
        '__NAMES__': _json.dumps(names),
        '__ENTRIES__': _json.dumps([[names.index(e['name']), e['offset']] for e in entries]),
        '__VALUES__': _json.dumps(values, indent=2),
        '__THREADS__': _json.dumps(threads),
        '__WARNINGS__': _json.dumps(warnings),
    }
    with open(_os.path.join(_os.path.dirname(__file__), 'webgpu_template.js')) as f:
        source = f.read()
    # one pass, so nothing substituted (the WGSL, say) is ever substituted again
    source = _re.sub(r'__[A-Z_]+__', lambda m: replacements.get(m.group(0), m.group(0)), source)
    if path is not None:
        with open(path, 'w') as f:
            f.write(source)
    return source
