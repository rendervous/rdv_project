"""
SPIR-V lowering pass that makes an rdv megakernel consumable by Tint (and so,
expressible in WGSL/WebGPU).

The megakernel is built on GL_EXT_buffer_reference: every Component, Pointer/
Buffer/FixArray, Deferred and the rdv_main_node push constant is a 64-bit
PhysicalStorageBuffer pointer. WebGPU has neither raw pointers nor 64-bit
integers, so this pass rewrites the (disassembled, --raw-id) SPIR-V as follows:

* Every PhysicalStorageBuffer pointer becomes a u32 byte offset into the
  heap. By default (split_heap) the heap is two same-sized storage buffers:
  every load reads the read-only `heap` (group 0 binding 0), every store
  writes the read-write `heap_rw` mirror, and the exported runtime copies
  the run-time (named-buffer) part of heap_rw back into heap after each
  dispatch - so reads see memory as of the start of the dispatch. Loads
  must never touch a read_write buffer: on D3D12 (Dawn on Windows) those go
  through the UAV path, measured 2-4x slower on these kernels, and both a
  per-load branch and a branch-free select between the two buffers (the
  shader compiler flattens the branch into the same thing) were slower
  still than a single read-write heap. A store below dynamic_base (static
  memory: node fields, data) is not performed; it raises a fault flag in
  heap_rw's first word instead, which the exported runtime reports -
  export with static_writes=True (split_heap=False, one read-write heap)
  for kernels that write static memory or read what other threads wrote
  in the same dispatch.
* Every other buffer-backed variable (StorageBuffer/Uniform descriptors, and
  the PushConstant block - WebGPU has no push constants) becomes its own
  u32-word storage buffer, accessed the same way.
* Every load/store through any of those becomes a call to a generated helper
  that reads/writes individual u32 words at byte offsets computed from the
  module's own Offset/ArrayStride/MatrixStride decorations - i.e. memory keeps
  exactly the same (scalar) layout the Vulkan path uses, host side included.
* 64-bit integers are narrowed to 32 bits in registers. In memory they still
  take 8 bytes: loads read the low word only, stores write the low word plus
  a sign/zero-extended high word. Heap addresses are therefore limited to
  32 bits (see the RDV_WEBGPU variant of deferred_library.h for the one place
  that relied on high pointer bits).
* NonSemantic.DebugPrintf calls (PRINT) are dropped.

The pass works on spirv-dis text rather than binary words: an id is always
spelled %name there, so remapping types never confuses an id with a literal.
"""

import re as _re


class SpirvLoweringError(RuntimeError):
    pass


_MEMORY_CLASSES = ('PhysicalStorageBuffer', 'StorageBuffer', 'Uniform', 'PushConstant')

_DROPPED_CAPABILITIES = {'PhysicalStorageBufferAddresses', 'Int64', 'Int64Atomics'}
_DROPPED_EXTENSIONS = {'"SPV_KHR_physical_storage_buffer"', '"SPV_EXT_physical_storage_buffer"',
                       '"SPV_KHR_non_semantic_info"'}
# Layout/aliasing decorations: meaningless once no typed memory access remains
# (every buffer is a plain u32 word array whose own decorations are re-added).
_DROPPED_DECORATIONS = {'Offset', 'ArrayStride', 'MatrixStride', 'Block', 'BufferBlock', 'ColMajor', 'RowMajor',
                        'NonWritable', 'NonReadable', 'Restrict', 'Aliased', 'AliasedPointer', 'RestrictPointer',
                        'Coherent', 'Volatile'}
_NO_RESULT_TYPE = {'OpLabel', 'OpString', 'OpExtInstImport', 'OpDecorationGroup'}
# Non-aggregate declarations that must be unique once 64-bit ints/pointers are
# narrowed (two `OpTypeInt 32 0` are invalid). Pointers/functions aren't
# required unique by the spec but are merged too, so values typed through
# formerly-distinct (u64 vs u32) pointer types stay interchangeable.
_UNIQUE_TYPES = {'OpTypeVoid', 'OpTypeBool', 'OpTypeInt', 'OpTypeFloat', 'OpTypeVector', 'OpTypeMatrix',
                 'OpTypePointer', 'OpTypeFunction'}
_ATOMIC_OPS = {'OpAtomicLoad', 'OpAtomicStore', 'OpAtomicExchange', 'OpAtomicCompareExchange', 'OpAtomicIIncrement',
               'OpAtomicIDecrement', 'OpAtomicIAdd', 'OpAtomicISub', 'OpAtomicSMin', 'OpAtomicUMin', 'OpAtomicSMax',
               'OpAtomicUMax', 'OpAtomicAnd', 'OpAtomicOr', 'OpAtomicXor'}
_SHIFT_OPS = {'OpShiftRightLogical', 'OpShiftRightArithmetic', 'OpShiftLeftLogical'}
_PSB = '<psb>'  # marker "memory" for PSB addresses under split_heap - see helper/split_store


class _Inst:
    __slots__ = ('result', 'op', 'args')

    def __init__(self, result, op, args):
        self.result = result
        self.op = op
        self.args = args

    def text(self):
        body = ' '.join([self.op, *self.args])
        return f'{self.result} = {body}' if self.result else body


def _tokenize(line):
    tokens = []
    i, n = 0, len(line)
    while i < n:
        c = line[i]
        if c.isspace():
            i += 1
        elif c == ';':
            break
        elif c == '"':
            j = i + 1
            while line[j] != '"':
                j += 2 if line[j] == '\\' else 1
            tokens.append(line[i:j + 1])
            i = j + 1
        else:
            j = i
            while j < n and not line[j].isspace():
                j += 1
            tokens.append(line[i:j])
            i = j
    return tokens


def _parse(text):
    insts = []
    for line in text.splitlines():
        tokens = _tokenize(line)
        if not tokens:
            continue
        if len(tokens) > 2 and tokens[1] == '=':
            insts.append(_Inst(tokens[0], tokens[2], tokens[3:]))
        else:
            insts.append(_Inst(None, tokens[0], tokens[1:]))
    return insts


def _is_id(token):
    return token.startswith('%')


def _identifier(name):
    """A WGSL-friendly version of a glslang debug name: function names come
    mangled with their signature ('forward(struct-Code_3-11;f1[3];'), which
    Tint would discard - keep just the name. '' when nothing usable is left."""
    name = _re.sub(r'[^A-Za-z0-9_]', '_', name.split('(')[0])
    name = _re.sub(r'^_+', '_', name)  # '__' prefixes are reserved in WGSL
    if name.strip('_') == '':
        return ''
    return f'_{name}' if name[0].isdigit() else name


class _Lowering:
    """One-shot lowering of a whole module - see the module docstring."""

    def __init__(self, text, heap_binding=0, push_binding=None, split_heap=True, dynamic_base=None):
        self.insts = _parse(text)
        self.split_heap = split_heap
        self.dynamic_base = dynamic_base
        self.heap_binding = heap_binding
        self.push_binding = push_binding
        self.counter = 0
        self.warnings = []
        self.bindings = []  # exported buffer bindings, see lower_spirv()
        self._analyze()

    # ------------------------------------------------------------ analysis

    def _analyze(self):
        self.types = {}            # id -> type declaration _Inst (OpTypePointer wins over its forward decl)
        self.int_consts = {}       # id -> int value
        self.member_offset = {}    # (struct id, member index) -> byte offset
        self.member_mstride = {}   # (struct id, member index) -> MatrixStride
        self.array_stride = {}     # array or pointer type id -> ArrayStride
        self.decorations = {}      # id -> [(decoration, args...)]
        self.global_vars = {}      # id -> (pointer type id, storage class)
        self.orig_type = {}        # SSA/global id -> original result type id
        self.printf_set = None
        self.names = {}            # id -> debug name (OpName)
        for inst in self.insts:
            op, a = inst.op, inst.args
            if op.startswith('OpType') and op != 'OpTypeForwardPointer':
                self.types[inst.result] = inst
            elif op == 'OpExtInstImport' and a[0] == '"NonSemantic.DebugPrintf"':
                self.printf_set = inst.result
            elif op == 'OpName' and _identifier(a[1].strip('"')):
                self.names[a[0]] = _identifier(a[1].strip('"'))
            elif op == 'OpDecorate':
                self.decorations.setdefault(a[0], []).append(a[1:])
                if a[1] == 'ArrayStride':
                    self.array_stride[a[0]] = int(a[2])
            elif op == 'OpMemberDecorate':
                if a[2] == 'Offset':
                    self.member_offset[(a[0], int(a[1]))] = int(a[3])
                elif a[2] == 'MatrixStride':
                    self.member_mstride[(a[0], int(a[1]))] = int(a[3])
                elif a[2] == 'RowMajor':
                    raise SpirvLoweringError('row_major matrices in buffers are not supported by the WebGPU lowering.')
            if inst.result is not None and op not in _NO_RESULT_TYPE and not op.startswith('OpType') and a:
                self.orig_type[inst.result] = a[0]
            if op == 'OpConstant' and self.types.get(a[0]) is not None and self.types[a[0]].op == 'OpTypeInt':
                self.int_consts[inst.result] = int(a[1], 0)
            if op == 'OpVariable' and a[1] in _MEMORY_CLASSES and self._in_globals(inst):
                self.global_vars[inst.result] = (a[0], a[1])

    def _in_globals(self, inst):
        # OpVariable can only have a non-Function storage class at module scope.
        return inst.args[1] != 'Function'

    def is_psb_ptr(self, t):
        d = self.types.get(t)
        return d is not None and d.op == 'OpTypePointer' and d.args[0] == 'PhysicalStorageBuffer'

    def is_mem_ptr_type(self, t):
        d = self.types.get(t)
        return d is not None and d.op == 'OpTypePointer' and d.args[0] in _MEMORY_CLASSES

    def pointee(self, t):
        return self.types[t].args[1]

    def int_info(self, t):
        """(width, signed) for an int (or PSB pointer, as u64) type, else None."""
        d = self.types.get(t)
        if d is None:
            return None
        if d.op == 'OpTypeInt':
            return int(d.args[0]), d.args[1] == '1'
        if self.is_psb_ptr(t):
            return 64, False
        return None

    def scalar_size(self, t):
        d = self.types[t]
        if d.op == 'OpTypeBool':
            return 4
        if d.op in ('OpTypeInt', 'OpTypeFloat'):
            return int(d.args[0]) // 8
        if self.is_psb_ptr(t):
            return 8
        raise SpirvLoweringError(f'not a scalar type: {d.text()}')

    def natural_size(self, t):
        """Scalar-layout size, only used where no explicit stride decoration exists."""
        d = self.types[t]
        if d.op == 'OpTypeVector':
            return self.scalar_size(d.args[0]) * int(d.args[1])
        if d.op == 'OpTypeMatrix':
            return self.natural_size(d.args[0]) * int(d.args[1])
        if d.op == 'OpTypeArray':
            return self.array_stride.get(t, self.natural_size(d.args[0])) * self.int_consts[d.args[1]]
        if d.op == 'OpTypeStruct':
            last = len(d.args) - 1
            return self.member_offset[(t, last)] + self.natural_size(d.args[last])
        return self.scalar_size(t)

    # ------------------------------------------------------------- helpers

    def fresh(self, hint='t'):
        self.counter += 1
        return f'%rdv_{hint}{self.counter}'

    def u32_const(self, value):
        value &= 0xFFFFFFFF
        cid = self.consts.get(value)
        if cid is None:
            cid = f'%rdv_c{value}'
            self.consts[value] = cid
            self.new_globals.append(_Inst(cid, 'OpConstant', [self.u32, str(value)]))
        return cid

    def emit(self, out, op, rtype, *args):
        rid = self.fresh()
        out.append(_Inst(rid, op, [rtype, *args]))
        return rid

    def add_offset(self, out, off, k):
        return off if k == 0 else self.emit(out, 'OpIAdd', self.u32, off, self.u32_const(k))

    def to_u32(self, out, value_id):
        """A u32 view of an integer SSA id (an index operand, typically)."""
        if value_id in self.int_consts:
            return self.u32_const(self.int_consts[value_id])
        info = self.int_info(self.orig_type.get(value_id, ''))
        if info is None:
            raise SpirvLoweringError(f'index {value_id} is not an integer')
        if info[1]:
            return self.emit(out, 'OpBitcast', self.u32, value_id)
        return value_id

    # ------------------------------------------------------------- memory

    def word_ptr(self, out, var, off):
        word = self.emit(out, 'OpShiftRightLogical', self.u32, off, self.u32_const(2))
        return self.emit(out, 'OpAccessChain', self.ptr_sb_u32, var, self.u32_const(0), word)

    def load_word(self, out, var, off):
        return self.emit(out, 'OpLoad', self.u32, self.word_ptr(out, var, off))

    def emit_load(self, out, var, t, off, mstride=None):
        d = self.types[t]
        info = self.int_info(t)
        if info is not None:
            width, signed = info
            if width not in (32, 64):
                raise SpirvLoweringError(f'{width}-bit integers in buffers are not supported by the WebGPU lowering.')
            word = self.load_word(out, var, off)
            return self.emit(out, 'OpBitcast', t, word) if signed else word
        if d.op == 'OpTypeFloat':
            if d.args[0] != '32':
                raise SpirvLoweringError(f'{d.args[0]}-bit floats in buffers are not supported by the WebGPU lowering.')
            word = self.load_word(out, var, off)
            return self.emit(out, 'OpBitcast', t, word)
        if d.op == 'OpTypeBool':
            word = self.load_word(out, var, off)
            return self.emit(out, 'OpINotEqual', t, word, self.u32_const(0))
        parts = [self.emit_load(out, var, et, self.add_offset(out, off, k), ms) for et, k, ms in self.members(t, mstride)]
        return self.emit(out, 'OpCompositeConstruct', t, *parts)

    def emit_store(self, out, var, t, off, value, mstride=None):
        d = self.types[t]
        info = self.int_info(t)
        if info is not None:
            width, signed = info
            if width not in (32, 64):
                raise SpirvLoweringError(f'{width}-bit integers in buffers are not supported by the WebGPU lowering.')
            word = self.emit(out, 'OpBitcast', self.u32, value) if signed else value
            out.append(_Inst(None, 'OpStore', [self.word_ptr(out, var, off), word]))
            if width == 64:
                if signed:
                    high = self.emit(out, 'OpShiftRightArithmetic', t, value, self.u32_const(31))
                    high = self.emit(out, 'OpBitcast', self.u32, high)
                else:
                    high = self.u32_const(0)
                out.append(_Inst(None, 'OpStore', [self.word_ptr(out, var, self.add_offset(out, off, 4)), high]))
            return
        if d.op == 'OpTypeFloat':
            if d.args[0] != '32':
                raise SpirvLoweringError(f'{d.args[0]}-bit floats in buffers are not supported by the WebGPU lowering.')
            word = self.emit(out, 'OpBitcast', self.u32, value)
            out.append(_Inst(None, 'OpStore', [self.word_ptr(out, var, off), word]))
            return
        if d.op == 'OpTypeBool':
            word = self.emit(out, 'OpSelect', self.u32, value, self.u32_const(1), self.u32_const(0))
            out.append(_Inst(None, 'OpStore', [self.word_ptr(out, var, off), word]))
            return
        for i, (et, k, ms) in enumerate(self.members(t, mstride)):
            part = self.emit(out, 'OpCompositeExtract', et, value, str(i))
            self.emit_store(out, var, et, self.add_offset(out, off, k), part, ms)

    def members(self, t, mstride=None):
        """[(member type, byte offset, matrix stride)] of a composite type."""
        d = self.types[t]
        if d.op == 'OpTypeVector':
            size = self.scalar_size(d.args[0])
            return [(d.args[0], i * size, None) for i in range(int(d.args[1]))]
        if d.op == 'OpTypeMatrix':
            stride = mstride or self.natural_size(d.args[0])
            return [(d.args[0], i * stride, None) for i in range(int(d.args[1]))]
        if d.op == 'OpTypeArray':
            count = self.int_consts[d.args[1]]
            if count > 4096:
                raise SpirvLoweringError(f'loading/storing a whole {count}-element array is not supported; index it instead.')
            stride = self.array_stride.get(t) or self.natural_size(d.args[0])
            return [(d.args[0], i * stride, mstride) for i in range(count)]
        if d.op == 'OpTypeStruct':
            return [(mt, self.member_offset[(t, i)], self.member_mstride.get((t, i))) for i, mt in enumerate(d.args)]
        raise SpirvLoweringError(f'cannot load/store {d.text()} from a buffer')

    def helper(self, kind, var, t, mstride):
        """Id of a generated `load(u32 offset) -> t` / `store(u32 offset, t)` function for var."""
        key = (kind, var, t, mstride)
        fid = self.helpers.get(key)
        if fid is not None:
            return fid
        if var == _PSB:  # split heap: loads read the read-only heap, stores go through split_store
            if kind == 'store':
                return self.split_store(t, mstride)
            return self.helper(kind, self.heap_ro, t, mstride)
        fid = self.fresh(kind)
        self.helpers[key] = fid
        body = []
        off = self.fresh('off')
        self.name(fid, f'{self.var_label(var)}_{kind}_{self.type_label(t)}')
        self.name(off, 'offset')
        if kind == 'load':
            ftype = self.function_type(t, self.u32)
            body.append(_Inst(fid, 'OpFunction', [t, 'None', ftype]))
            body.append(_Inst(off, 'OpFunctionParameter', [self.u32]))
            body.append(_Inst(self.fresh('l'), 'OpLabel', []))
            result = self.emit_load(body, var, t, off, mstride)
            body.append(_Inst(None, 'OpReturnValue', [result]))
        else:
            ftype = self.function_type(self.void, self.u32, t)
            value = self.fresh('v')
            self.name(value, 'value')
            body.append(_Inst(fid, 'OpFunction', [self.void, 'None', ftype]))
            body.append(_Inst(off, 'OpFunctionParameter', [self.u32]))
            body.append(_Inst(value, 'OpFunctionParameter', [t]))
            body.append(_Inst(self.fresh('l'), 'OpLabel', []))
            self.emit_store(body, var, t, off, value, mstride)
            body.append(_Inst(None, 'OpReturn', []))
        body.append(_Inst(None, 'OpFunctionEnd', []))
        self.helper_functions.extend(body)
        return fid

    def split_store(self, t, mstride):
        """A store through a PSB address (split_heap): to heap_rw when it's at
        or above dynamic_base, otherwise only the fault flag is raised."""
        fid = self.fresh('store_psb')
        self.helpers[('store', _PSB, t, mstride)] = fid
        off = self.fresh('off')
        value = self.fresh('v')
        self.name(fid, f'heap_store_{self.type_label(t)}')
        self.name(off, 'offset')
        self.name(value, 'value')
        l_entry, l_rw, l_fault, l_merge = (self.fresh('l') for _ in range(4))
        body = [_Inst(fid, 'OpFunction', [self.void, 'None', self.function_type(self.void, self.u32, t)]),
                _Inst(off, 'OpFunctionParameter', [self.u32]),
                _Inst(value, 'OpFunctionParameter', [t]),
                _Inst(l_entry, 'OpLabel', [])]
        is_dynamic = self.emit(body, 'OpUGreaterThanEqual', self.bool, off, self.u32_const(self.dynamic_base))
        body += [_Inst(None, 'OpSelectionMerge', [l_merge, 'None']),
                 _Inst(None, 'OpBranchConditional', [is_dynamic, l_rw, l_fault]),
                 _Inst(l_rw, 'OpLabel', [])]
        self.emit(body, 'OpFunctionCall', self.void, self.helper('store', self.heap_rw, t, mstride), off, value)
        body += [_Inst(None, 'OpBranch', [l_merge]), _Inst(l_fault, 'OpLabel', [])]
        body.append(_Inst(None, 'OpStore', [self.word_ptr(body, self.heap_rw, self.u32_const(0)), self.u32_const(1)]))
        body += [_Inst(None, 'OpBranch', [l_merge]), _Inst(l_merge, 'OpLabel', []), _Inst(None, 'OpReturn', []),
                 _Inst(None, 'OpFunctionEnd', [])]
        self.helper_functions.extend(body)
        return fid

    def name(self, target, text):
        """Debug name for a generated id (Tint turns it into the WGSL name)."""
        self.new_names.append(_Inst(None, 'OpName', [target, f'"{text}"']))

    def type_label(self, t):
        """Short readable name of a type, for generated helper names."""
        d = self.types[t]
        info = self.int_info(t)
        if info is not None:
            return f"{'i' if info[1] else 'u'}{info[0]}"
        if d.op == 'OpTypeFloat':
            return f'f{d.args[0]}'
        if d.op == 'OpTypeBool':
            return 'bool'
        if d.op == 'OpTypeVector':
            return f'vec{d.args[1]}{self.type_label(d.args[0])[0]}'
        if d.op == 'OpTypeMatrix':
            column = self.types[d.args[0]]
            return f'mat{d.args[1]}x{column.args[1]}{self.type_label(column.args[0])[0]}'
        if d.op == 'OpTypeArray':
            return f'{self.type_label(d.args[0])}_x{self.int_consts[d.args[1]]}'
        return self.names.get(t, 'struct')

    def var_label(self, var):
        if var == self.heap_ro:
            return 'heap_ro' if self.split_heap else 'heap'
        if var == self.heap_rw:
            return 'heap_rw'
        return self.names.get(var, 'buffer')

    def function_type(self, ret, *params):
        key = (ret, *params)
        fid = self.function_types.get(key)
        if fid is None:
            fid = self.fresh('fn')
            self.function_types[key] = fid
            self.new_globals.append(_Inst(fid, 'OpTypeFunction', [ret, *params]))
        return fid

    # ------------------------------------------------------------- the pass

    def run(self):
        self.new_globals = []
        self.new_names = []
        self.chain_info = {}  # memory-pointer SSA id -> (pointee type, matrix stride)
        self.consts = {}
        self.helpers = {}
        self.helper_functions = []
        self.function_types = {}
        self.u32 = '%rdv_u32'
        self.void = '%rdv_void'
        self.bool = '%rdv_bool'
        self.ptr_sb_u32 = '%rdv_ptr_sb_u32'
        self.new_globals += [
            _Inst(self.u32, 'OpTypeInt', ['32', '0']),
            _Inst(self.void, 'OpTypeVoid', []),
            _Inst(self.bool, 'OpTypeBool', []),
            _Inst(self.ptr_sb_u32, 'OpTypePointer', ['StorageBuffer', self.u32]),
            _Inst('%rdv_words', 'OpTypeRuntimeArray', [self.u32]),
            _Inst('%rdv_rw_block', 'OpTypeStruct', ['%rdv_words']),
            _Inst('%rdv_ro_block', 'OpTypeStruct', ['%rdv_words']),
            _Inst('%rdv_ptr_rw_block', 'OpTypePointer', ['StorageBuffer', '%rdv_rw_block']),
            _Inst('%rdv_ptr_ro_block', 'OpTypePointer', ['StorageBuffer', '%rdv_ro_block']),
        ]
        self.name('%rdv_rw_block', 'Words')
        self.name('%rdv_ro_block', 'ReadOnlyWords')
        self.new_names += [_Inst(None, 'OpMemberName', [b, '0', '"words"']) for b in ('%rdv_rw_block', '%rdv_ro_block')]
        new_annotations = [
            _Inst(None, 'OpDecorate', ['%rdv_words', 'ArrayStride', '4']),
            _Inst(None, 'OpDecorate', ['%rdv_rw_block', 'Block']),
            _Inst(None, 'OpMemberDecorate', ['%rdv_rw_block', '0', 'Offset', '0']),
            _Inst(None, 'OpDecorate', ['%rdv_ro_block', 'Block']),
            _Inst(None, 'OpMemberDecorate', ['%rdv_ro_block', '0', 'Offset', '0']),
            _Inst(None, 'OpMemberDecorate', ['%rdv_ro_block', '0', 'NonWritable']),
        ]

        # Every buffer-backed global becomes a u32-word storage buffer, keeping
        # its own id (and so its OpName) - descriptors keep their set/binding.
        # every binding already taken in the module - buffers, but also the
        # images/samplers that pass through untouched
        used = {int(d[1]) for decs in self.decorations.values() for d in decs if d[0] == 'Binding'}
        next_free = max(used | {self.heap_binding}) + 1
        self.mem_var = {}   # original global id -> word buffer id
        self.heap = None
        uses_psb = any(self.is_psb_ptr(t) for t in self.types)
        self.heap_ro = self.heap_rw = None
        if uses_psb:
            # PSB addresses resolve through self.heap: the heap variable
            # itself, or (split_heap) the _PSB marker helper() resolves to
            # the read-only heap for loads and to split_store for stores.
            self.heap_ro = '%rdv_heap'
            block = '%rdv_ptr_ro_block' if self.split_heap else '%rdv_ptr_rw_block'
            self.new_globals.append(_Inst(self.heap_ro, 'OpVariable', [block, 'StorageBuffer']))
            new_annotations += [_Inst(None, 'OpDecorate', [self.heap_ro, 'DescriptorSet', '0']),
                                _Inst(None, 'OpDecorate', [self.heap_ro, 'Binding', str(self.heap_binding)])]
            self.bindings.append(dict(binding=self.heap_binding, kind='heap_static' if self.split_heap else 'heap',
                                      read_only=self.split_heap, var=self.heap_ro))
            self.heap = self.heap_ro
            self.name(self.heap_ro, self.var_label(self.heap_ro))
            if self.split_heap:
                if self.dynamic_base is None:
                    raise SpirvLoweringError('split_heap needs dynamic_base (the first run-time heap address).')
                self.heap = _PSB
                self.heap_rw = '%rdv_heap_rw'
                rw_binding = next_free
                next_free += 1
                self.name(self.heap_rw, 'heap_rw')
                self.new_globals.append(_Inst(self.heap_rw, 'OpVariable', ['%rdv_ptr_rw_block', 'StorageBuffer']))
                new_annotations += [_Inst(None, 'OpDecorate', [self.heap_rw, 'DescriptorSet', '0']),
                                    _Inst(None, 'OpDecorate', [self.heap_rw, 'Binding', str(rw_binding)])]
                self.bindings.append(dict(binding=rw_binding, kind='heap_dynamic', read_only=False, var=self.heap_rw))
        for var, (ptr_type, sc) in self.global_vars.items():
            decs = {d[0]: d[1:] for d in self.decorations.get(var, [])}
            read_only = sc in ('Uniform', 'PushConstant')
            if sc == 'PushConstant':
                binding = self.push_binding if self.push_binding is not None else next_free
                next_free = max(next_free, binding + 1)
                new_annotations += [_Inst(None, 'OpDecorate', [var, 'DescriptorSet', '0']),
                                    _Inst(None, 'OpDecorate', [var, 'Binding', str(binding)])]
                if var not in self.names:
                    self.names[var] = 'push'
                    self.name(var, 'push')
                self.bindings.append(dict(binding=binding, kind='push', read_only=True, var=var,
                                          size=self.natural_size(self.pointee(ptr_type))))
            else:
                if decs.get('DescriptorSet', ['0'])[0] != '0':
                    raise SpirvLoweringError('only descriptor set 0 is supported by the WebGPU lowering.')
                binding = int(decs['Binding'][0])
                new_annotations += [_Inst(None, 'OpDecorate', [var, 'DescriptorSet', '0']),
                                    _Inst(None, 'OpDecorate', [var, 'Binding', str(binding)])]
                self.bindings.append(dict(binding=binding, kind='descriptor', read_only=read_only, var=var))
            block = '%rdv_ptr_ro_block' if read_only else '%rdv_ptr_rw_block'
            self.new_globals.append(_Inst(var, 'OpVariable', [block, 'StorageBuffer']))
            self.mem_var[var] = var

        # Plain (non-array) images and samplers pass through untouched - Tint
        # maps them to texture_Nd/sampler bindings; just report them. Arrays
        # of them (ImagingLibrary's bindless registry) have no WebGPU
        # equivalent - see SingleImageLibrary instead.
        for inst in self.insts:
            if inst.op != 'OpVariable' or inst.args[1] != 'UniformConstant':
                continue
            pointee = self.types.get(self.pointee(inst.args[0]))
            if pointee is None or pointee.op not in ('OpTypeImage', 'OpTypeSampler'):
                continue
            decs = {d[0]: d[1:] for d in self.decorations.get(inst.result, [])}
            if 'Binding' not in decs:
                continue
            self.bindings.append(dict(binding=int(decs['Binding'][0]), var=inst.result,
                                      kind='texture' if pointee.op == 'OpTypeImage' else 'sampler'))

        header, debug, annotations, globals_, functions = [], [], [], [], []
        forward_declared = set()
        section = header
        for inst in self.insts:
            op, a = inst.op, inst.args
            if op in ('OpCapability', 'OpExtension', 'OpExtInstImport', 'OpMemoryModel', 'OpEntryPoint',
                      'OpExecutionMode', 'OpExecutionModeId'):
                section = header
            elif op in ('OpString', 'OpSource', 'OpSourceExtension', 'OpSourceContinued', 'OpName', 'OpMemberName',
                        'OpModuleProcessed'):
                section = debug
            elif op in ('OpDecorate', 'OpMemberDecorate', 'OpDecorationGroup', 'OpGroupDecorate',
                        'OpGroupMemberDecorate', 'OpDecorateId', 'OpDecorateString', 'OpMemberDecorateString'):
                section = annotations
            elif op == 'OpFunction':
                section = functions
            elif section is not functions and (op.startswith('OpType') or op.startswith('OpConstant') or
                                               op.startswith('OpSpecConstant') or op in ('OpVariable', 'OpUndef')):
                section = globals_
            if section is functions:
                functions.append(inst)
                continue
            # ---- module-level rewrites
            if op == 'OpCapability' and a[0] in _DROPPED_CAPABILITIES:
                continue
            if op == 'OpExtension' and a[0] in _DROPPED_EXTENSIONS:
                continue
            if op == 'OpExtInstImport' and inst.result == self.printf_set:
                continue
            if op == 'OpMemoryModel':
                inst = _Inst(None, op, ['Logical', a[1]])
            if op == 'OpName':
                if a[0] not in self.names:
                    continue  # nameless (or unusable) - the generated names below may fill it in
                inst = _Inst(None, op, [a[0], f'"{self.names[a[0]]}"'])
            if op == 'OpMemberName':
                name = _identifier(a[2].strip('"'))
                if not name:
                    continue
                inst = _Inst(None, op, [a[0], a[1], f'"{name}"'])
            if op == 'OpEntryPoint':
                inst = _Inst(None, op, [x for x in a if not (_is_id(x) and x in self.global_vars and
                                                             self.global_vars[x][1] != 'Input')])
            if op in ('OpDecorate', 'OpMemberDecorate'):
                deco = a[1] if op == 'OpDecorate' else a[2]
                if deco in _DROPPED_DECORATIONS or a[0] in self.mem_var:
                    continue
            if op == 'OpTypeForwardPointer':
                forward_declared.add(a[0])
                inst = _Inst(a[0], 'OpTypeInt', ['32', '0'])
            elif op == 'OpTypePointer' and a[0] == 'PhysicalStorageBuffer':
                if inst.result in forward_declared:
                    continue
                inst = _Inst(inst.result, 'OpTypeInt', ['32', '0'])
            elif op == 'OpTypePointer' and a[0] in _MEMORY_CLASSES:
                continue  # only ever used by the variables/access chains rewritten away
            elif op == 'OpTypeInt' and a[0] == '64':
                inst = _Inst(inst.result, op, ['32', a[1]])
            elif op == 'OpTypeFloat' and a[0] == '64':
                raise SpirvLoweringError('double precision is not supported by WebGPU.')
            elif op in ('OpConstant', 'OpSpecConstant') and (self.int_info(a[0]) or (0, 0))[0] == 64:
                value = int(a[1], 0)
                low = value & 0xFFFFFFFF
                signed = self.int_info(a[0])[1]
                if signed and low >= 1 << 31:
                    low -= 1 << 32
                if low != value and value != (1 << 64) - 1 and value & 0xFFFFFFFF00000000 != 0xFFFFFFFF00000000:
                    self.warnings.append(f'64-bit constant {value:#x} truncated to {low & 0xFFFFFFFF:#x}')
                self.int_consts[inst.result] = low
                inst = _Inst(inst.result, op, [a[0], str(low)])
            elif op == 'OpVariable' and inst.result in self.mem_var:
                continue
            section.append(inst)

        functions = self.rewrite_functions(functions)
        module = header + debug + self.new_names + annotations + new_annotations + globals_ + self.new_globals + functions + \
            self.helper_functions
        module = self.dedupe_types(module)
        return '\n'.join(inst.text() for inst in module) + '\n'

    def rewrite_functions(self, insts):
        out = []
        ptr_var = {}  # logical memory-pointer SSA id -> word buffer it points into
        for inst in insts:
            op, a, r = inst.op, inst.args, inst.result

            def var_of(pid):
                if pid in self.mem_var:
                    return self.mem_var[pid]
                if pid in ptr_var:
                    return ptr_var[pid]
                if self.is_psb_ptr(self.orig_type.get(pid)):
                    return self.heap
                return None

            def offset_of(pid):
                return self.u32_const(0) if pid in self.mem_var else pid

            if op == 'OpExtInst' and a[1] == self.printf_set:
                continue
            if op in ('OpAccessChain', 'OpInBoundsAccessChain', 'OpPtrAccessChain', 'OpInBoundsPtrAccessChain') \
                    and var_of(a[1]) is not None:
                base = a[1]
                var = var_of(base)
                t = self.pointee(self.orig_type[base])
                off = offset_of(base)
                indices = a[2:]
                mstride = None
                if op in ('OpPtrAccessChain', 'OpInBoundsPtrAccessChain'):
                    stride = self.array_stride.get(self.orig_type[base]) or self.natural_size(t)
                    off = self.index_offset(out, off, indices[0], stride)
                    indices = indices[1:]
                for idx in indices:
                    d = self.types[t]
                    if d.op == 'OpTypeStruct':
                        member = self.int_consts[idx]
                        off = self.add_offset(out, off, self.member_offset[(t, member)])
                        mstride = self.member_mstride.get((t, member))
                        t = d.args[member]
                    elif d.op in ('OpTypeArray', 'OpTypeRuntimeArray'):
                        stride = self.array_stride.get(t) or self.natural_size(d.args[0])
                        off = self.index_offset(out, off, idx, stride)
                        t = d.args[0]
                    elif d.op == 'OpTypeMatrix':
                        off = self.index_offset(out, off, idx, mstride or self.natural_size(d.args[0]))
                        t = d.args[0]
                    elif d.op == 'OpTypeVector':
                        off = self.index_offset(out, off, idx, self.scalar_size(d.args[0]))
                        t = d.args[0]
                    else:
                        raise SpirvLoweringError(f'cannot index into {d.text()}')
                # the chain's own result id becomes its u32 byte offset
                out.append(_Inst(r, 'OpCopyObject', [self.u32, off]))
                if not self.is_psb_ptr(a[0]):
                    ptr_var[r] = var
                self.chain_info[r] = (t, mstride)
                continue
            if op == 'OpLoad' and var_of(a[1]) is not None:
                var = var_of(a[1])
                t, mstride = self.pointee_info(a[1])
                out.append(_Inst(r, 'OpFunctionCall', [a[0], self.helper('load', var, t, mstride), offset_of(a[1])]))
                continue
            if op == 'OpStore' and var_of(a[0]) is not None:
                var = var_of(a[0])
                t, mstride = self.pointee_info(a[0])
                out.append(_Inst(self.fresh(), 'OpFunctionCall',
                                 [self.void, self.helper('store', var, t, mstride), offset_of(a[0]), a[1]]))
                continue
            if op == 'OpCopyMemory' and (var_of(a[0]) is not None or var_of(a[1]) is not None):
                raise SpirvLoweringError('OpCopyMemory on buffer memory is not supported yet.')
            if op in _ATOMIC_OPS:
                pidx = 0 if op == 'OpAtomicStore' else 1
                var = var_of(a[pidx])
                if var is not None:
                    t, _ = self.pointee_info(a[pidx])
                    if self.int_info(t) != (32, False):
                        raise SpirvLoweringError('only uint atomics on buffer memory are supported by the WebGPU lowering.')
                    args = list(a)
                    off = offset_of(a[pidx])
                    if var == _PSB:  # atomics only make sense on run-time (heap_rw) memory
                        var = self.heap_rw
                    args[pidx] = self.word_ptr(out, var, off)
                    out.append(_Inst(r, op, args))
                    continue
            if op == 'OpArrayLength' and a[1] in self.mem_var:
                struct = self.pointee(self.orig_type[a[1]])
                member = int(a[2])
                array_type = self.types[struct].args[member]
                stride = self.array_stride.get(array_type) or self.natural_size(self.types[array_type].args[0])
                words = self.emit(out, 'OpArrayLength', self.u32, self.mem_var[a[1]], '0')
                nbytes = self.emit(out, 'OpIMul', self.u32, words, self.u32_const(4))
                nbytes = self.emit(out, 'OpISub', self.u32, nbytes, self.u32_const(self.member_offset[(struct, member)]))
                out.append(_Inst(r, 'OpUDiv', [a[0], nbytes, self.u32_const(stride)]))
                continue
            if op == 'OpFunctionParameter' and self.is_mem_ptr_type(a[0]) and not self.is_psb_ptr(a[0]):
                raise SpirvLoweringError('functions taking buffer pointers as parameters are not supported.')
            if op in ('OpUConvert', 'OpSConvert', 'OpConvertUToPtr', 'OpConvertPtrToU'):
                src = self.int_info(self.orig_type.get(a[1], '')) or (32, False)
                dst = self.int_info(a[0])
                if dst is not None and min(dst[0], 32) == min(src[0], 32):
                    out.append(_Inst(r, 'OpCopyObject' if dst[1] == src[1] else 'OpBitcast', a))
                    continue
            if op == 'OpBitcast':
                src_t = self.orig_type.get(a[1], '')
                src, dst = self.int_info(src_t), self.int_info(a[0])
                if src is not None and dst is not None and src[0] == 64 and dst[0] == 64:
                    out.append(_Inst(r, 'OpCopyObject' if dst[1] == src[1] else 'OpBitcast', a))
                    continue
                if (src is not None and src[0] == 64) or (dst is not None and dst[0] == 64):
                    raise SpirvLoweringError('bitcasts between 64-bit integers and other types are not supported.')
            if op in _SHIFT_OPS:
                info = self.int_info(a[0])
                if info is not None and info[0] == 64 and self.int_consts.get(a[2], 0) >= 32:
                    raise SpirvLoweringError(
                        f'a 64-bit shift by {self.int_consts[a[2]]} cannot be narrowed to 32 bits '
                        f'(guard that code with #ifdef RDV_WEBGPU).')
            if op in ('OpLoad', 'OpStore') and len(a) > (2 if op == 'OpLoad' else 2):
                # Aligned/Nontemporal memory operands only apply to PSB access
                inst = _Inst(r, op, a[:2])
            out.append(inst)
        return out

    def index_offset(self, out, off, idx, stride):
        if idx in self.int_consts:
            return self.add_offset(out, off, self.int_consts[idx] * stride)
        scaled = self.emit(out, 'OpIMul', self.u32, self.to_u32(out, idx), self.u32_const(stride))
        return self.emit(out, 'OpIAdd', self.u32, off, scaled)

    def pointee_info(self, pid):
        if pid in self.chain_info:
            return self.chain_info[pid]
        t = self.orig_type[pid]
        return self.pointee(t), None

    def dedupe_types(self, module):
        """Merges now-identical non-aggregate type declarations (first wins)."""
        alias = {}
        seen = {}
        out = []
        for inst in module:
            args = [alias.get(x, x) for x in inst.args]
            inst = _Inst(inst.result, inst.op, args)
            if inst.op in _UNIQUE_TYPES:
                key = (inst.op, *args)
                if key in seen:
                    alias[inst.result] = seen[key]
                    continue
                seen[key] = inst.result
            out.append(inst)
        # forward references (e.g. a function type declared before a later
        # alias was found) - a second pass settles every remaining use.
        return [_Inst(inst.result, inst.op, [alias.get(x, x) for x in inst.args]) for inst in out]


def _dead_global_elimination(text):
    """Drops module-scope declarations (types/constants/variables) no function
    or entry point (transitively) uses - e.g. the old explicit-layout block
    types, which Tint would otherwise still try to translate."""
    insts = _parse(text)
    declared = {}
    inside_function = False
    for inst in insts:
        if inst.op == 'OpFunction':
            inside_function = True
        elif inst.op == 'OpFunctionEnd':
            inside_function = False
        elif not inside_function and inst.result is not None and (
                inst.op.startswith('OpType') or inst.op.startswith('OpConstant') or
                inst.op.startswith('OpSpecConstant') or inst.op in ('OpVariable', 'OpUndef')):
            declared[inst.result] = inst
    live = set()
    stack = []
    for inst in insts:
        if inst.result in declared and declared[inst.result] is inst:
            continue
        if inst.op in ('OpName', 'OpMemberName', 'OpDecorate', 'OpMemberDecorate'):
            continue
        for x in inst.args:
            if _is_id(x) and x in declared and x not in live:
                live.add(x)
                stack.append(x)
    while stack:
        for x in declared[stack.pop()].args:
            if _is_id(x) and x in declared and x not in live:
                live.add(x)
                stack.append(x)
    out = []
    for inst in insts:
        if inst.result in declared and declared[inst.result] is inst and inst.result not in live:
            continue
        if inst.op in ('OpName', 'OpMemberName', 'OpDecorate', 'OpMemberDecorate') and \
                inst.args[0] in declared and inst.args[0] not in live:
            continue
        out.append(inst)
    return '\n'.join(inst.text() for inst in out) + '\n'


def lower_spirv(text, heap_binding=0, push_binding=None, split_heap=True, dynamic_base=None):
    """
    Lowers a disassembled (spirv-dis --raw-id) megakernel module - see the
    module docstring. Returns (lowered assembly text, bindings, warnings),
    where bindings is a list of dicts {binding, kind ('heap', or
    'heap_static'/'heap_dynamic' with split_heap, |'push'|'descriptor'),
    read_only[, size]} describing every storage buffer the
    lowered module declares in group 0.
    """
    lowering = _Lowering(text, heap_binding, push_binding, split_heap, dynamic_base)
    lowered = lowering.run()
    lowered = _dead_global_elimination(lowered)
    # a buffer no function ever touches was just eliminated - don't make the
    # caller bind it.
    alive = {inst.result for inst in _parse(lowered) if inst.op == 'OpVariable'}
    bindings = [{k: v for k, v in b.items() if k != 'var'} for b in lowering.bindings if b['var'] in alive]
    return lowered, sorted(bindings, key=lambda b: b['binding']), lowering.warnings
