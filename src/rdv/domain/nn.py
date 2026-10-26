"""
Rendervous evaluation of small deep-learning models, fused directly into
Vulkan compute shaders instead of going through a framework (torch/ONNX
Runtime/etc) per-op dispatch - see sandbox_nn_mlp_compare.py for a torch
vs. rendervous comparison on a small MLP.

LinearMap/ReLU are plain Map nodes (pointwise over a runtime input tensor's
own last-dim row, one thread per row - see Map's own docstring in
rdv/maths/__init__.py), so they compose and fuse into a single megakernel
via `>>` exactly like any other Map (Add/Sub/Composite/...). Conv2dGrid is
a Grid (it needs neighboring *cells*, not just its own row) that wraps
another, already buffer-backed Grid - see its own docstring for why a
`>> barrier` must come before it in a pipeline.
"""
from ..core import *
from ..core._core import _classproperty
from .. import maths as _maps
from ..maths import (
    Map, Grid, TensorBase, TensorOperation, Generic, Generics, Deferred, Buffer, Type, RDVObject, wrap, IN, OUT,
    _as_op_input,
)
import vk as _vk


class _LinearBase(Map):
    """Shared weight/bias/id plumbing for Linear/LinearFast - see either's
    own docstring for the actual forward() strategy each uses."""
    weight: Deferred[Buffer[Type.FLOAT32]]
    bias: Deferred[Buffer[Type.FLOAT32]]
    id: int

    __next_id__ = 0

    def __init__(self, weight, bias) -> None:
        weight = weight if isinstance(weight, RDVObject) else wrap(weight)
        bias = bias if isinstance(bias, RDVObject) else wrap(bias)
        assert len(weight.shape) == 2, "weight must be [out_features, in_features]"
        out_features, in_features = weight.shape
        assert tuple(bias.shape) == (out_features,), "bias must be [out_features]"
        self.match(IN, in_features)
        self.match(OUT, out_features)
        self.weight = weight
        self.bias = bias
        # Distinguishes this instance from any other instance of the same
        # concrete class sharing the same (IN, OUT) - and so the same
        # compiled specialization/shader code - purely at the GLSL/struct
        # level (weight/bias are just buffer addresses there, uninspectable
        # for identity) - see CooperativeLibrary.all_thread_sync, which
        # every subclass's own forward() calls with this id to check
        # whether the whole dispatch workgroup is actually cooperating on
        # this exact instance right now.
        cls = type(self)
        self.id = cls.__next_id__
        cls.__next_id__ += 1


class Linear(_LinearBase):
    """y = W @ x + b, full fp32 precision - one input row per thread. See
    shaders/maps/linear.h: whenever the whole 32-row dispatch workgroup
    turns out to be evaluating this exact Linear instance (checked at
    runtime via CooperativeLibrary.all_thread_sync, never assumed - covers
    both a batched/offset dispatch mixing instances in one workgroup, and
    simply a row count that isn't a multiple of 32), each output row's
    weight values are staged into shared memory once per workgroup instead
    of independently re-read by every thread; otherwise every thread reads/
    multiplies fully independently. Always numerically exact (no tensor-
    core precision loss) - see LinearFast for the faster, lower-precision
    alternative.

    weight: [OUT, IN] (PyTorch nn.Linear layout), bias: [OUT].
    """
    __code__ = "shaders/maps/linear.h"
    __requires__ = [CooperativeLibrary]
    __next_id__ = 0


class LinearFast(_LinearBase):
    """y = W @ x + b via GL_KHR_cooperative_matrix (Vulkan's analogue of
    CUDA's WMMA/tensor cores) - see shaders/maps/linear_fast.h. Only
    usable, and only actually faster, when the current device supports
    cooperative matrices *and* the whole 32-row dispatch workgroup is
    evaluating this exact LinearFast instance (same
    CooperativeLibrary.all_thread_sync check as Linear); otherwise falls
    back to a plain, fully independent fp32 multiply - no shared-memory
    tiling tier of its own, cooperative-matrix-or-bust by design (use
    Linear if that fallback would be the common case for this layer).
    Weight/input are staged as float16 for the tensor-core matmul itself
    (this class of tensor core doesn't support float32 operands at all),
    trading precision (~1e-2 max abs error on randn-scale values) for
    throughput - the accumulator stays float32.

    weight: [OUT, IN] (PyTorch nn.Linear layout), bias: [OUT].
    """
    __code__ = "shaders/maps/linear_fast.h"
    __requires__ = [CooperativeLibrary]
    __next_id__ = 0


def linear(weight, bias, f16: bool = False):
    """`x >> linear(weight, bias)` - Linear (default, exact fp32) or, with
    f16=True, LinearFast (GL_KHR_cooperative_matrix/tensor-core matmul -
    faster when supported and the workgroup cooperates, lower precision).
    See Linear/LinearFast."""
    return (LinearFast if f16 else Linear)(weight, bias)


class ReLU(Map):
    """f(x) = max(x, 0), elementwise - a plain composable Map, no tiling
    needed."""
    __code__ = "shaders/maps/relu.h"

    def __init__(self) -> None:
        self.match(Generics.IN, Generics.OUT(self))
        self.match(Generics.OUT, Generics.IN(self))


relu = ReLU()
"""Reusable generic ReLU instance - `x >> relu`, same idiom as rdv.X
(Identity)."""


class Conv2dGrid(Grid, Generic[Generics.G, Generics.KH, Generics.KW, Generics.CIN]):
    """'same'-padded (zero border), stride-1 2D convolution over an
    already buffer-backed grid g (IN=2, OUT=CIN) - see
    shaders/grids/conv2d.h. Each of its KH*KW taps is read directly via
    g's own forward(), so g must already be materialized (e.g. via
    `previous >> barrier`) for that to be a cheap buffer read instead of
    redundantly re-evaluating however g's own value was computed once per
    tap.

    weight: [out_channels, in_channels, KH, KW] (PyTorch nn.Conv2d
    layout), bias: [out_channels].

    Built via `g >> conv2d(weight, bias)` (see conv2d()/_Conv2dDeclaration
    below, and Grid.__rshift__/__rlshift__'s own explicit check for it -
    the same "declaration" idiom already used by `>> average(n)`/
    `>> barrier`), not by calling Conv2dGrid(...) directly, since g isn't
    known yet at the point `conv2d(weight, bias)` is written in a
    pipeline.
    """
    __code__ = "shaders/grids/conv2d.h"
    __aliases__ = {
        Generics.F1: FixArray[Type.FLOAT32, 1],
    }
    g: Generics.G
    weight: Deferred[Type.UINT64]
    bias: Deferred[Type.UINT64]

    def __init__(self, g, weight, bias) -> None:
        weight = weight if isinstance(weight, RDVObject) else wrap(weight)
        bias = bias if isinstance(bias, RDVObject) else wrap(bias)
        assert len(weight.shape) == 4, "weight must be [out_channels, in_channels, KH, KW]"
        out_channels, in_channels, kh, kw = weight.shape
        assert tuple(bias.shape) == (out_channels,), "bias must be [out_channels]"
        assert OUT(type(g)) == in_channels, "g's own OUT must match weight's in_channels"
        self.match(Generics.G, type(g))
        self.match(Generics.KH, kh)
        self.match(Generics.KW, kw)
        self.match(Generics.CIN, in_channels)
        self.match(IN, IN(type(g)))
        self.match(OUT, out_channels)
        super().__init__(g._shape)
        self.g = g
        self.weight = weight
        self.bias = bias


class _Conv2dDeclaration:
    """`g >> conv2d(weight, bias)` (see Grid.__rshift__/__rlshift__, which
    both check for this explicitly, and _AverageDeclaration/
    _BarrierDeclaration for the same pipeline-style idiom) builds
    Conv2dGrid(g, weight, bias) - lets weight/bias be given before the
    grid they apply to is known."""
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def __rrshift__(self, other):
        if not isinstance(other, Grid):
            return NotImplemented
        return Conv2dGrid(other, self.weight, self.bias)


def conv2d(weight, bias) -> _Conv2dDeclaration:
    """Defines a 2D convolution to pipeline after an already buffer-backed
    Grid, e.g. `features >> barrier >> conv2d(weight, bias)` - see
    Conv2dGrid."""
    return _Conv2dDeclaration(weight, bias)


class Conv2D(Grid, Generic[Generics.G, Generics.KH, Generics.KW, Generics.CIN, Generics.COUT]):
    """'same'-padded (zero border), stride-1 2D convolution over an
    already buffer-backed grid g (H x W x CIN, IN=2, OUT=CIN) - unlike
    Conv2dGrid, whose own shape is just (H, W) with one invocation per
    pixel computing the whole out_channels-length output vector, this
    grid's shape is (H, W, out_channels, 1): one invocation per (h, w, o)
    output channel alone (OUT=1, a single scalar) - see
    shaders/grids/conv2d_per_output.h. Splits work across out_channels
    too (useful when out_channels is large relative to H*W), and - since
    global_to_indices unflattens the dispatch index fastest-axis-last -
    groups threads that share the same (h, w) but differ only in o next
    to each other in dispatch order, the natural shape for a later
    cooperative-matrix upgrade: the same K*K*CIN input patch read once
    per (h, w), multiplied against different columns of the
    [K*K*CIN, out_channels] weight matrix by different lanes (not yet
    implemented - this version still loops CIN/KH/KW independently per
    lane, same total work as Conv2dGrid).

    weight: [KH, KW, in_channels, out_channels] (channels-last - NOT
    Conv2dGrid's PyTorch [out_channels, in_channels, KH, KW] layout, kept
    out-channel-fastest to match this grid's own out_channels dispatch
    axis), bias: [out_channels].

    Built via `g >> conv2d_o(weight, bias)` (see conv2d_o()/
    _Conv2DDeclaration below, same "declaration" idiom as conv2d()/
    _Conv2dDeclaration), not by calling Conv2D(...) directly, since g
    isn't known yet at the point `conv2d_o(weight, bias)` is written in a
    pipeline.
    """
    __code__ = "shaders/grids/conv2d_per_output.h"
    g: Generics.G
    weight: Deferred[Buffer[Type.FLOAT32]]
    bias: Deferred[Buffer[Type.FLOAT32]]

    def __init__(self, g, weight, bias) -> None:
        weight = weight if isinstance(weight, RDVObject) else wrap(weight)
        bias = bias if isinstance(bias, RDVObject) else wrap(bias)
        assert len(weight.shape) == 4, "weight must be [KH, KW, in_channels, out_channels]"
        kh, kw, in_channels, out_channels = weight.shape
        assert tuple(bias.shape) == (out_channels,), "bias must be [out_channels]"
        assert OUT(type(g)) == in_channels, "g's own OUT must match weight's in_channels"
        assert len(g._shape) == 2, "g must be a 2D (H, W) grid"
        H, W = g._shape
        self.match(Generics.G, type(g))
        self.match(Generics.KH, kh)
        self.match(Generics.KW, kw)
        self.match(Generics.CIN, in_channels)
        self.match(Generics.COUT, out_channels)
        self.match(IN, 4)
        self.match(OUT, 1)
        super().__init__([H, W, out_channels, 1])
        self.g = g
        self.weight = weight
        self.bias = bias


class _Conv2DDeclaration:
    """`g >> conv2d_o(weight, bias)` (see Grid.__rshift__/__rlshift__'s
    NotImplemented fallback, which lets this __rrshift__ run) builds
    Conv2D(g, weight, bias) - lets weight/bias be given before the grid
    they apply to is known."""
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def __rrshift__(self, other):
        if not isinstance(other, Grid):
            return NotImplemented
        return Conv2D(other, self.weight, self.bias)


def conv2d_o(weight, bias) -> _Conv2DDeclaration:
    """Defines a per-output-channel 2D convolution (one invocation per
    (h, w, o), see Conv2D) to pipeline after an already buffer-backed
    Grid, e.g. `features >> barrier >> conv2d_o(weight, bias)`."""
    return _Conv2DDeclaration(weight, bias)


class Downsample2x(Grid, Generic[Generics.G]):
    """2x2 average-pool, stride 2 - shape is g's own (H, W) halved, OUT
    unchanged (per-channel) - see shaders/grids/downsample2x.h. g should
    already be buffer-backed (e.g. via `>> barrier`), since every output
    cell reads 4 of g's own cells."""
    __code__ = "shaders/grids/downsample2x.h"
    g: Generics.G

    def __init__(self, g) -> None:
        assert len(g._shape) == 2, "Downsample2x requires a 2D (H, W) grid"
        H, W = g._shape
        assert H % 2 == 0 and W % 2 == 0, "Downsample2x requires even H and W"
        self.match(Generics.G, type(g))
        self.match(IN, 2)
        self.match(OUT, OUT(type(g)))
        super().__init__([H // 2, W // 2])
        self.g = g


class Upsample2x(Grid, Generic[Generics.G]):
    """Nearest-neighbor 2x upsample - shape is g's own (H, W) doubled, OUT
    unchanged (per-channel) - see shaders/grids/upsample2x.h."""
    __code__ = "shaders/grids/upsample2x.h"
    g: Generics.G

    def __init__(self, g) -> None:
        assert len(g._shape) == 2, "Upsample2x requires a 2D (H, W) grid"
        H, W = g._shape
        self.match(Generics.G, type(g))
        self.match(IN, 2)
        self.match(OUT, OUT(type(g)))
        super().__init__([H * 2, W * 2])
        self.g = g


class ReLUTensor(TensorBase, Generic[Generics.G]):
    """f(x) = max(x, 0) - wraps another TensorBase g, same shape, one 4x8
    block per thread (see TensorBase) - see shaders/tensors/relu_tensor.h."""
    __code__ = "shaders/tensors/relu_tensor.h"
    g: Generics.G

    def __init__(self, g) -> None:
        self.match(Generics.G, type(g))
        self.match(IN, IN(type(g)))
        self.g = g
        super().__init__(g._shape)


class LinearTensor(TensorBase, Generic[Generics.G, Generics.INF, Generics.OUTF]):
    """y = W @ x + b - wraps a 2D TensorBase g (shape [BATCH, in_features]),
    producing shape [BATCH, out_features] - one 4x8 (rows x out_features)
    block per thread (see TensorBase), computed as a tiled 4x8x8 GEMM over
    8-wide chunks of in_features: each chunk's own 8x8 weight tile is read
    once and cached in registers (reused across the block's own 4 rows) -
    see shaders/tensors/linear_tensor.h, whose own docstring also covers a
    group-shared version of this (caching the input tile across the whole
    32-lane workgroup, via CooperativeLibrary.all_thread_sync, instead of
    each thread re-walking g's own forward() independently) that was
    tried and measured consistently *slower* - the extra barrier()s cost
    more than the redundant recomputation they save, at the depths/widths
    measured. No cross-thread cooperation needed for this design.

    weight: [out_features, in_features] (PyTorch nn.Linear layout),
    bias: [out_features]. batch must be a multiple of 4, in_features and
    out_features multiples of 8 (see TensorBase - no partial-block
    padding yet).
    """
    __code__ = "shaders/tensors/linear_tensor.h"
    g: Generics.G
    weight: Deferred[Buffer[Type.FLOAT32]]
    bias: Deferred[Buffer[Type.FLOAT32]]

    def __init__(self, g, weight, bias) -> None:
        weight = weight if isinstance(weight, RDVObject) else wrap(weight)
        bias = bias if isinstance(bias, RDVObject) else wrap(bias)
        assert len(g._shape) == 2, "LinearTensor requires a 2D (BATCH, in_features) tensor"
        batch, in_features = g._shape
        assert len(weight.shape) == 2, "weight must be [out_features, in_features]"
        out_features, weight_in_features = weight.shape
        assert weight_in_features == in_features, "weight's own in_features must match g's own trailing dim"
        assert tuple(bias.shape) == (out_features,), "bias must be [out_features]"
        assert in_features % 8 == 0, f"in_features ({in_features}) must be a multiple of 8"
        assert out_features % 8 == 0, f"out_features ({out_features}) must be a multiple of 8"
        self.match(Generics.G, type(g))
        self.match(Generics.INF, in_features)
        self.match(Generics.OUTF, out_features)
        self.match(IN, 2)
        self.g = g
        self.weight = weight
        self.bias = bias
        super().__init__([batch, out_features])


class ReLUTensorOp(TensorOperation):
    """f(x) = max(x, 0) - see shaders/tensors/relu_tensor_op.h and
    TensorOperation's own docstring. x may be another TensorOperation
    (reads its already-computed .result directly, no barrier needed) or
    raw external data (the first stage of a chain)."""
    __code__ = "shaders/tensors/relu_tensor_op.h"
    x_buf: Deferred[Buffer[Type.FLOAT32]]

    def __init__(self, x) -> None:
        data, shape, upstream = _as_op_input(x)
        self.x_buf = data
        self._upstream = upstream
        super().__init__(shape)


class LinearTensorOp(TensorOperation, Generic[Generics.INF, Generics.OUTF]):
    """y = W @ x + b - see shaders/tensors/linear_tensor_op.h (tiled
    4x8x8 GEMM, weight tile cached in registers per K-chunk) and
    TensorOperation's own docstring. x may be another TensorOperation or
    raw external data.

    weight: [out_features, in_features] (PyTorch nn.Linear layout),
    bias: [out_features]. batch must be a multiple of 4, in_features and
    out_features multiples of 8 (see TensorOperation - no partial-tile
    padding yet).
    """
    __code__ = "shaders/tensors/linear_tensor_op.h"
    x_buf: Deferred[Buffer[Type.FLOAT32]]
    weight: Deferred[Buffer[Type.FLOAT32]]
    bias: Deferred[Buffer[Type.FLOAT32]]

    def __init__(self, x, weight, bias) -> None:
        data, shape, upstream = _as_op_input(x)
        assert len(shape) == 2, "LinearTensorOp requires a 2D (BATCH, in_features) input"
        batch, in_features = shape
        weight = weight if isinstance(weight, RDVObject) else wrap(weight)
        bias = bias if isinstance(bias, RDVObject) else wrap(bias)
        assert len(weight.shape) == 2, "weight must be [out_features, in_features]"
        out_features, weight_in_features = weight.shape
        assert weight_in_features == in_features, "weight's own in_features must match x's own trailing dim"
        assert tuple(bias.shape) == (out_features,), "bias must be [out_features]"
        assert in_features % 8 == 0, f"in_features ({in_features}) must be a multiple of 8"
        assert out_features % 8 == 0, f"out_features ({out_features}) must be a multiple of 8"
        self.match(Generics.INF, in_features)
        self.match(Generics.OUTF, out_features)
        self.x_buf = data
        self.weight = weight
        self.bias = bias
        self._upstream = upstream
        super().__init__([batch, out_features])


class Conv2DTensorOp(TensorOperation, Generic[Generics.KH, Generics.KW, Generics.CIN, Generics.COUT, Generics.H, Generics.W]):
    """'same'-padded (zero border), stride-1 2D convolution - see
    shaders/tensors/conv2d_tensor_op.h (tiled 4x8x8x{KH*KW} GEMM, each
    tap's own weight tile cached in registers per K-chunk, same principle
    as LinearTensorOp) and TensorOperation's own docstring.

    x's own shape is [H*W, in_channels] (row-major, channels-last, H*W
    flattened the same way LinearTensorOp's own BATCH dimension is - not
    a genuine 2D/3D tensor at the TensorOperation level, which only ever
    works in flat [rows, channels] blocks) - H/W (the actual image
    dimensions that flat row count unpacks to) are given explicitly,
    since they can't be recovered from H*W alone, and bound as generics
    (like every other shape parameter here - COUT, CIN, etc.) rather than
    plain runtime fields: every row/col decode in the shader (h_of/w_of,
    the padding bounds check) becomes a divide/modulo by a compile-time
    constant instead of a genuine runtime division - one compiled
    specialization per distinct (H, W), same "implicit GEMM" idea as
    baking in in_channels/out_channels. x may be another TensorOperation
    or raw external (already H*W-flattened) data.

    weight: [KH, KW, in_channels, out_channels] (channels-last - NOT
    Conv2dGrid's PyTorch [out_channels, in_channels, KH, KW] layout, kept
    out-channel-fastest to match this op's own out_channels-block-width
    access pattern), bias: [out_channels]. H*W must be a multiple of 4,
    in_channels and out_channels multiples of 8 (see TensorOperation - no
    partial-tile padding yet).
    """
    __code__ = "shaders/tensors/conv2d_tensor_op.h"
    x_buf: Deferred[Buffer[Type.FLOAT32]]
    weight: Deferred[Buffer[Type.FLOAT32]]
    bias: Deferred[Buffer[Type.FLOAT32]]

    def __init__(self, x, weight, bias, H: int, W: int) -> None:
        data, shape, upstream = _as_op_input(x)
        assert len(shape) == 2, "Conv2DTensorOp requires a 2D (H*W, in_channels) input"
        hw, in_channels = shape
        assert hw == H * W, f"H*W ({H * W}) must match x's own row count ({hw})"
        weight = weight if isinstance(weight, RDVObject) else wrap(weight)
        bias = bias if isinstance(bias, RDVObject) else wrap(bias)
        assert len(weight.shape) == 4, "weight must be [KH, KW, in_channels, out_channels]"
        kh, kw, weight_in_channels, out_channels = weight.shape
        assert weight_in_channels == in_channels, "weight's own in_channels must match x's own trailing dim"
        assert tuple(bias.shape) == (out_channels,), "bias must be [out_channels]"
        assert in_channels % 8 == 0, f"in_channels ({in_channels}) must be a multiple of 8"
        assert out_channels % 8 == 0, f"out_channels ({out_channels}) must be a multiple of 8"
        self.match(Generics.KH, kh)
        self.match(Generics.KW, kw)
        self.match(Generics.CIN, in_channels)
        self.match(Generics.COUT, out_channels)
        self.match(Generics.H, H)
        self.match(Generics.W, W)
        self.x_buf = data
        self.weight = weight
        self.bias = bias
        self._upstream = upstream
        super().__init__([hw, out_channels])


def _tile_weight_fp16(weight_rdv, out_features, in_features):
    """Rearranges weight_rdv ([out_features, in_features], any float
    dtype) into a flat, contiguous [n_tile][k_tile][16][16] float16
    buffer - one 256-element block per (n_tile, k_tile) 16x16 tile - so
    LinearTensorOpFast's own coopMatLoad can read matB directly from it
    (no shared-memory staging/fp32->fp16 conversion at dispatch time,
    unlike x - see linear_tensor_op_fast.h). Done once, eagerly, in
    __init__ - torch-only (import kept local to avoid a hard torch
    dependency for the rest of this module)."""
    import torch
    w = weight_rdv.as_tensor('torch')
    n_tiles = out_features // 16
    k_tiles = in_features // 16
    # w is [out_features, in_features] = (nt, n_local, kt, k_local) once
    # reshaped - coopMatLoad's own matB needs each tile laid out [K, N]
    # (k_local rows, n_local cols), not [N, K], so k_local goes before
    # n_local here.
    w = w.reshape(n_tiles, 16, k_tiles, 16).permute(0, 2, 3, 1).contiguous()
    return w.to(torch.float16).reshape(-1).contiguous()


def _cast_fp16(weight_rdv):
    """Converts weight_rdv to float16, same shape/layout, no rearranging -
    for ops (LinearTensorOpSimple) that stage a weight row-tile into
    shared memory via a plain per-index copy, so all that's needed is for
    the *source* to already be float16 (skipping the float16_t(...)
    runtime conversion in that copy), unlike _tile_weight_fp16's own
    tiling (for direct, no-shared-memory buffer reads instead)."""
    import torch
    return weight_rdv.as_tensor('torch').to(torch.float16).contiguous()


def _max_subgroups(fixed_bytes, per_subgroup_bytes, divides, budget=49152, cap=32):
    """Largest power-of-2 subgroup count spt (1, 2, 4, ...) such that
    fixed_bytes + spt*per_subgroup_bytes stays within budget (49152 -
    this hardware's own real maxComputeSharedMemorySize, confirmed via
    vulkaninfo; there's no headroom above the "48KB" figure already used
    as a safety budget elsewhere in this file - it IS the hard limit) and
    divides(spt) holds (the op's own extra per-instance constraint, e.g.
    batch or out_features needing to be a multiple of 32*spt or 16*spt).
    Used to pack as many subgroups as will fit into one workgroup, so a
    single resident workgroup keeps as much of the SM busy as shared
    memory allows (see LinearTensorOpSmall/LinearTensorOpSimple/
    Conv2DTensorOpSimple3x3's own docstrings)."""
    spt = 1
    while spt * 2 <= cap and fixed_bytes + (spt * 2) * per_subgroup_bytes <= budget and divides(spt * 2):
        spt *= 2
    return spt


_WINOGRAD_G = [[1, 0, 0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0, 0, 1]]


def _winograd_transform_weight(weight_rdv, in_channels, out_channels):
    """Precomputes U = G @ g @ G^T for every (cin, cout) 3x3 filter in
    weight_rdv ([3, 3, in_channels, out_channels], channels-last), once,
    at construction - see Conv2DTensorOpWinograd3x3/
    conv2d_tensor_op_winograd3x3.h. Returns a flat, contiguous
    [16][in_channels][out_channels] float16 buffer (domain-position-
    major) - read directly by the shader (RowMajor, already [K, N])."""
    import torch
    w = weight_rdv.as_tensor('torch')
    g = w.permute(2, 3, 0, 1)  # [CIN, COUT, 3, 3]
    G = torch.tensor(_WINOGRAD_G, dtype=torch.float32, device=g.device)
    U = G @ g.to(torch.float32) @ G.transpose(0, 1)  # [CIN, COUT, 4, 4]
    U = U.reshape(in_channels, out_channels, 16).permute(2, 0, 1).contiguous()  # [16, CIN, COUT]
    return U.to(torch.float16).reshape(-1).contiguous()


class LinearTensorOpFast(TensorOperation, Generic[Generics.INF, Generics.OUTF]):
    """y = W @ x + b via GL_KHR_cooperative_matrix (tensor cores) - see
    shaders/tensors/linear_tensor_op_fast.h and TensorOperation's own
    docstring. Unlike LinearTensorOp (one 4x8 tile per thread), this one
    register-tiles C_ROWS=2 x C_COLS=2 32x16 accumulators per *workgroup*
    (so each workgroup call covers 64 rows x 32 out_features) - every one
    of the 32 lanes collectively computes each of those tiles via
    coopMatMulAdd, reusing each loaded weight tile across all C_ROWS row
    tiles and each staged input tile across all C_COLS column tiles
    (NVIDIA's own reference tiled-coopmat GEMM pattern). weight is
    rearranged once, in __init__ (see _tile_weight_fp16), into a
    contiguous, pre-converted-to-float16 tiled buffer read directly by
    coopMatLoad - only x still needs float16 staging/conversion at
    dispatch time, since only it actually changes at runtime. Trades
    precision (~1e-2 max abs error on randn-scale values, this class of
    tensor core not supporting float32 operands at all) for throughput;
    the accumulator stays float32. x may be another TensorOperation or
    raw external data.

    weight: [out_features, in_features] (PyTorch nn.Linear layout),
    bias: [out_features]. batch must be a multiple of 64, in_features a
    multiple of 16, out_features a multiple of 32 (no partial-tile
    padding, and no occupancy fallback - every dispatched workgroup is
    always fully, uniformly occupied by construction).

    x_buf/weight_tiled/bias are plain Buffer[T] fields, not
    Deferred[Buffer[T]] - a TensorOperation's own inputs are always bound
    at construction time (never left as a placeholder for a later
    deferreds() rebind - see TensorOperation's own "fixed nn pipeline"
    regime), so Deferred's own extra indirection (a resolve() call
    branching on whether it's a raw pointer or a deferred-buffer lookup)
    buys nothing here.
    """
    __code__ = "shaders/tensors/linear_tensor_op_fast.h"
    __aliases__ = {
        Generics.F256: FixArray[Type.FLOAT16, 256],
    }
    x_buf: Buffer[Type.FLOAT32]
    weight_tiled: Buffer[Type.FLOAT16]
    bias: Buffer[Type.FLOAT32]

    _TILE_ROWS = 32 * 2  # C_ROWS
    _TILE_COLS = 16 * 2  # C_COLS
    _THREADS_PER_TILE = 32

    def __init__(self, x, weight, bias) -> None:
        data, shape, upstream = _as_op_input(x)
        assert len(shape) == 2, "LinearTensorOpFast requires a 2D (BATCH, in_features) input"
        batch, in_features = shape
        weight = weight if isinstance(weight, RDVObject) else wrap(weight)
        bias = bias if isinstance(bias, RDVObject) else wrap(bias)
        assert len(weight.shape) == 2, "weight must be [out_features, in_features]"
        out_features, weight_in_features = weight.shape
        assert weight_in_features == in_features, "weight's own in_features must match x's own trailing dim"
        assert tuple(bias.shape) == (out_features,), "bias must be [out_features]"
        assert batch % self._TILE_ROWS == 0, f"batch ({batch}) must be a multiple of {self._TILE_ROWS}"
        assert in_features % 16 == 0, f"in_features ({in_features}) must be a multiple of 16"
        assert out_features % self._TILE_COLS == 0, f"out_features ({out_features}) must be a multiple of {self._TILE_COLS}"
        self.match(Generics.INF, in_features)
        self.match(Generics.OUTF, out_features)
        self.x_buf = data
        self.weight_tiled = wrap(_tile_weight_fp16(weight, out_features, in_features))
        self.bias = bias
        self._upstream = upstream
        super().__init__([batch, out_features])


class Conv2DTensorOpFast(TensorOperation, Generic[Generics.KH, Generics.KW, Generics.CIN, Generics.COUT, Generics.H, Generics.W]):
    """'same'-padded (zero border), stride-1 2D convolution via
    GL_KHR_cooperative_matrix (tensor cores) - see
    shaders/tensors/conv2d_tensor_op_fast.h and TensorOperation's own
    docstring. Unlike Conv2DTensorOp (one 4x8 tile per thread), one 32x16
    (M x N) tile per *workgroup* - all 32 lanes collectively compute this
    tile via coopMatMulAdd, tiled over 16-wide in_channels chunks *and*
    every (kh, kw) tap. The kernel (weight) tile is the one thing kept
    workgroup-shared/cached (one coopMatLoad per (kh, kw, K-chunk),
    reused identically across all 32 lanes, same as LinearTensorOpFast's
    own weight tile) rather than the spatial position, which necessarily
    varies per lane (each lane owns one fixed (h, w) throughout, so a tap
    landing outside the image is a per-*lane* zero-fill of the staged
    input tile, not a per-tile skip) - the weight tile is the smaller,
    far more reused resource (shared by every pixel of the whole image),
    so it's the one worth keeping resident/shared; the input tile cannot
    be shared the same way since every lane's own receptive field differs.

    x's own shape is [H*W, in_channels] (row-major, channels-last - see
    Conv2DTensorOp's own docstring for why H/W are given explicitly).
    weight: [KH, KW, in_channels, out_channels] (channels-last), bias:
    [out_channels]. H*W must be a multiple of 32, in_channels and
    out_channels multiples of 16 (no partial-tile padding, and no
    occupancy fallback - every dispatched workgroup is always fully,
    uniformly occupied by construction).
    """
    __code__ = "shaders/tensors/conv2d_tensor_op_fast.h"
    x_buf: Deferred[Buffer[Type.FLOAT32]]
    weight: Deferred[Buffer[Type.FLOAT32]]
    bias: Deferred[Buffer[Type.FLOAT32]]

    _TILE_ROWS = 32
    _TILE_COLS = 16
    _THREADS_PER_TILE = 32

    def __init__(self, x, weight, bias, H: int, W: int) -> None:
        data, shape, upstream = _as_op_input(x)
        assert len(shape) == 2, "Conv2DTensorOpFast requires a 2D (H*W, in_channels) input"
        hw, in_channels = shape
        assert hw == H * W, f"H*W ({H * W}) must match x's own row count ({hw})"
        weight = weight if isinstance(weight, RDVObject) else wrap(weight)
        bias = bias if isinstance(bias, RDVObject) else wrap(bias)
        assert len(weight.shape) == 4, "weight must be [KH, KW, in_channels, out_channels]"
        kh, kw, weight_in_channels, out_channels = weight.shape
        assert weight_in_channels == in_channels, "weight's own in_channels must match x's own trailing dim"
        assert tuple(bias.shape) == (out_channels,), "bias must be [out_channels]"
        assert in_channels % 16 == 0, f"in_channels ({in_channels}) must be a multiple of 16"
        assert out_channels % 16 == 0, f"out_channels ({out_channels}) must be a multiple of 16"
        self.match(Generics.KH, kh)
        self.match(Generics.KW, kw)
        self.match(Generics.CIN, in_channels)
        self.match(Generics.COUT, out_channels)
        self.match(Generics.H, H)
        self.match(Generics.W, W)
        self.x_buf = data
        self.weight = weight
        self.bias = bias
        self._upstream = upstream
        super().__init__([hw, out_channels])


class Conv2DTensorOpSimple(TensorOperation, Generic[Generics.KH, Generics.KW, Generics.CIN, Generics.COUT, Generics.H, Generics.W]):
    """'same'-padded (zero border), stride-1 2D convolution via
    GL_KHR_cooperative_matrix (tensor cores) - see
    shaders/tensors/conv2d_tensor_op_simple.h, the same "simple" idea as
    LinearTensorOpSimple (weight row-tile staged once per out-channel
    block and reused across every K-chunk; the input tile staged once per
    workgroup and reused across every out-channel block - see
    LinearTensorOpSimple's own docstring) with the reduction dimension
    flattened to KH*KW*in_channels (every tap's own in_channels slice,
    back to back) instead of just in_features.

    x's own shape is [H*W, in_channels] (row-major, channels-last - see
    Conv2DTensorOp's own docstring for why H/W are given explicitly).
    weight: [KH, KW, in_channels, out_channels] (channels-last), bias:
    [out_channels]. H*W must be a multiple of 32, in_channels and
    out_channels multiples of 16 (no partial-tile padding, and no
    occupancy fallback - every dispatched workgroup is always fully,
    uniformly occupied by construction). x_buf/weight/bias/result are
    plain (non-Deferred) Buffer fields - see LinearTensorOpFast's own
    docstring for why. Trades precision (this class of tensor core
    doesn't support float32 operands at all) for throughput; the
    accumulator stays float32.

    Unlike LinearTensorOpSimple, weight is read straight from its own
    plain float32 source (cast to float16 while staging, at dispatch
    time) rather than precast once at construction, and there's only ever
    one subgroup per workgroup.

    KH*KW*in_channels (this op's own "K") is capped to a fixed-width
    (_CHUNK_K=256) shared-memory footprint regardless of how large it
    actually is - see conv2d_tensor_op_simple.h's own header comment -
    instead of scaling shared memory with it directly the way
    LinearTensorOpSimple scales with in_features: a wide in_channels
    count or large kernel there was measured to overrun this hardware's
    per-workgroup shared memory budget and crash the whole Vulkan device
    outright (ErrorDeviceLost - not a catchable error), so shared memory
    usage here never exceeds a fixed, safe bound (~26KB, well under the
    typical 48KB budget) no matter the shape.
    """
    __code__ = "shaders/tensors/conv2d_tensor_op_simple.h"
    __aliases__ = {
        Generics.F16: FixArray[Type.FLOAT32, 16],
        Generics.F512: FixArray[Type.FLOAT32, 512],
    }
    x_buf: Buffer[Type.FLOAT32]
    weight: Buffer[Type.FLOAT32]
    bias: Buffer[Type.FLOAT32]

    _TILE_ROWS = 32
    _TILE_COLS = 16
    _THREADS_PER_TILE = 32

    @classmethod
    def _dispatch_thread_count(cls, shape):
        # Unlike Conv2DTensorOpFast, out_channels isn't tiled by dispatch
        # at all (forward() itself sweeps every out-channel block
        # internally) - only row (H*W) tiles need their own workgroup -
        # see LinearTensorOpSimple's own _dispatch_thread_count.
        row_tiles = shape[0] // cls._TILE_ROWS
        return row_tiles * cls._THREADS_PER_TILE

    def __init__(self, x, weight, bias, H: int, W: int) -> None:
        data, shape, upstream = _as_op_input(x)
        assert len(shape) == 2, "Conv2DTensorOpSimple requires a 2D (H*W, in_channels) input"
        hw, in_channels = shape
        assert hw == H * W, f"H*W ({H * W}) must match x's own row count ({hw})"
        weight = weight if isinstance(weight, RDVObject) else wrap(weight)
        bias = bias if isinstance(bias, RDVObject) else wrap(bias)
        assert len(weight.shape) == 4, "weight must be [KH, KW, in_channels, out_channels]"
        kh, kw, weight_in_channels, out_channels = weight.shape
        assert weight_in_channels == in_channels, "weight's own in_channels must match x's own trailing dim"
        assert tuple(bias.shape) == (out_channels,), "bias must be [out_channels]"
        assert hw % 32 == 0, f"H*W ({hw}) must be a multiple of 32"
        assert in_channels % 16 == 0, f"in_channels ({in_channels}) must be a multiple of 16"
        assert out_channels % 16 == 0, f"out_channels ({out_channels}) must be a multiple of 16"
        self.match(Generics.KH, kh)
        self.match(Generics.KW, kw)
        self.match(Generics.CIN, in_channels)
        self.match(Generics.COUT, out_channels)
        self.match(Generics.H, H)
        self.match(Generics.W, W)
        self.x_buf = data
        self.weight = weight
        self.bias = bias
        self._upstream = upstream
        super().__init__([hw, out_channels])


class Conv2DTensorOpSimple3x3(TensorOperation, Generic[Generics.CIN, Generics.COUT, Generics.SPT, Generics.H, Generics.W]):
    """'same'-padded (zero border), stride-1 2D convolution, fixed to a
    3x3 kernel/padding=1 - see shaders/tensors/conv2d_tensor_op_simple3x3.h,
    the same "simple" idea as LinearTensorOpSimple (weight row-tile
    staged once per out-channel block and reused across every K-chunk;
    the input tile staged once per workgroup and reused across every
    out-channel block) with the reduction dimension flattened to 9*CIN
    (9 fixed taps, not generic KH/KW like Conv2DTensorOpSimple - no
    _CHUNK_K-style chunking either, so shared memory scales directly with
    CIN, checked up front here the same way LinearTensorOpSmall does).

    SPT (subgroups per tile - a third generic, alongside CIN/COUT, not a
    fixed constant) is computed once per (in_channels, out_channels) at
    construction (see _max_subgroups), the same idea as
    LinearTensorOpSimple's own SPT: each of a workgroup's SPT subgroups
    processes a *different* out-channel block concurrently, against the
    one, shared 32-row _SH_X - the largest power of two such that _SH_X
    (32*9*CIN) + _SH_A (SPT*16*9*CIN), both float16, still fits this
    hardware's real maxComputeSharedMemorySize (49152 bytes - confirmed
    via vulkaninfo) and out_channels is still a multiple of 16*SPT.

    x's own shape is [H*W, in_channels] (row-major, channels-last - see
    Conv2DTensorOp's own docstring for why H/W are given explicitly).
    weight: [3, 3, in_channels, out_channels] (channels-last), bias:
    [out_channels]. H*W must be a multiple of 32, in_channels and
    out_channels multiples of 16 (no partial-tile padding, and no
    occupancy fallback - every dispatched workgroup is always fully,
    uniformly occupied by construction). x_buf/weight/bias are plain
    (non-Deferred) Buffer fields - see LinearTensorOpFast's own docstring
    for why. Trades precision (this class of tensor core doesn't support
    float32 operands at all) for throughput; the accumulator stays
    float32.
    """
    __code__ = "shaders/tensors/conv2d_tensor_op_simple3x3.h"
    __aliases__ = {
        Generics.F16: FixArray[Type.FLOAT32, 16],
        Generics.FOUT: FixArray[Type.FLOAT32, 512],
    }
    x_buf: Buffer[Type.FLOAT32]
    weight: Buffer[Type.FLOAT32]
    bias: Buffer[Type.FLOAT32]

    _TILE_ROWS = 32
    _TILE_COLS = 16
    _THREADS_PER_TILE = 32

    @_classproperty
    def _WORKGROUP_SIZE(cls):
        return 32 * Generics.SPT(cls)

    @classmethod
    def _dispatch_thread_count(cls, shape):
        # out_channels isn't tiled by dispatch at all (forward() itself
        # sweeps every out-channel block internally, SPT at a time) -
        # only row (H*W) tiles need their own workgroup - see
        # LinearTensorOpSimple's own _dispatch_thread_count.
        row_tiles = shape[0] // cls._TILE_ROWS
        return row_tiles * cls._WORKGROUP_SIZE

    def __init__(self, x, weight, bias, H: int, W: int) -> None:
        data, shape, upstream = _as_op_input(x)
        assert len(shape) == 2, "Conv2DTensorOpSimple3x3 requires a 2D (H*W, in_channels) input"
        hw, in_channels = shape
        assert hw == H * W, f"H*W ({H * W}) must match x's own row count ({hw})"
        weight = weight if isinstance(weight, RDVObject) else wrap(weight)
        bias = bias if isinstance(bias, RDVObject) else wrap(bias)
        assert tuple(weight.shape[:2]) == (3, 3), "weight must be [3, 3, in_channels, out_channels]"
        _, _, weight_in_channels, out_channels = weight.shape
        assert weight_in_channels == in_channels, "weight's own in_channels must match x's own trailing dim"
        assert tuple(bias.shape) == (out_channels,), "bias must be [out_channels]"
        assert hw % 32 == 0, f"H*W ({hw}) must be a multiple of 32"
        assert in_channels % 16 == 0, f"in_channels ({in_channels}) must be a multiple of 16"
        assert out_channels % 16 == 0, f"out_channels ({out_channels}) must be a multiple of 16"
        # _SH_X (32 rows, fixed) + _SH_A (SPT*16 rows) of 9*in_channels
        # float16 elements each - see conv2d_tensor_op_simple3x3.h and
        # _max_subgroups.
        tap_k = 9 * in_channels
        spt = _max_subgroups(
            fixed_bytes=32 * tap_k * 2,
            per_subgroup_bytes=16 * tap_k * 2,
            divides=lambda s: out_channels % (16 * s) == 0,
        )
        self.match(Generics.CIN, in_channels)
        self.match(Generics.COUT, out_channels)
        self.match(Generics.SPT, spt)
        self.match(Generics.H, H)
        self.match(Generics.W, W)
        self.x_buf = data
        self.weight = weight
        self.bias = bias
        self._upstream = upstream
        super().__init__([hw, out_channels])


class Conv2DTensorOpWindow3x3(TensorOperation, Generic[Generics.CIN, Generics.COUT, Generics.H, Generics.W]):
    """'same'-padded (zero border), stride-1 2D convolution, fixed to a
    3x3 kernel/padding=1 - see shaders/tensors/conv2d_tensor_op_window3x3.h.
    Unlike Conv2DTensorOpSimple3x3 (every lane gathers its own 9-tap
    neighborhood straight from global memory, redundantly re-reading
    pixels shared by adjacent output positions), one workgroup here
    cooperatively fetches the whole 10x10 input window an 8x8 tile of
    *output* pixels actually needs - every distinct input pixel read
    from global memory exactly once - then each of the tile's 64 threads
    (one per output pixel, 2 subgroups) gathers its own tap neighborhood
    back out of that small, already-resident window (shared memory only,
    no further global traffic).

    x's own shape is [H*W, in_channels] (row-major, channels-last - see
    Conv2DTensorOp's own docstring for why H/W are given explicitly).
    weight: [3, 3, in_channels, out_channels] (channels-last), bias:
    [out_channels]. H and W must both be multiples of 8 (one workgroup
    per 8x8 output tile - no partial-tile padding), in_channels and
    out_channels multiples of 16, in_channels at most 64 (so the fetched
    window + weight row-tile + per-K-chunk scratch always fit well under
    this hardware's 48KB per-workgroup shared memory budget - see
    __init__). x_buf/weight/bias are plain (non-Deferred) Buffer fields -
    see LinearTensorOpFast's own docstring for why. Trades precision
    (this class of tensor core doesn't support float32 operands at all)
    for throughput; the accumulator stays float32.

    Output goes back through a shared _SH_Y (coopMatStore'd there, then
    copied out per lane into result, same as the very original design -
    result's own layout still isn't uniform-stride for a subgroup's own
    32 rows, so a direct coopMatStore there still isn't possible).

    x itself is pre-padded once, up front (see evaluate()/__init__'s own
    phase 0), into temp - a scratch buffer, not part of this op's own
    public result, shaped (H+2)x(W+2)xCIN with an explicit, already-
    zeroed 1-pixel border - instead of reading directly from x_buf with a
    per-element bounds check/clamp against the image edge every fetch.
    Phase 1 (the actual convolution, one workgroup per tile, otherwise
    identical to the original design) then fetches its own 10x10 window
    straight out of temp with plain offset arithmetic only - no bounds
    check, no clamp() - since every index it could ever read is
    guaranteed already valid (either real data or the pre-baked zero
    border). Two phases here aren't optional the way splitting the
    output computation by out-channel block was: the padding has to be
    fully written, image-wide, before *any* tile starts fetching from
    it (adjacent tiles' own windows overlap into each other's padding),
    so it can't be folded into each tile's own workgroup the way the
    original cooperative fetch was.
    """
    __code__ = "shaders/tensors/conv2d_tensor_op_window3x3.h"
    __aliases__ = {
        Generics.F16: FixArray[Type.FLOAT32, 16],
        Generics.FBUFFER: Buffer[Type.FLOAT32],
    }
    x_buf: Buffer[Type.FLOAT32]
    weight: Buffer[Type.FLOAT32]
    bias: Buffer[Type.FLOAT32]
    temp: Buffer[Type.FLOAT32]
    phase: int

    _WIN = 8  # matches shaders/tensors/conv2d_tensor_op_window3x3.h's own _WIN
    _TILE_ROWS = 1  # unused - see _dispatch_thread_count below (tiled 2D, not 1D-row)
    _TILE_COLS = 16
    _THREADS_PER_TILE = 32
    _WORKGROUP_SIZE = _WIN * _WIN

    @classmethod
    def _pad_thread_count(cls):
        # phase 0: one thread per padded pixel (over-dispatched up to a
        # multiple of the workgroup size, with a bounds check inside the
        # shader for the remainder - the padded image size itself has no
        # reason to already be a multiple of 64).
        H = Generics.H(cls)
        W = Generics.W(cls)
        padded_pixels = (H + 2) * (W + 2)
        return -(-padded_pixels // cls._WORKGROUP_SIZE) * cls._WORKGROUP_SIZE

    @classmethod
    def _conv_thread_count(cls, shape):
        # phase 1: one workgroup per WIN x WIN output tile - (H/WIN)*(W/WIN)
        # tiles is always exactly hw/WIN^2 regardless of H/W individually
        # (H/WIN * W/WIN == H*W/WIN^2), so shape[0] (hw) alone is enough,
        # same as every other TensorOperation's own override here.
        hw = shape[0]
        return hw // (cls._WIN * cls._WIN) * cls._WORKGROUP_SIZE

    @classmethod
    def _dispatch_thread_count(cls, shape):
        # This is what TensorOperation's own automatic first dispatch
        # (inside super().__init__(), with phase already set to 0
        # beforehand - see __init__) uses - phase 1 (the actual
        # convolution) is dispatched manually afterward, both here and in
        # evaluate().
        return cls._pad_thread_count()

    def evaluate(self):
        self.phase = 0
        dispatch(self._evaluator, type(self)._pad_thread_count())
        self.phase = 1
        dispatch(self._evaluator, type(self)._conv_thread_count(self._shape))

    def __init__(self, x, weight, bias, H: int, W: int) -> None:
        data, shape, upstream = _as_op_input(x)
        assert len(shape) == 2, "Conv2DTensorOpWindow3x3 requires a 2D (H*W, in_channels) input"
        hw, in_channels = shape
        assert hw == H * W, f"H*W ({H * W}) must match x's own row count ({hw})"
        weight = weight if isinstance(weight, RDVObject) else wrap(weight)
        bias = bias if isinstance(bias, RDVObject) else wrap(bias)
        assert tuple(weight.shape[:2]) == (3, 3), "weight must be [3, 3, in_channels, out_channels]"
        _, _, weight_in_channels, out_channels = weight.shape
        assert weight_in_channels == in_channels, "weight's own in_channels must match x's own trailing dim"
        assert tuple(bias.shape) == (out_channels,), "bias must be [out_channels]"
        assert H % self._WIN == 0, f"H ({H}) must be a multiple of {self._WIN}"
        assert W % self._WIN == 0, f"W ({W}) must be a multiple of {self._WIN}"
        assert in_channels % 16 == 0, f"in_channels ({in_channels}) must be a multiple of 16"
        assert out_channels % 16 == 0, f"out_channels ({out_channels}) must be a multiple of 16"
        assert in_channels <= 64, f"in_channels ({in_channels}) must be at most 64 (see this op's own docstring)"
        # _SH_WIN (10x10xCIN) + _SH_A (16x9xCIN) + _SH_XC (64x16) + _SH_Y
        # (64x16, float32) - see conv2d_tensor_op_window3x3.h. Checked
        # explicitly (rather than left to fail at dispatch time) since
        # overrunning this hardware's per-workgroup shared memory budget
        # was observed elsewhere (Conv2DTensorOpSimple) to crash the
        # whole Vulkan device, not raise a catchable error.
        fetch = self._WIN + 2
        shared_bytes = (fetch * fetch * in_channels + 16 * 9 * in_channels) * 2 + 64 * 16 * 2 + 64 * 16 * 4
        assert shared_bytes <= 49152, f"in_channels ({in_channels}) needs {shared_bytes} bytes of shared memory (48KB budget)"
        self.match(Generics.CIN, in_channels)
        self.match(Generics.COUT, out_channels)
        self.match(Generics.H, H)
        self.match(Generics.W, W)
        self.x_buf = data
        self.weight = weight
        self.bias = bias
        temp_t = RDVObject(_vk.tensor([(H + 2) * (W + 2), in_channels], Type.FLOAT32, location=MemoryLocation.DEVICE))
        self.temp = temp_t
        self.phase = 0
        self._upstream = upstream
        was_generic = type(self).__is_generic__
        super().__init__([hw, out_channels])
        if not was_generic:
            # super().__init__() just above already issued phase 0 (pad)
            # as its own automatic first dispatch (see
            # _dispatch_thread_count) - __init__'s own first (still-
            # generic) pass returns early before that, before
            # self._evaluator even exists (see TensorOperation's own
            # __init__), so only this, the second, fully-specialized
            # pass reaches here at all - issue phase 1 (the actual
            # convolution) now.
            self.phase = 1
            dispatch(self._evaluator, type(self)._conv_thread_count([hw, out_channels]))


class Conv2DTensorOpWinograd3x3(TensorOperation, Generic[Generics.CIN, Generics.COUT, Generics.H, Generics.W]):
    """'same'-padded (zero border), stride-1 2D convolution, fixed to a
    3x3 kernel/padding=1, via Winograd F(2x2, 3x3) - see
    shaders/tensors/conv2d_tensor_op_winograd3x3.h. Unlike every other
    Conv2D op here (a direct GEMM reducing over 9*in_channels), a 2x2
    output tile is produced from its own 4x4 input tile by transforming
    both into a 4x4 ("Winograd domain") representation, running one
    in_channels-wide GEMM per domain position (16 of them, each a
    coopMatMulAdd - the actual tensor-core work) instead of one
    9*in_channels-wide GEMM, then inverse-transforming the 16 results
    back into the 2x2 output - fewer effective multiplies for the same
    output, at the cost of the (cheap, additions-only) forward/inverse
    transforms. weight is pretransformed once, at construction (see
    _winograd_transform_weight), into a flat [16][in_channels][out_channels]
    float16 buffer.

    x's own shape is [H*W, in_channels] (row-major, channels-last - see
    Conv2DTensorOp's own docstring for why H/W are given explicitly). H
    and W must both be even (one lane per 2x2 output tile - no partial-
    tile padding) and (H/2)*(W/2) a multiple of 32 (one workgroup per 32
    tiles). weight: [3, 3, in_channels, out_channels] (channels-last),
    bias: [out_channels]. in_channels and out_channels multiples of 16,
    at most 48 (see __init__ for the exact shared memory accounting -
    _SH_M alone, holding all 16 domain positions' own [32 tiles, 16
    out-channels] running sums simultaneously for the final inverse
    transform, is a fixed 32KB regardless of in_channels/out_channels,
    leaving comparatively little headroom for _SH_V/_SH_U to scale into).
    x_buf/weight_u/bias are plain (non-Deferred) Buffer fields - see
    LinearTensorOpFast's own docstring for why. Trades precision (this
    class of tensor core doesn't support float32 operands at all) for
    throughput; every accumulation (both the per-domain-position GEMM and
    the running sum across in_channels chunks) stays float32.
    """
    __code__ = "shaders/tensors/conv2d_tensor_op_winograd3x3.h"
    x_buf: Buffer[Type.FLOAT32]
    weight_u: Buffer[Type.FLOAT16]
    bias: Buffer[Type.FLOAT32]

    _TILE_ROWS = 1  # unused - see _dispatch_thread_count below (tiled, not row-flat)
    _TILE_COLS = 16
    _THREADS_PER_TILE = 32
    _WORKGROUP_SIZE = 32

    @classmethod
    def _dispatch_thread_count(cls, shape):
        # one thread per 2x2 output tile - (H/2)*(W/2) is always exactly
        # hw/4 regardless of H/W individually, same reasoning as
        # Conv2DTensorOpWindow3x3's own override.
        hw = shape[0]
        return hw // 4

    def __init__(self, x, weight, bias, H: int, W: int) -> None:
        data, shape, upstream = _as_op_input(x)
        assert len(shape) == 2, "Conv2DTensorOpWinograd3x3 requires a 2D (H*W, in_channels) input"
        hw, in_channels = shape
        assert hw == H * W, f"H*W ({H * W}) must match x's own row count ({hw})"
        weight = weight if isinstance(weight, RDVObject) else wrap(weight)
        bias = bias if isinstance(bias, RDVObject) else wrap(bias)
        assert tuple(weight.shape[:2]) == (3, 3), "weight must be [3, 3, in_channels, out_channels]"
        _, _, weight_in_channels, out_channels = weight.shape
        assert weight_in_channels == in_channels, "weight's own in_channels must match x's own trailing dim"
        assert tuple(bias.shape) == (out_channels,), "bias must be [out_channels]"
        assert H % 2 == 0, f"H ({H}) must be even"
        assert W % 2 == 0, f"W ({W}) must be even"
        assert (H // 2) * (W // 2) % 32 == 0, f"(H/2)*(W/2) ({(H // 2) * (W // 2)}) must be a multiple of 32"
        assert in_channels % 16 == 0, f"in_channels ({in_channels}) must be a multiple of 16"
        assert out_channels % 16 == 0, f"out_channels ({out_channels}) must be a multiple of 16"
        assert in_channels <= 48, f"in_channels ({in_channels}) must be at most 48 (see this op's own docstring)"
        assert out_channels <= 48, f"out_channels ({out_channels}) must be at most 48 (see this op's own docstring)"
        # _SH_V (32xCIN) + _SH_U (16x16), float16, plus the fixed
        # _SH_M (16x32x16, float32) - see conv2d_tensor_op_winograd3x3.h.
        # Checked explicitly (rather than left to fail at dispatch time)
        # since overrunning this hardware's per-workgroup shared memory
        # budget was observed elsewhere (Conv2DTensorOpSimple) to crash
        # the whole Vulkan device, not raise a catchable error.
        shared_bytes = (32 * in_channels + 16 * 16) * 2 + 16 * 32 * 16 * 4
        assert shared_bytes <= 49152, f"in_channels ({in_channels}) needs {shared_bytes} bytes of shared memory (48KB budget)"
        self.match(Generics.CIN, in_channels)
        self.match(Generics.COUT, out_channels)
        self.match(Generics.H, H)
        self.match(Generics.W, W)
        self.x_buf = data
        self.weight_u = wrap(_winograd_transform_weight(weight, in_channels, out_channels))
        self.bias = bias
        self._upstream = upstream
        super().__init__([hw, out_channels])


class Downsample2xTensorOp(TensorOperation):
    """2x2 average-pool, stride 2, over a flattened [H*W, C] input (H, W
    both even) - channel count unchanged - see
    shaders/tensors/downsample2x_tensor_op.h."""
    __code__ = "shaders/tensors/downsample2x_tensor_op.h"
    x_buf: Deferred[Buffer[Type.FLOAT32]]
    W: int

    def __init__(self, x, H: int, W: int) -> None:
        data, shape, upstream = _as_op_input(x)
        assert len(shape) == 2, "Downsample2xTensorOp requires a 2D (H*W, C) input"
        hw, channels = shape
        assert hw == H * W, f"H*W ({H * W}) must match x's own row count ({hw})"
        assert H % 2 == 0 and W % 2 == 0, "Downsample2xTensorOp requires even H and W"
        self.x_buf = data
        self.W = W // 2  # output width
        self._upstream = upstream
        super().__init__([(H // 2) * (W // 2), channels])


class Upsample2xTensorOp(TensorOperation):
    """Nearest-neighbor 2x upsample over a flattened [H*W, C] input -
    channel count unchanged - see shaders/tensors/upsample2x_tensor_op.h."""
    __code__ = "shaders/tensors/upsample2x_tensor_op.h"
    x_buf: Deferred[Buffer[Type.FLOAT32]]
    W: int

    def __init__(self, x, H: int, W: int) -> None:
        data, shape, upstream = _as_op_input(x)
        assert len(shape) == 2, "Upsample2xTensorOp requires a 2D (H*W, C) input"
        hw, channels = shape
        assert hw == H * W, f"H*W ({H * W}) must match x's own row count ({hw})"
        self.x_buf = data
        self.W = W * 2  # output width
        self._upstream = upstream
        super().__init__([(H * 2) * (W * 2), channels])


class ConcatTensorOp(TensorOperation):
    """Channel concat of two ops/tensors sharing the same H*W leading dim
    - see shaders/tensors/concat_tensor_op.h. Both operands' own channel
    counts must be multiples of 8 (so no 4x8 tile ever straddles the
    C1/C2 boundary - see TensorOperation)."""
    __code__ = "shaders/tensors/concat_tensor_op.h"
    x1_buf: Deferred[Buffer[Type.FLOAT32]]
    x2_buf: Deferred[Buffer[Type.FLOAT32]]
    C1: int

    def __init__(self, g1, g2) -> None:
        data1, shape1, up1 = _as_op_input(g1)
        data2, shape2, up2 = _as_op_input(g2)
        assert len(shape1) == 2 and len(shape2) == 2, "ConcatTensorOp requires 2D (H*W, C) inputs"
        assert shape1[0] == shape2[0], "g1/g2 must share the same H*W leading dim"
        assert shape1[1] % 8 == 0, f"g1's own channels ({shape1[1]}) must be a multiple of 8"
        assert shape2[1] % 8 == 0, f"g2's own channels ({shape2[1]}) must be a multiple of 8"
        self.x1_buf = data1
        self.x2_buf = data2
        self.C1 = shape1[1]
        self._upstream = up1
        self._extra_upstreams = (up2,) if up2 is not None else ()
        super().__init__([shape1[0], shape1[1] + shape2[1]])


class LinearTensorOpMine(TensorOperation, Generic[Generics.INF, Generics.OUTF]):
    """y = W @ x + b via GL_KHR_cooperative_matrix (tensor cores) - see
    shaders/tensors/linear_tensor_op_mine.h, adapted from
    jeffbolznv/vk_cooperative_matrix_perf's own shmem.comp: shared memory
    padded to avoid bank conflicts, and a double-buffered prefetch
    pipeline (each K-chunk's global reads for x/weight are issued one
    iteration ahead, so they run in the background while the previous
    chunk's coopMatMulAdd executes). One 32x16 tile per workgroup (no
    register tiling, unlike LinearTensorOpFast). x_buf/weight/bias/result
    are plain (non-Deferred) Buffer fields - see LinearTensorOpFast's own
    docstring for why. Trades precision (this class of tensor core
    doesn't support float32 operands at all) for throughput; the
    accumulator stays float32.

    weight: [out_features, in_features] (PyTorch nn.Linear layout),
    bias: [out_features]. batch must be a multiple of 32, in_features and
    out_features multiples of 16 (no partial-tile padding, and no
    occupancy fallback - every dispatched workgroup is always fully,
    uniformly occupied by construction).
    """
    __code__ = "shaders/tensors/linear_tensor_op_mine.h"
    x_buf: Buffer[Type.FLOAT32]
    weight: Buffer[Type.FLOAT32]
    bias: Buffer[Type.FLOAT32]

    _TILE_ROWS = 32
    _TILE_COLS = 16
    _THREADS_PER_TILE = 32

    def __init__(self, x, weight, bias) -> None:
        data, shape, upstream = _as_op_input(x)
        assert len(shape) == 2, "LinearTensorOpMine requires a 2D (BATCH, in_features) input"
        batch, in_features = shape
        weight = weight if isinstance(weight, RDVObject) else wrap(weight)
        bias = bias if isinstance(bias, RDVObject) else wrap(bias)
        assert len(weight.shape) == 2, "weight must be [out_features, in_features]"
        out_features, weight_in_features = weight.shape
        assert weight_in_features == in_features, "weight's own in_features must match x's own trailing dim"
        assert tuple(bias.shape) == (out_features,), "bias must be [out_features]"
        assert batch % 32 == 0, f"batch ({batch}) must be a multiple of 32"
        assert in_features % 16 == 0, f"in_features ({in_features}) must be a multiple of 16"
        assert out_features % 16 == 0, f"out_features ({out_features}) must be a multiple of 16"
        self.match(Generics.INF, in_features)
        self.match(Generics.OUTF, out_features)
        self.x_buf = data
        self.weight = weight
        self.bias = bias
        self._upstream = upstream
        super().__init__([batch, out_features])


class LinearTensorOpSimple(TensorOperation, Generic[Generics.INF, Generics.OUTF, Generics.SPT]):
    """y = W @ x + b via GL_KHR_cooperative_matrix (tensor cores) - see
    shaders/tensors/linear_tensor_op_simple.h. One workgroup per 32-row
    batch of x (not per (row, out_feature) tile - unlike every other
    coopmat-based op here, the sweep over out_features happens *inside*
    one thread's own forward(), not via a separate dispatch tile): x's
    own 32 rows are staged (float16) once and reused for every
    out_feature block, and each block's own weight row-tile (16 x
    in_features) is staged once and reused across all of that block's
    own K-chunks - trading more shared memory (both x's whole row-batch
    and one full weight row-tile resident at once) for far fewer
    barriers than tiling every (row, out_feature, K-chunk) triple
    separately. x_buf/weight/bias/result are plain (non-Deferred) Buffer
    fields - see LinearTensorOpFast's own docstring for why. Trades
    precision (this class of tensor core doesn't support float32
    operands at all) for throughput; the accumulator stays float32.

    weight: [out_features, in_features] (PyTorch nn.Linear layout),
    bias: [out_features]. batch must be a multiple of 32, in_features a
    multiple of 16, out_features a multiple of 16*_SUBGROUPS_PER_TILE (no
    partial-tile padding, and no occupancy fallback - every dispatched
    workgroup is always fully, uniformly occupied by construction).

    weight is precast to float16 once, at construction (_cast_fp16) - same
    layout, no rearranging (see _tile_weight_fp16's own docstring for why
    this is simpler than LinearTensorOpFast's own tiling: this op still
    stages a weight row-tile into shared memory via a plain copy, so only
    the source dtype needs to change, not its layout).

    SPT (subgroups per tile - a third generic, alongside INF/OUTF, not a
    fixed constant) is computed once per (in_features, out_features) at
    construction (see _max_subgroups): each of a workgroup's SPT
    subgroups processes a *different* 16-out-feature block concurrently
    (staging and consuming its own 16xIN slice of _SH_A independently,
    against the one, shared 32xIN _SH_X row-batch), so _SH_A's own total
    size scales to SPT*16*IN. SPT is the largest power of two such that
    _SH_X (32*IN) + _SH_A (SPT*16*IN), both float16, still fits this
    hardware's real maxComputeSharedMemorySize (49152 bytes - confirmed
    via vulkaninfo) and out_features is still a multiple of 16*SPT -
    packing as many subgroups as shared memory allows into one workgroup,
    so a single resident workgroup keeps as much of the SM busy as
    possible.
    """
    __code__ = "shaders/tensors/linear_tensor_op_simple.h"
    __aliases__ = {
        Generics.F16: FixArray[Type.FLOAT32, 16],
        Generics.F512: FixArray[Type.FLOAT32, 512],
        Generics.FBUFFER: Buffer[Type.FLOAT32]
    }
    x_buf: Buffer[Type.FLOAT32]
    weight: Buffer[Type.FLOAT16]
    bias: Buffer[Type.FLOAT32]

    _TILE_ROWS = 32
    _TILE_COLS = 16
    _THREADS_PER_TILE = 32

    @_classproperty
    def _WORKGROUP_SIZE(cls):
        return 32 * Generics.SPT(cls)

    @classmethod
    def _dispatch_thread_count(cls, shape):
        # Unlike every other coopmat op here, out_features isn't tiled by
        # dispatch at all (forward() itself sweeps every out_feature
        # block internally, SPT at a time) - only row (batch) tiles need
        # their own workgroup.
        row_tiles = shape[0] // cls._TILE_ROWS
        return row_tiles * cls._WORKGROUP_SIZE

    def __init__(self, x, weight, bias) -> None:
        data, shape, upstream = _as_op_input(x)
        assert len(shape) == 2, "LinearTensorOpSimple requires a 2D (BATCH, in_features) input"
        batch, in_features = shape
        weight = weight if isinstance(weight, RDVObject) else wrap(weight)
        bias = bias if isinstance(bias, RDVObject) else wrap(bias)
        assert len(weight.shape) == 2, "weight must be [out_features, in_features]"
        out_features, weight_in_features = weight.shape
        assert weight_in_features == in_features, "weight's own in_features must match x's own trailing dim"
        assert tuple(bias.shape) == (out_features,), "bias must be [out_features]"
        assert batch % 32 == 0, f"batch ({batch}) must be a multiple of 32"
        assert in_features % 16 == 0, f"in_features ({in_features}) must be a multiple of 16"
        spt = _max_subgroups(
            fixed_bytes=32 * in_features * 2,
            per_subgroup_bytes=16 * in_features * 2,
            divides=lambda s: out_features % (16 * s) == 0,
        )
        out_tile = 16 * spt
        assert out_features % out_tile == 0, f"out_features ({out_features}) must be a multiple of {out_tile}"
        self.match(Generics.INF, in_features)
        self.match(Generics.OUTF, out_features)
        self.match(Generics.SPT, spt)
        self.x_buf = data
        self.weight = wrap(_cast_fp16(weight))
        self.bias = bias
        self._upstream = upstream
        super().__init__([batch, out_features])


class LinearTensorOpSmall(TensorOperation, Generic[Generics.INF, Generics.OUTF, Generics.SPT]):
    """y = W @ x + b via GL_KHR_cooperative_matrix (tensor cores) - see
    shaders/tensors/linear_tensor_op_small.h. For weight small enough
    (both in_features and out_features <= 128) to be staged into shared
    memory *entirely*, once, as float16 - unlike LinearTensorOpSimple,
    which only ever keeps one 16-row slice of weight resident at a time
    (re-staged per out-feature block), every out-feature block here reads
    directly off the one, already-resident copy: no re-staging, and no
    barrier at all between out-feature blocks (nothing shared is
    overwritten once staged). x is staged once too (converted to
    float16, straight off the buffer). Each out-feature block's own
    accumulator is seeded with bias (broadcast via a stride-0
    coopMatLoad) and, once accumulated over every in_features/16
    K-chunk, stored once directly into
    deref(resolve(deref(node).result)) - no intermediate shared output
    buffer/copy-out loop (see LinearTensorOpSimple's own later revision
    for the same store trick).

    Since weight is already fully resident and shared by the whole
    workgroup regardless of subgroup count, extra subgroups are spent on
    *rows*, not out-feature blocks (unlike LinearTensorOpSimple, where
    weight itself isn't fully resident): each subgroup processes its own,
    different 32-row slice of x concurrently, sweeping every out-feature
    block against the one shared weight copy - so dispatch is exactly one
    thread per output row (_dispatch_thread_count == shape[0], i.e.
    batch), independent of subgroup count.

    SPT (subgroups per tile - a third generic, alongside INF/OUTF, not a
    fixed constant) is computed once per (batch, in_features,
    out_features) at construction (see _max_subgroups) as the largest
    power of two such that _SH_X (SPT*32*in_features) + _SH_W
    (out_features*in_features), both float16, still fits this hardware's
    real maxComputeSharedMemorySize (49152 bytes - confirmed via
    vulkaninfo, not just a conservative guess) and batch is still a
    multiple of 32*SPT - packing as many subgroups as shared memory
    allows into one workgroup, so a single resident workgroup keeps as
    much of the SM busy as possible (Vulkan reports
    maxComputeWorkGroupInvocations=1024, so this is capped at 32
    subgroups regardless, but shared memory binds first for every shape
    this op supports).

    weight: [out_features, in_features] (PyTorch nn.Linear layout),
    bias: [out_features]. batch must be a multiple of 32, in_features and
    out_features multiples of 16 and at most 128 (no partial-tile
    padding, and no occupancy fallback - every dispatched workgroup is
    always fully, uniformly occupied by construction). x_buf/weight/bias
    are plain (non-Deferred) Buffer fields - see LinearTensorOpFast's own
    docstring for why. Trades precision (this class of tensor core
    doesn't support float32 operands at all) for throughput; the
    accumulator stays float32.
    """
    __code__ = "shaders/tensors/linear_tensor_op_small.h"
    __aliases__ = {
        Generics.F16: FixArray[Type.FLOAT32, 16],
        Generics.FOUT: FixArray[Type.FLOAT32, 4096],
        Generics.FBUFFER: Buffer[Type.FLOAT32]
    }
    x_buf: Buffer[Type.FLOAT32]
    weight: Buffer[Type.FLOAT32]
    bias: Buffer[Type.FLOAT32]

    _TILE_ROWS = 32
    _TILE_COLS = 16
    _THREADS_PER_TILE = 32

    @_classproperty
    def _WORKGROUP_SIZE(cls):
        return 32 * Generics.SPT(cls)

    @classmethod
    def _dispatch_thread_count(cls, shape):
        # One thread per output row - out_features isn't tiled by
        # dispatch at all (forward() itself sweeps every out-feature
        # block internally, against the one already-resident weight
        # matrix), and SPT only changes how many rows one workgroup
        # covers, not the total thread count.
        return shape[0]

    def __init__(self, x, weight, bias) -> None:
        data, shape, upstream = _as_op_input(x)
        assert len(shape) == 2, "LinearTensorOpSmall requires a 2D (BATCH, in_features) input"
        batch, in_features = shape
        weight = weight if isinstance(weight, RDVObject) else wrap(weight)
        bias = bias if isinstance(bias, RDVObject) else wrap(bias)
        assert len(weight.shape) == 2, "weight must be [out_features, in_features]"
        out_features, weight_in_features = weight.shape
        assert weight_in_features == in_features, "weight's own in_features must match x's own trailing dim"
        assert tuple(bias.shape) == (out_features,), "bias must be [out_features]"
        assert batch % 32 == 0, f"batch ({batch}) must be a multiple of 32"
        assert in_features % 16 == 0, f"in_features ({in_features}) must be a multiple of 16"
        assert out_features % 16 == 0, f"out_features ({out_features}) must be a multiple of 16"
        assert in_features <= 128, f"in_features ({in_features}) must be at most 128 (see this op's own docstring)"
        assert out_features <= 128, f"out_features ({out_features}) must be at most 128 (see this op's own docstring)"
        # _SH_W (the whole out_features x in_features matrix) is fixed;
        # _SH_X (32*SPT rows) scales with SPT - see _max_subgroups and
        # this class's own docstring.
        spt = _max_subgroups(
            fixed_bytes=out_features * in_features * 2,
            per_subgroup_bytes=32 * in_features * 2,
            divides=lambda s: batch % (32 * s) == 0,
        )
        self.match(Generics.INF, in_features)
        self.match(Generics.OUTF, out_features)
        self.match(Generics.SPT, spt)
        self.x_buf = data
        self.weight = weight
        self.bias = bias
        self._upstream = upstream
        super().__init__([batch, out_features])
