from ..core import *
from ..core._core import _layout_of, _classproperty
from ..core._common import _reinterpret, _reshape
import typing as _typing
import vk as _vk
import math as _math


IN = Generics.IN
OUT = Generics.OUT

pi = _math.pi
e = _math.e
two_pi = 2 * _math.pi
pi_over_4 = _math.pi / 4
pi_over_2 = _math.pi / 2
four_pi = 4 * _math.pi
one_over_four_pi = 1 / (4 * _math.pi)


def _direct_offsets(node, *names):
    """
    Lazily resolves (once per instance, cached as a plain attribute - see
    set_direct on the Main-derived evaluator types below) the byte offset
    of each of node's own fields named in `names`, via the normal
    Accessor._field_handle machinery - set_direct itself then writes
    through them directly (Accessor.write_uint64/write_int/write_float),
    bypassing Node.__setattr__/Accessor.__setattr__/_set/_set_leaf's own
    per-field dispatch entirely on every later call.
    """
    offsets = getattr(node, '_direct_offsets_cache', None)
    if offsets is None:
        acc = node.__accessor__
        offsets = tuple(acc._field_handle(name).offset for name in names)
        node._direct_offsets_cache = offsets
    return offsets


# ================ GRIDS =====================


class Grid(Component, Generic[IN, OUT]):
    """
    Base class for grids. A grid represents a map
    from index locations [0...SHAPE[i]-1] to float vectors.
    """
    __requires__ = [
        GridLibrary[IN]
    ]
    __precode__ = "shaders/grids/grid_precode.h"
    shape: Array[IN, int]

    __evaluator__ = None
    """
    Cached evaluator dispatching the full grid (see GridEvaluator).
    """
    __batch_evaluator__ = None
    """
    Cached evaluator dispatching only the requested indices (see BatchGridEvaluator).
    """
    __capture_evaluators__ = None
    """
    Per-class cache of Capture evaluators, keyed by the concrete map type
    used to capture() with - see capture().
    """
    __batch_capture_evaluators__ = None
    """
    Per-class cache of BatchCapture evaluators, keyed by the concrete map
    type used to batch_capture() with - see batch_capture().
    """
    _shape = None

    def __init__(self, shape: list[int]):
        self._shape = shape  # fast access
        self.shape = shape

    def __call__(self, *args, **kwargs):
        cls = type(self)
        __OUT__ = OUT(cls)
        if len(args) == 0:
            # Evaluator: one thread per grid cell, in row-major order -
            # equivalent to copying the whole grid into a tensor of shape
            # [*shape, OUT].
            grid_shape = list(self._shape)
            evaluator = cls.__evaluator__
            if evaluator is None or evaluator._cached_output_shape != grid_shape:
                # A fresh RDVObject here means a fresh vk allocation and a
                # fresh DLPack export the first time as_tensor() is called
                # on it - reused (like the evaluator itself already was)
                # across calls whenever the output shape doesn't actually
                # change, e.g. every frame of a fixed-resolution render
                # loop, instead of paying both costs again every time.
                output_t = RDVObject(_vk.tensor([*grid_shape, __OUT__], Type.FLOAT32, location=MemoryLocation.DEVICE))
                if evaluator is None:
                    evaluator = GridEvaluator(self, output_t.device_ptr)
                    cls.__evaluator__ = evaluator
                evaluator._cached_output = output_t
                evaluator._cached_output_shape = grid_shape
            else:
                output_t = evaluator._cached_output
            if getattr(evaluator, '_cached_grid', None) is not self:
                # set_direct below only ever refreshes the raw device
                # pointer, not the Python-level shadow value __getattr__
                # reads back (see Accessor._set_leaf vs _direct_offsets'
                # own docstring) - refreshed here instead, only on the rare
                # call where a *different* grid instance reuses this same
                # cached evaluator, so e.g. dispatch()'s own
                # _collect_compute_dependencies (which does need the actual
                # current instance, not whichever one first built this
                # evaluator) still sees the right one.
                evaluator.__accessor__._values['g'] = self
                evaluator._cached_grid = self
            evaluator.set_direct(self, output_t.device_ptr)
            if kwargs:
                DeferredLibrary.bind(**kwargs)
            dispatch(evaluator, _math.prod(grid_shape))
            if kwargs:
                DeferredLibrary.unbind(**kwargs)
            return output_t
        # Batch: one thread per requested index vector - batch's last
        # dimension holds the IN-sized (int) index into the grid, mirroring
        # Map.__call__'s x.shape[-1] convention.
        assert len(args) == 1
        batch, = args
        if not isinstance(batch, RDVObject):
            batch = wrap(batch)
        batch_shape = list(batch.shape[:-1])
        evaluator = cls.__batch_evaluator__
        if evaluator is None or evaluator._cached_output_shape != batch_shape:
            output_t = RDVObject(_vk.tensor([*batch_shape, __OUT__], Type.FLOAT32, location=MemoryLocation.DEVICE))
            if evaluator is None:
                evaluator = BatchGridEvaluator(self, batch.device_ptr, output_t.device_ptr)
                cls.__batch_evaluator__ = evaluator
            evaluator._cached_output = output_t
            evaluator._cached_output_shape = batch_shape
        else:
            output_t = evaluator._cached_output
        if getattr(evaluator, '_cached_grid', None) is not self:
            # See the same guard in the no-args branch above.
            evaluator.__accessor__._values['g'] = self
            evaluator._cached_grid = self
        evaluator.set_direct(self, batch.device_ptr, output_t.device_ptr)
        if kwargs:
            DeferredLibrary.bind(**kwargs)
        dispatch(evaluator, _math.prod(batch_shape))
        if kwargs:
            DeferredLibrary.unbind(**kwargs)
        return output_t

    def __rshift__(self, map):
        if isinstance(map, Identity):
            return self
        if isinstance(map, _BarrierDeclaration):
            return map.create_barrier(self)
        if isinstance(map, _AverageDeclaration):
            return GridAverage(self, map.n)
        if isinstance(map, Map):
            return Sensor(self, map)
        # Not a Map and not one of the pipeline declarations checked
        # explicitly above - give map's own __rrshift__ a chance (e.g.
        # rdv.domain.nn's `g >> conv2d(weight, bias)`, the same
        # "declaration built before the grid it applies to is known"
        # idiom _AverageDeclaration/_BarrierDeclaration use, just defined
        # outside this module so it can't be isinstance-checked here
        # directly without a circular import).
        return NotImplemented

    def __rlshift__(self, map):
        if isinstance(map, Identity):
            return self
        if isinstance(map, _BarrierDeclaration):
            return map.create_barrier(self)
        if isinstance(map, _AverageDeclaration):
            return GridAverage(self, map.n)
        if isinstance(map, Map):
            return Sensor(self, map)
        return NotImplemented

    def __getitem__(self, indices) -> 'Grid':
        """Picks components (ints and/or slices, resolved against this
        grid's own OUT) from this grid's own output at every index - see
        GridSelect."""
        return GridSelect(self, indices)

    def promote(self, dim: int):
        assert OUT(self) == 1
        if dim == 1:
            return self
        return GridPromote(self, dim)

    def __add__(self, other):
        return GridAdd(self, other)

    def __radd__(self, other):
        return GridAdd(other, self)

    def __mul__(self, other):
        return GridMul(self, other)

    def __rmul__(self, other):
        return GridMul(other, self)

    def __sub__(self, other):
        return GridSub(self, other)

    def __rsub__(self, other):
        return GridSub(other, self)

    def __truediv__(self, other):
        return GridDiv(self, other)

    def __rtruediv__(self, other):
        return GridDiv(other, self)

    def __floordiv__(self, other):
        return GridMod(self, other)

    def __rfloordiv__(self, other):
        return GridMod(other, self)

    def __or__(self, other):
        return ConcatGrid(self, other)

    def __ror__(self, other):
        return ConcatGrid(other, self)


class RefGrid(Grid, Generic[Generics.G]):
    grid: Generics.G
    __code__ = "shaders/grids/ref_grid.h"
    def __init__(self, grid: Generics.G) -> None:
        self.match(Generics.IN, Generics.IN(grid))
        self.match(Generics.OUT, Generics.OUT(grid))
        self.match(Generics.G, type(grid))
        # Grid.__init__ (never called here otherwise) is what actually
        # populates both _shape (Python-only, used by e.g. this instance's
        # own __call__ dispatch sizing) and shape (the GLSL-visible field -
        # read by anything that cares about a grid's own dimensions, e.g.
        # VoxelMajorantField/VolumeProfileDDA indexing into a wrapped
        # TensorGrid) - without it both silently stay None/uninitialized
        # forever, since RefGrid has no shape of its own besides whatever
        # it's currently wrapping. Mirrors grid's own _shape at
        # construction time only - callers that later swap .grid in place
        # (e.g. ScatNet.set_extinction's self.majorant_grid.grid = ...,
        # precisely so downstream nodes already built on top of this same
        # RefGrid instance don't need rebuilding) must update .shape/
        # ._shape the same way, or they'll silently go stale.
        super().__init__(list(grid._shape) if grid._shape is not None else None)
        self.grid = grid


class Sensor(Grid, Generic[Generics.G, Generics.M]):
    __aliases__ = {
        Generics.V: OUT(Generics.G)
    }
    grid: Generics.G
    map: Generics.M

    __code__ = "shaders/grids/sensor.h"

    def __init__(self, grid: Generics.G, map: Generics.M):
        map = map.cast(Map[IN: OUT(grid)])
        grid = grid.cast(Grid[OUT: IN(map)])
        assert OUT(grid) == IN(map)
        self.match(IN, IN(grid))
        self.match(OUT, OUT(map))
        self.match(Generics.G, type(grid))
        self.match(Generics.M, type(map))
        super().__init__(grid.shape)
        self.grid = grid
        self.map = map


    # def capture(self, map, samples: int = 1, **kwargs) -> RDVObject:
    #     """
    #     Evaluates map(self(idx)) for every position idx in this grid's own
    #     domain, averaging `samples` independent grid-then-map round trips
    #     per position - see Capture. The evaluator is cached per (grid class,
    #     map type) - self/map/samples are refreshed on the cached instance
    #     on every call, so samples can freely change call to call. Returns a
    #     tensor of shape [*shape, map's OUT].
    #     """
    #     cls = type(self)
    #     M = type(map)
    #     __OUT__ = OUT(M)
    #     if cls.__capture_evaluators__ is None:
    #         cls.__capture_evaluators__ = {}
    #     grid_shape = list(self._shape)
    #     evaluator = cls.__capture_evaluators__.get(M)
    #     if evaluator is None or evaluator._cached_output_shape != grid_shape:
    #         output_t = RDVObject(_vk.tensor([*grid_shape, __OUT__], Type.FLOAT32, location=MemoryLocation.DEVICE))
    #         if evaluator is None:
    #             evaluator = Capture(self, map, samples, output_t.device_ptr)
    #             cls.__capture_evaluators__[M] = evaluator
    #         evaluator._cached_output = output_t
    #         evaluator._cached_output_shape = grid_shape
    #     else:
    #         output_t = evaluator._cached_output
    #     evaluator.set_direct(self, map, samples, output_t.device_ptr)
    #     if kwargs:
    #         DeferredLibrary.bind(**kwargs)
    #     dispatch(evaluator, _math.prod(grid_shape))
    #     if kwargs:
    #         DeferredLibrary.unbind(**kwargs)
    #     return output_t
    #
    # def batch_capture(self, map, batch, samples: int = 1, **kwargs) -> RDVObject:
    #     """
    #     Like capture(), but only at the requested indices (see BatchCapture)
    #     - batch's own last dimension holds an IN-sized index into this grid,
    #     mirroring __call__'s batch convention. The evaluator is cached per
    #     (grid class, map type), same as capture(). Returns a tensor of
    #     shape [*batch.shape[:-1], map's OUT].
    #     """
    #     cls = type(self)
    #     if not isinstance(batch, RDVObject):
    #         batch = wrap(batch)
    #     M = type(map)
    #     __OUT__ = OUT(M)
    #     if cls.__batch_capture_evaluators__ is None:
    #         cls.__batch_capture_evaluators__ = {}
    #     batch_shape = list(batch.shape[:-1])
    #     evaluator = cls.__batch_capture_evaluators__.get(M)
    #     if evaluator is None or evaluator._cached_output_shape != batch_shape:
    #         output_t = RDVObject(_vk.tensor([*batch_shape, __OUT__], Type.FLOAT32, location=MemoryLocation.DEVICE))
    #         if evaluator is None:
    #             evaluator = BatchCapture(self, map, samples, batch.device_ptr, output_t.device_ptr)
    #             cls.__batch_capture_evaluators__[M] = evaluator
    #         evaluator._cached_output = output_t
    #         evaluator._cached_output_shape = batch_shape
    #     else:
    #         output_t = evaluator._cached_output
    #     evaluator.set_direct(self, map, samples, batch.device_ptr, output_t.device_ptr)
    #     if kwargs:
    #         DeferredLibrary.bind(**kwargs)
    #     dispatch(evaluator, _math.prod(batch_shape))
    #     if kwargs:
    #         DeferredLibrary.unbind(**kwargs)
    #     return output_t


class GridSelect(Grid, Generic[Generics.G, Generics.V]):
    """Grid equivalent of Select: picks a fixed list of components (by index
    into g's own V-sized output) from the wrapped grid g's own output at
    every index - see Grid.__getitem__, the preferred way to build one
    (resolves slices for you). Unlike Map's Select, g is always already
    concrete here (grids, unlike Map's X/Identity, are never built generic
    ahead of their own data), so this stays a plain, eager constructor."""
    g: Generics.G
    sel: Array[OUT, int]
    __code__ = "shaders/grids/select.h"

    def __init__(self, g, indices) -> None:
        if not isinstance(indices, tuple):
            indices = (indices,)
        G_ = type(g)
        v = OUT(G_)
        picked = []
        for idx in indices:
            if not isinstance(idx, slice):
                picked.append(idx)
                continue
            picked.extend(range(*idx.indices(v)))
        assert all(0 <= i < v for i in picked), \
            f"GridSelect: every index must be in [0, {v}) - g's own OUT."
        self.match(IN, IN(G_))
        self.match(OUT, len(picked))
        self.match(Generics.G, G_)
        self.match(Generics.V, v)
        super().__init__(g._shape)
        self.g = g
        self.sel = picked


class GridPromote(Grid, Generic[Generics.G]):
    """Grid equivalent of Promote: repeats a scalar-valued grid (g's own
    OUT == 1) `dimension` times: result(idx) = [g(idx), g(idx), ...]."""
    g: Generics.G
    __code__ = "shaders/grids/promote.h"

    def __init__(self, g, dimension: int) -> None:
        G_ = type(g)
        assert OUT(G_) == 1, "g must be scalar-valued (OUT == 1) to promote."
        self.match(IN, IN(G_))
        self.match(OUT, dimension)
        self.match(Generics.G, G_)
        super().__init__(g._shape)
        self.g = g


class GridAverage(Grid, Generic[Generics.G]):
    """Grid equivalent of Average: evaluates the wrapped grid g at the same
    index n times and averages the results - see average(), the preferred
    way to build one via `g >> average(n)`."""
    g: Generics.G
    samples: int
    __code__ = "shaders/grids/average.h"

    def __init__(self, g, n: int) -> None:
        G_ = type(g)
        self.match(IN, IN(G_))
        self.match(OUT, OUT(G_))
        self.match(Generics.G, G_)
        super().__init__(g._shape)
        self.g = g
        self.samples = n


class GridBinaryOperation(Grid, Generic[Generics.G1, Generics.G2]):
    """Grid equivalent of BinaryOperation - unlike Map's version, both
    operands must already be grids (there's no grid equivalent of a
    constant/scalar Map to broadcast a bare Python number through)."""
    __abstract__ = True
    g1: Generics.G1
    g2: Generics.G2
    __code__ = "shaders/grids/binary.h"

    def __init__(self, g1, g2) -> None:
        assert isinstance(g1, Grid) and isinstance(g2, Grid), \
            "Grid binary operations require both operands to already be grids."
        if OUT(type(g1)) == 1 and OUT(type(g2)) != 1:
            g1 = g1.promote(OUT(type(g2)))
        if OUT(type(g2)) == 1 and OUT(type(g1)) != 1:
            g2 = g2.promote(OUT(type(g1)))
        assert IN(type(g1)) == IN(type(g2)), "Grid binary operations require both operands to share the same IN."
        assert OUT(type(g1)) == OUT(type(g2)), "Grid binary operations require both operands to share the same OUT."
        self.match(IN, IN(type(g1)))
        self.match(OUT, OUT(type(g1)))
        self.match(Generics.G1, type(g1))
        self.match(Generics.G2, type(g2))
        super().__init__(g1._shape)
        self.g1 = g1
        self.g2 = g2


class GridAdd(GridBinaryOperation):
    __precode__ = "#define BINARY_OP(x, y) ((x) + (y))"
    __poscode__ = "#undef BINARY_OP"


class GridSub(GridBinaryOperation):
    __precode__ = "#define BINARY_OP(x, y) ((x) - (y))"
    __poscode__ = "#undef BINARY_OP"


class GridMul(GridBinaryOperation):
    __precode__ = "#define BINARY_OP(x, y) ((x) * (y))"
    __poscode__ = "#undef BINARY_OP"


class GridDiv(GridBinaryOperation):
    __precode__ = "#define BINARY_OP(x, y) ((x) / (y))"
    __poscode__ = "#undef BINARY_OP"


class GridMod(GridBinaryOperation):
    __precode__ = "#define BINARY_OP(x, y) ((x) % (y))"
    __poscode__ = "#undef BINARY_OP"


class ConcatGrid(Grid, Generic[Generics.G1, Generics.G2]):
    """Grid equivalent of ConcatMap - both operands must already be grids
    (see GridBinaryOperation) and must share IN; the resulting OUT is the
    sum of both operands' own OUT."""
    __aliases__ = {
        Generics.G1_OUT: OUT(Generics.G1),
        Generics.G2_OUT: OUT(Generics.G2),
    }
    g1: Generics.G1
    g2: Generics.G2
    __code__ = "shaders/grids/concat.h"

    def __init__(self, g1, g2) -> None:
        assert isinstance(g1, Grid) and isinstance(g2, Grid), \
            "Grid concatenation requires both operands to already be grids."
        assert IN(type(g1)) == IN(type(g2)), "ConcatGrid requires both operands to share the same IN."
        self.match(IN, IN(type(g1)))
        if OUT(type(g1)) is not None and OUT(type(g2)) is not None:
            self.match(OUT, OUT(type(g1)) + OUT(type(g2)))
        self.match(Generics.G1, type(g1))
        self.match(Generics.G2, type(g2))
        super().__init__(g1._shape)
        self.g1 = g1
        self.g2 = g2


class TensorGrid(Grid):
    """
    Represents a grid with a tensor object.
    Last dimension of the tensor matches the OUT generics.
    len(shape) - 1 matches the IN.
    """
    __aliases__ = {
        Generics.B: FixArray[Type.FLOAT32, OUT]
    }
    ptr: Deferred[Type.UINT64]
    __code__ = 'shaders/grids/tensor_grid.h'

    def __init__(self, data: _typing.Any | RDVObject):
        if not isinstance(data, RDVObject):
            data = wrap(data)
        self.match(IN, len(data.shape) - 1)
        self.match(OUT, data.shape[-1])
        super().__init__(data.shape[:-1])
        self.ptr = data


class TensorGridBarrier(TensorGrid):
    """
    TensorGrid.__barrier_type__ (see _BarrierDeclaration / `grid >>
    barrier(...)` below) - the default barrier target: allocates its own
    tensor sized from `grid`, then records `grid`'s own dispatch (via a
    dedicated GridEvaluator) as a compute dependency, so this instance's own
    data is only actually filled the first time *it* gets dispatched (as
    part of whatever reads from it) - never eagerly at composition time. See
    Node.append_computation/has_compute_dependency/
    contains_compute_dependencies.
    """

    def __init__(self, grid, **kwargs):
        assert not kwargs, f"TensorGridBarrier takes no extra arguments, got {kwargs!r}"
        output_t = RDVObject(_vk.tensor([*grid._shape, OUT(type(grid))], Type.FLOAT32, location=MemoryLocation.DEVICE))
        super().__init__(output_t)
        self._grid_evaluator = GridEvaluator(grid, output_t.device_ptr)
        self._grid_threads = (_math.prod(grid._shape), 1, 1)

    def append_computation(self, cmd):
        append_dispatch(cmd, self._grid_evaluator, self._grid_threads)


TensorGrid.__barrier_type__ = TensorGridBarrier


class MajorantGrid(Grid, Generic[Generics.G]):
    """
    Wraps `grid` at a new (typically coarser) `shape`: each output voxel is
    the elementwise maximum over every fine voxel a continuous-coordinate
    sampler (GridSampler/TensorSampler/GPUSampler) reading `grid` under the
    given `aligned`/`linear` convention could read from anywhere inside
    that coarse cell - see shaders/grids/majorant_grid.h for exactly which
    fine voxels that is (not simply "grid partitioned into blocks": aligned
    changes the coordinate -> voxel mapping, and linear reaches one voxel
    past a cell's own naive range at internal/outer boundaries, which
    matters most exactly where density can jump abruptly, e.g. the volume's
    own outer boundary). IN/OUT match `grid`'s own - only the resolution
    changes; the result is still a plain piecewise-constant grid.

    `aligned`/`linear` must match whatever sampler is actually built on top
    of `grid` at render time (e.g. PBVR/PBVRT's own `s`) - passing
    different values produces an invalid (possibly biased) bound.

    The typical use is building a spatially-varying majorant for
    delta-tracking volume rendering (PBVR/PBVRT instead take a single
    global float `majorant` - a low-resolution MajorantGrid, sampled along
    the ray, tightens that bound and lets the tracker skip empty space).
    """
    __code__ = "shaders/grids/majorant_grid.h"
    grid: Generics.G
    aligned: int
    linear: int

    def __init__(self, grid, shape: list[int], aligned: bool = False, linear: bool = True):
        self.match(Generics.G, type(grid))
        self.match(IN, IN(type(grid)))
        self.match(OUT, OUT(type(grid)))
        super().__init__(shape)
        self.grid = grid
        self.aligned = 1 if aligned else 0
        self.linear = 1 if linear else 0


class DilatedMaxPoolGrid(Grid, Generic[Generics.G]):
    """
    Sliding-window max, same resolution as g: each output voxel is the max
    over the 3^IN neighborhood at offsets {-dilation, 0, +dilation} per
    axis (boundary-clamped - the volume's own edge value repeats past the
    border). dilation=1 is a dense (adjacent) 3x3x3; dilation>1 is the "a
    trous" (dilated) variant sphere_radius_field uses to grow its own
    guaranteed radius exponentially instead of linearly, at the same
    O(3^IN) cost per level regardless of dilation.
    """
    g: Generics.G
    dilation: int
    __code__ = "shaders/grids/dilated_max_pool_grid.h"

    def __init__(self, g, dilation: int = 1) -> None:
        G_ = type(g)
        assert OUT(G_) == 1, "DilatedMaxPoolGrid expects a scalar (OUT=1) grid."
        assert dilation >= 1, "dilation must be at least 1."
        self.match(IN, IN(G_))
        self.match(OUT, 1)
        self.match(Generics.G, G_)
        super().__init__(g._shape)
        self.g = g
        self.dilation = dilation


class SphereRadiusInit(Grid, Generic[Generics.G]):
    """(majorant, radius) per voxel, majorant taken as-is from g - the
    state sphere_radius_field's own growth chain starts from. Used with
    g = the level-1 dense 3x3x3 pool and radius = 1 (never 0 - see
    sphere_radius_field's own docstring for why level 1 is always
    accepted unconditionally, not put through SphereRadiusMerge's growth
    test like every later level is)."""
    g: Generics.G
    radius: float
    __code__ = "shaders/grids/sphere_radius_init.h"

    def __init__(self, g, radius: float = 0.0) -> None:
        G_ = type(g)
        assert OUT(G_) == 1
        self.match(IN, IN(G_))
        self.match(OUT, 2)
        self.match(Generics.G, G_)
        super().__init__(g._shape)
        self.g = g
        self.radius = radius


class SphereRadiusMerge(Grid, Generic[Generics.C, Generics.M]):
    """
    One "a trous" growth step of sphere_radius_field's own construction:
    chosen (Grid[IN, 2], the (majorant, radius) state carried in from the
    previous level) and candidate (Grid[IN, 1], this level's own dilated
    max-pool result - see DilatedMaxPoolGrid) combine into an updated
    (majorant, radius) pair per voxel - candidate's own bound is only
    actually adopted (growing this voxel's own radius to new_radius) where
    doing so doesn't inflate the majorant by more than
    1 / (voxel_size * new_radius * density_scale); voxels that fail this
    test keep whatever (majorant, radius) they already had. Since
    candidate only ever grows (monotonically) from here on, a voxel that
    fails this test once naturally keeps failing it against every later,
    even-larger candidate too - no separate "frozen" bookkeeping needed,
    the unchanged (majorant, radius) state carries the freeze forward by
    itself.
    """
    chosen: Generics.C
    candidate: Generics.M
    new_radius: float
    voxel_size: float
    density_scale: float
    __code__ = "shaders/grids/sphere_radius_merge.h"

    def __init__(self, chosen, candidate, new_radius: float, voxel_size: float, density_scale: float = 1.0) -> None:
        C_ = type(chosen)
        M_ = type(candidate)
        self.match(IN, IN(C_))
        self.match(OUT, 2)
        self.match(Generics.C, C_)
        self.match(Generics.M, M_)
        super().__init__(chosen._shape)
        self.chosen = chosen
        self.candidate = candidate
        self.new_radius = new_radius
        self.voxel_size = voxel_size
        self.density_scale = density_scale


def sphere_radius_field(majorant_grid, levels: int = 4, density_scale: float = 1.0) -> 'TensorGrid':
    """
    Builds, from a fine per-voxel majorant grid (a scalar, OUT=1 grid -
    e.g. MajorantGrid's own output), a piecewise-constant "sphere radius
    field" via an "a trous" (dilated) max-pool pyramid, at a constant
    O(3^IN) cost per level, with *dilation* growing linearly (by exactly 1
    voxel each level, not doubling - trading a lower max reach for a given
    `levels` for finer-grained intermediate radius options, letting
    SphereRadiusMerge's own growth test adapt more gradually to the data
    instead of jumping straight from radius 1 to a much larger one):

    - Level 1 is a dense (adjacent) 3x3x3 max-pool of the fine grid, with
      radius 1 (one full voxel width - not 1.5: the guarantee has to hold
      from *any* point inside the source voxel, not just its center, and
      the worst case is a query sitting right at that voxel's own wall,
      which the 3x3x3 block still covers by exactly one more voxel width
      in every direction). Level 1 is always accepted unconditionally, not
      put through SphereRadiusMerge's own growth test like every later
      level is - a voxel's own chosen radius must never be 0 (0 isn't a
      usable free-flight step at all - SphereMajorantField would report a
      near-zero safe distance there, forcing the tracer into the same
      many-tiny-steps pathology a per-cell VoxelMajorantField already has,
      defeating the entire point of building this in the first place), and
      there's never a reason to reject level 1 specifically: it costs
      nothing more than VoxelMajorantField's own single-cell bound already
      does, just to grow one extra voxel of guaranteed radius for free.
    - Level L (L >= 2) is a 3x3x3 max-pool of level (L-1)'s own grid, at
      dilation L (i.e. sampling neighbors L voxels away, "a trous" - "with
      holes" - instead of adjacent ones, dilation itself growing by
      exactly 1 each level). Its own TRUE radius (both reported, and used
      by SphereRadiusMerge's own growth test) is *not* just the dilation:
      the candidate is built *from* level (L-1)'s own values, which
      already carry level (L-1)'s own true reach, so the candidate's own
      box guarantee chains contiguously with its neighbors' (as long as
      dilation never exceeds 2x the previous level's own true radius - for
      dilation growing by exactly 1 each level that's always true, L <=
      2*(L-1) for every L >= 2) out to radius(L) = radius(L-1) + L -
      giving a triangular sequence (1, 3, 6, 10, 15, ... for L = 1..5)
      instead of either a purely linear one (which would under-credit how
      far a candidate actually reaches, making SphereRadiusMerge's own
      test reject growth *sooner* than the candidate's real coverage
      would justify - voxels several levels away from a bright feature
      would "see" it, and freeze, well before their own reported radius
      suggests they should) or the doubling-dilation exponential one this
      replaced (radius 2^L - 1: 1, 3, 7, 15, ...). Unlike level 1, growing
      to level L here is only worth it if the majorant that comes with it
      hasn't grown too much (more null collisions per unit distance once a
      tracer actually queries this deep) - see SphereRadiusMerge for the
      exact per-voxel growth test (1 / (voxel_size * radius *
      density_scale), voxel_size the grid's own smallest per-axis size for
      conservativeness): a voxel simply keeps whatever (majorant, radius)
      it last accepted once growing
      further stops being worth it, even though the underlying pyramid
      keeps growing for every other, still-growing voxel regardless.

    OUT is 2: [0] the chosen majorant, [1] the chosen radius (voxel-index
    units, always >= 1 - multiply by 2/shape[axis] for an object-space
    distance along a given axis, same convention as MajorantGrid's own
    coordinate mapping). density_scale should match whatever scale is
    actually applied to the extinction field at render time (e.g.
    PBVR/PBVRT's own transmittance_rt density_scale).
    """
    if not isinstance(majorant_grid, Grid):
        majorant_grid = TensorGrid(majorant_grid)
    assert OUT(type(majorant_grid)) == 1, "sphere_radius_field expects a scalar (OUT=1) majorant grid."
    assert levels >= 1

    shape = majorant_grid._shape
    voxel_size = min(2.0 / n for n in shape)

    level1_grid = TensorGrid(DilatedMaxPoolGrid(majorant_grid, 1)())
    chosen_grid = TensorGrid(SphereRadiusInit(level1_grid, 1.0)())
    current_grid = level1_grid
    dilation = 2
    true_radius = 1.0  # level 1's own true (== reported) reach
    for level in range(2, levels + 1):
        candidate_grid = TensorGrid(DilatedMaxPoolGrid(current_grid, dilation)())
        # The candidate's own actual coverage is dilation + the previous
        # level's own true reach (the chaining argument - see this
        # function's own docstring) - not just `dilation` alone: the
        # candidate is built *from* level (L-1)'s own values, which
        # already carry that reach, so it "sees" a bright feature exactly
        # that much sooner than dilation alone would suggest. Testing
        # (and reporting) the true cumulative reach instead keeps the
        # accept/reject decision honest about what the candidate actually
        # covers - dilation itself still grows by exactly 1 each level
        # (the "linear, for finer adjustment" part), it's only radius that
        # now grows faster (triangular: 1, 3, 6, 10, 15, ...) to match.
        true_radius = true_radius + dilation
        chosen_grid = TensorGrid(SphereRadiusMerge(chosen_grid, candidate_grid, true_radius, voxel_size, density_scale)())
        current_grid = candidate_grid
        # dilation += 1
    return chosen_grid


class RegularGrid(Grid):
    """
    Turns index k (in [0, shape[dim]-1]) into a float coordinate in
    [-1, 1] along that dim - no backing data, just an index -> coordinate
    map (OUT == IN). `mode` selects the point-placement convention:
      0 - corner:   shape[dim]-1 voxels fit the volume; grid points are
                     the voxel corners (index 0 -> -1, index shape[dim]-1
                     -> 1, exactly like GridSampler/TensorSampler's own
                     `aligned=True`).
      1 - center:   shape[dim] voxels fit the volume; grid points are
                     voxel centers (like `aligned=False`).
      2 - jittered: like center, but a uniform-random point inside the
                     voxel instead of its exact center (a fresh point
                     every dispatch - see RandomLibrary).
    Output order is reversed relative to _input's own index order (dim0
    slowest .. dim(IN-1) fastest), matching every other rdv sampler's own
    fastest-axis-first coordinate convention.
    """
    __code__ = "shaders/grids/regular_grid.h"
    mode: int  # 0 - corner, 1 - center, 2 - jittered
    __requires__ = [
        RandomLibrary
    ]

    def __init__(self, shape: list[int], mode: int):
        self.match(IN, len(shape))
        self.match(OUT, len(shape))
        super().__init__(shape)
        self.mode = mode


class GridEvaluator(Main, Generic[IN, OUT, Generics.G]):
    """
    output_ptr is a plain device pointer, not Deferred - see MapEvaluator
    (same rationale/measured win) and Grid.__call__, which caches one
    GridEvaluator per concrete Grid class and writes g/output_ptr directly
    onto it on every later call.
    """
    __aliases__ = {
        Generics.OB: FixArray[Type.FLOAT32, OUT]
    }
    g: Generics.G
    output_ptr: Single[Type.UINT64]
    __local_size__ = (32, 1, 1)
    __code__ = "shaders/grids/grid_evaluator.h"

    def __init__(self, grid, output: RDVObject | int) -> None:
        G = type(grid)
        self.match(IN, IN(G))
        self.match(OUT, OUT(G))
        self.match(Generics.G, G)
        self.output_ptr = output
        self.g = grid

    def set_direct(self, g, output_ptr: int) -> None:
        """Writes g/output_ptr directly - see _direct_offsets."""
        acc = self.__accessor__
        acc.write_uint64(0, resolve_ptr(g))
        acc.write_uint64(8, output_ptr)


class BatchGridEvaluator(Main, Generic[IN, OUT, Generics.G]):
    """
    input_ptr/output_ptr are plain device pointers, not Deferred - see
    MapEvaluator/GridEvaluator.
    """
    __aliases__ = {
        Generics.IB: FixArray[Type.INT32, IN],
        Generics.OB: FixArray[Type.FLOAT32, OUT]
    }
    g: Generics.G
    input_ptr: Single[Type.UINT64]
    output_ptr: Single[Type.UINT64]
    __local_size__ = (32, 1, 1)
    __code__ = "shaders/grids/batch_grid_evaluator.h"

    def __init__(self, grid, input: RDVObject | int, output: RDVObject | int) -> None:
        G = type(grid)
        self.match(IN, IN(G))
        self.match(OUT, OUT(G))
        self.match(Generics.G, G)
        self.input_ptr = input
        self.output_ptr = output
        self.g = grid

    def set_direct(self, g, input_ptr: int, output_ptr: int) -> None:
        """Writes g/input_ptr/output_ptr directly - see _direct_offsets."""
        acc = self.__accessor__
        acc.write_uint64(0, resolve_ptr(g))
        acc.write_uint64(8, input_ptr)
        acc.write_uint64(16, output_ptr)


# class Capture(Main, Generic[IN, OUT, Generics.G, Generics.M, Generics.V]):
#     """
#     Represents a capture of a map with the values coming from a grid: one
#     thread per grid cell (dispatched the same way as GridEvaluator, in
#     row-major order), forward(g, indices, values) samples the grid's own
#     V-sized values (V == the grid's own OUT, and also the map's own IN - a
#     grid position's sampled values are exactly the map's input), then
#     forward(m, values, result) maps them into this Capture's own OUT-sized
#     result. `samples` repeats this grid-then-map round trip that many times
#     per cell and stores the average - meaningful whenever g and/or m are
#     stochastic (e.g. Uniform/Normal).
#     """
#     __aliases__ = {
#         Generics.OB: FixArray[Type.FLOAT32, OUT]
#     }
#     g: Generics.G
#     m: Generics.M
#     output_ptr: Single[Type.UINT64]
#     samples: int
#     __local_size__ = (32, 1, 1)
#     __code__ = "shaders/grids/capture.h"
#
#     def __init__(self, grid, map, samples: int, output: RDVObject | int) -> None:
#         G = type(grid)
#         M = type(map)
#         assert OUT(G) == IN(M), \
#             "Grid's OUT must match Map's IN - the grid's sampled values are the map's input."
#         self.match(IN, IN(G))
#         self.match(OUT, OUT(M))
#         self.match(Generics.G, G)
#         self.match(Generics.M, M)
#         self.match(Generics.V, OUT(G))
#         self.g = grid
#         self.m = map
#         self.samples = samples
#         self.output_ptr = output
#
#     def set_direct(self, g, m, samples: int, output_ptr: int) -> None:
#         """Writes g/m/samples/output_ptr directly - see _direct_offsets."""
#         # g_off, m_off, samples_off, output_off = _direct_offsets(self, 'g', 'm', 'samples', 'output_ptr')
#         acc = self.__accessor__
#         acc.as_uint64s[0] = resolve_ptr(g)
#         acc.as_uint64s[1] = resolve_ptr(m)
#         acc.as_uint64s[2] = output_ptr
#         acc.as_ints[6] = samples
#
#
# class BatchCapture(Main, Generic[IN, OUT, Generics.G, Generics.M, Generics.V]):
#     """
#     Like Capture, but samples the grid only at requested indices (see
#     BatchGridEvaluator) instead of over the whole grid domain.
#     """
#     __aliases__ = {
#         Generics.IB: FixArray[Type.INT32, IN],
#         Generics.OB: FixArray[Type.FLOAT32, OUT],
#     }
#     g: Generics.G
#     m: Generics.M
#     input_ptr: Single[Type.UINT64]
#     output_ptr: Single[Type.UINT64]
#     samples: int
#     __local_size__ = (32, 1, 1)
#     __code__ = "shaders/grids/batch_capture.h"
#
#     def __init__(self, grid, map, samples: int, input: RDVObject | int, output: RDVObject | int) -> None:
#         G = type(grid)
#         M = type(map)
#         assert OUT(G) == IN(M), \
#             "Grid's OUT must match Map's IN - the grid's sampled values are the map's input."
#         self.match(IN, IN(G))
#         self.match(OUT, OUT(M))
#         self.match(Generics.G, G)
#         self.match(Generics.M, M)
#         self.match(Generics.V, OUT(G))
#         self.g = grid
#         self.m = map
#         self.samples = samples
#         self.input_ptr = input
#         self.output_ptr = output
#
#     def set_direct(self, g, m, samples: int, input_ptr: int, output_ptr: int) -> None:
#         """Writes g/m/samples/input_ptr/output_ptr directly - see _direct_offsets."""
#         # g_off, m_off, samples_off, input_off, output_off = _direct_offsets(
#         #     self, 'g', 'm', 'samples', 'input_ptr', 'output_ptr')
#         acc = self.__accessor__
#         acc.as_uint64s[0] = resolve_ptr(g)
#         acc.as_uint64s[1] = resolve_ptr(m)
#         acc.as_uint64s[2] = input_ptr
#         acc.as_uint64s[3] = output_ptr
#         acc.as_ints[8] = samples


# ================ TENSORS =====================


T = Generics.T


BLOCK_ROWS = 4
BLOCK_CHANNELS = 8
BLOCK_SIZE = BLOCK_ROWS * BLOCK_CHANNELS  # 32 - matches a 32-lane workgroup 1:1


class TensorBase(Component, Generic[IN]):
    """
    Base class for tensors. Like a Grid, a tensor represents a map from
    index locations over its own shape - but a tensor's own forward()
    always works in fixed 4x8 (BLOCK_ROWS x BLOCK_CHANNELS = 32) blocks:
    _input[IN] (IN is always 2 here - rows, channels) is a *block* index,
    not an element index, and _output[32] holds that block's 32 values,
    row-major (_output[r*8+c], r in [0,4), c in [0,8)) - one thread per
    block instead of one thread per element, cutting dispatch thread
    count by 32x versus the old one-scalar-per-thread design. shape is
    still in *element* units (matches Grid/Tensor's own convention, and
    what .as_type()/as_tensor() actually return) - both dims must
    currently be exact multiples of BLOCK_ROWS/BLOCK_CHANNELS (no partial-
    block padding yet). See ReLUTensor/LinearTensor (rdv.domain.nn, the
    latter also caching the weight tile each block needs) and Tensor
    (below, the buffer-backed leaf).
    """
    __abstract__ = True
    __requires__ = [
        GridLibrary[IN]
    ]
    __precode__ = "void forward(RDV_NODE node, int _input[IN], out float _output[32]);"
    shape: Array[IN, int]

    __evaluator__ = None
    """Cached evaluator dispatching the full tensor (see TensorEvaluator)."""
    _shape = None

    def __init__(self, shape: list[int]):
        # No shape constraints here - Tensor (this family's own leaf) is
        # also used as a plain, arbitrary-rank buffer-backed data
        # container that never goes through forward()/block dispatch at
        # all (e.g. TensorSampler's own t field, or test_tensor_reinterpret
        # .py's ReinterpretCopy, both reading via get_value() directly) -
        # see __call__ below for where the 2D/4x8-multiple requirement
        # this family's own forward() convention actually needs is
        # enforced, only for whatever actually dispatches through it.
        self._shape = shape
        self.shape = shape

    def __call__(self):
        """Evaluates every block of this tensor (one thread per 4x8 block,
        in row-major block order) into a fresh (or cached, if this exact
        shape was already dispatched before) device tensor of this
        tensor's own (element) shape - equivalent to Grid.__call__'s own
        no-args form, just 32 elements per thread instead of OUT."""
        cls = type(self)
        shape = list(self._shape)
        assert len(shape) == 2, "TensorBase.__call__ (block-based forward()) only supports 2D (rows, channels) shapes"
        assert shape[0] % BLOCK_ROWS == 0, f"rows ({shape[0]}) must be a multiple of {BLOCK_ROWS}"
        assert shape[1] % BLOCK_CHANNELS == 0, f"channels ({shape[1]}) must be a multiple of {BLOCK_CHANNELS}"
        evaluator = cls.__evaluator__
        if evaluator is None or evaluator._cached_output_shape != shape:
            output_t = RDVObject(_vk.tensor(shape, Type.FLOAT32, location=MemoryLocation.DEVICE))
            if evaluator is None:
                evaluator = TensorEvaluator(self, output_t.device_ptr)
                cls.__evaluator__ = evaluator
            evaluator._cached_output = output_t
            evaluator._cached_output_shape = shape
        else:
            output_t = evaluator._cached_output
        # Plain field assignment (not GridEvaluator/MapEvaluator's own
        # direct-pointer-write set_direct fast path) - g's own field type
        # here isn't always a pointer the way Grid/Map's own G always is:
        # a leaf Tensor is deliberately inlined wherever referenced, never
        # behind a pointer (see Tensor's own docstring), so a raw
        # resolve_ptr(g)-and-write-bytes shortcut would be wrong for it.
        # Plain assignment goes through the normal field machinery, which
        # already knows how to do either (inline-embed or resolve-and-
        # write-pointer) correctly per g's own concrete type.
        evaluator.g = self
        evaluator.output_ptr = output_t.device_ptr
        evaluator.shape = shape
        block_count = (shape[0] // BLOCK_ROWS) * (shape[1] // BLOCK_CHANNELS)
        dispatch(evaluator, block_count)
        return output_t

    def __rshift__(self, other):
        if isinstance(other, _BarrierDeclaration):
            return other.create_barrier(self)
        return NotImplemented

    def __rlshift__(self, other):
        if isinstance(other, _BarrierDeclaration):
            return other.create_barrier(self)
        return NotImplemented


class TensorEvaluator(Main, Generic[IN, Generics.G]):
    """output_ptr is a plain device pointer, not Deferred - see
    GridEvaluator (same rationale). Dispatches one thread per 4x8 block
    (see TensorBase), writing that block's 32 values out to their real
    (row, channel) positions in the element-shaped output buffer. Keeps
    its own copy of shape (element units, set once from g's own _shape at
    construction) rather than reading g.shape back out in the shader -
    unlike Grid (always pointer-referenced), a TensorBase's g can be
    either inline (a leaf Tensor) or pointer-referenced (ReLUTensor/
    LinearTensor/TensorBarrier), so a single hardcoded deref(g).shape
    (correct for one, wrong for the other) can't work for both."""
    __aliases__ = {
        Generics.F1: FixArray[Type.FLOAT32, 1]
    }
    g: Generics.G
    output_ptr: Single[Type.UINT64]
    shape: Array[IN, int]
    __local_size__ = (32, 1, 1)
    __code__ = "shaders/core/tensor_evaluator.h"

    def __init__(self, g, output: RDVObject | int) -> None:
        G = type(g)
        self.match(IN, IN(G))
        self.match(Generics.G, G)
        self.output_ptr = output
        self.g = g
        self.shape = list(g._shape)


class Tensor(TensorBase, Generic[T]):
    # deferred_buffer is a real, accessor-backed field (see
    # Node.__getattr__/__setattr__) - matching shaders/core/tensor.h's own
    # `struct RDV_NODE { D deferred_buffer; int shape[IN]; }`, a Tensor is
    # inlined wherever it's referenced (like a Struct), never behind a
    # pointer. D is declared as an alias in terms of this type's own T -
    # _NodeMeta.__new__ resolves an alias against its owning class's own
    # generics fresh at every specialization (Tensor[T=float].D and
    # Tensor[T=int].D differ), so D is #define'd correctly and
    # automatically by _build_code, the same way it already does for every
    # other alias/generic.
    __aliases__ = {
        Generics.D: Deferred[Buffer[T]]
    }
    deferred_buffer: Deferred[Buffer[T]]

    @classmethod
    def _generate_code(cls):
        return '#include "shaders/core/tensor.h"'

    @classmethod
    def _fields_layout_decription(cls):
        # inlined wherever referenced (see shaders/core/tensor.h) - same recursive
        # layout its own __fields__ already describes, matching how GLSL
        # codegen treats it.
        return _layout_of(cls.__fields__)

    def _fields_value(self):
        return self.__accessor__.rdv_values()

    def __init__(self, data: RDVObject | Deferred[RDVObject], shape: list[int], type: Type = None) -> None:
        self.match(IN, len(shape))
        if type is not None:
            # only used to *infer* T when it isn't already bound (e.g. plain
            # Tensor(data, shape, Type.X)) - never re-checked against an
            # already-concrete T, so replaying these same args (see
            # Node.cast) against a further-specialized class - e.g. a
            # cast()-reinterpreted Tensor[Y, IN] sharing data's own bytes as
            # a different scalar type - isn't rejected as a conflict.
            self.match(T, type)
        self._wrap_data = data
        self.deferred_buffer = data
        super().__init__(shape)

    @classmethod
    def from_data(cls, data):
        wrap_data = wrap(data)
        return Tensor(wrap_data, wrap_data.shape, wrap_data.scalar_type)

    def as_type(self, backend):
        """
        Exports this tensor's own data in backend, reinterpreted (bit for
        bit, never converted - see Node.cast) as this tensor's own T and
        reshaped to its own shape - not necessarily whatever raw type/shape
        the wrapped data natively had (e.g. after a cast() to a different
        scalar type, or when the wrapped data is a flat buffer sharing bytes
        with an N-dimensional tensor of a different rank).
        """
        raw = self._wrap_data.as_tensor(backend)
        own_type = T(self)
        if own_type is not None and own_type != self._wrap_data.scalar_type:
            raw = _reinterpret(raw, backend, own_type)
        return _reshape(raw, backend, list(self.shape))


def tensor(*shape: int, scalar_type: Type = Type.FLOAT32, location: MemoryLocation | None = None) -> Tensor:
    """Creates a new tensor of shape on the current device, backed by a vulky
    (vk) tensor and wrapped as a rdv Data object. Defaults to host-visible memory
    (like every other buffer rdv itself allocates - _build_accessor,
    DeferredPtrBase.buffer(), ...), since a device-only tensor has no host/
    CUDA-visible pointer and can't be exported through DLPack (what
    Tensor.host()/.device() use) at all."""
    if location is None:
        location = _vk.MemoryLocation.HOST
    data = RDVObject(_vk.tensor(list(shape), scalar_type, location))
    # T, IN pre-bound via the subscript rather than passed to __init__ (which
    # would otherwise become part of __init_args__ - see Node.cast) - so a
    # tensor()-built Tensor can still be cast()-reinterpreted as a different
    # scalar type later, sharing the same underlying data.
    return Tensor[Generics.T:scalar_type, IN:len(shape)](data, list(shape))


class TensorBarrier(TensorBase):
    """Tensor.__barrier_type__ (see _BarrierDeclaration / `t >>
    barrier[Tensor]` - Tensor's own docstring for why this must be
    written explicitly, unlike a plain Grid's `>> barrier`) - the default
    barrier target for the Tensor family: allocates its own buffer sized
    from g, then records g's own dispatch (via a dedicated TensorEvaluator)
    as a compute dependency, so this instance's own data is only actually
    (re)filled once *it* (or a dispatch depending on it) actually runs -
    see Node.append_computation/has_compute_dependency/
    contains_compute_dependencies, and TensorGridBarrier for the same
    pattern applied to Grid.

    Pointer-referenced (Component's own default), unlike Tensor's own
    leaf (deliberately inlined wherever referenced) -
    _collect_compute_dependencies only ever finds a dependency through a
    field whose value getattr(node, name) returns is the original Node
    instance itself (identity-tracked via id()); for an *inlined* field
    that same getattr instead returns a live Accessor view into the
    inline bytes, not the original object, so an inlined barrier's own
    append_computation would silently never run - only actually
    observable once a Tensor-family node (unlike Grid, always pointer-
    referenced) carries its own compute dependency, which a plain Tensor
    leaf never does but this barrier target always does. Always float32
    (matches every other Tensor-family node in this codebase -
    LinearTensor/TensorEvaluator's own output)."""
    __code__ = "shaders/core/tensor_barrier.h"
    deferred_buffer: Deferred[Buffer[Type.FLOAT32]]

    def __init__(self, g, **kwargs):
        assert not kwargs, f"TensorBarrier takes no extra arguments, got {kwargs!r}"
        shape = list(g._shape)
        assert len(shape) == 2, "TensorBarrier (block-based forward()) only supports 2D (rows, channels) shapes"
        assert shape[0] % BLOCK_ROWS == 0, f"rows ({shape[0]}) must be a multiple of {BLOCK_ROWS}"
        assert shape[1] % BLOCK_CHANNELS == 0, f"channels ({shape[1]}) must be a multiple of {BLOCK_CHANNELS}"
        self.match(IN, IN(type(g)))
        output_t = RDVObject(_vk.tensor(shape, Type.FLOAT32, location=MemoryLocation.DEVICE))
        self.deferred_buffer = output_t
        self._output_t = output_t
        self._tensor_evaluator = TensorEvaluator(g, output_t.device_ptr)
        block_count = (shape[0] // BLOCK_ROWS) * (shape[1] // BLOCK_CHANNELS)
        self._tensor_threads = (block_count, 1, 1)
        # Explicit upstream-barrier chain, walked here (over the plain
        # Python g/.g attribute references, still reliable at this point -
        # see _collect_tensor_barriers) rather than relying on dispatch()'s
        # own _collect_compute_dependencies for this: that walk only goes
        # one barrier deep reliably (each TensorBarrier's own dependency is
        # only reachable through its _tensor_evaluator - a plain Python
        # attribute, not a __fields__ entry - so a *chain* of 2+ barriers
        # silently only ever re-runs the last one; confirmed by testing a
        # 3-linear-layer chain, which desyncs past the first
        # relu+barrier). refresh() below replays this whole list itself
        # instead, in order, so any chain length works.
        self._upstream_barriers = []
        _collect_tensor_barriers(g, set(), self._upstream_barriers)
        super().__init__(list(g._shape))

    def append_computation(self, cmd):
        append_dispatch(cmd, self._tensor_evaluator, self._tensor_threads)

    def refresh(self):
        """Re-runs this barrier's own dispatch (and, first, every upstream
        barrier stage it depends on, in order - see _collect_tensor_barriers)
        directly - use this (not __call__, which would redundantly evaluate
        this tensor *again* into a separate, fresh buffer) to refresh this
        tensor's own data on demand, e.g. after rebinding a deferreds(...)-
        bound upstream input to new data on an already-built pipeline."""
        for b in self._upstream_barriers:
            dispatch(b._tensor_evaluator, *b._tensor_threads)
        dispatch(self._tensor_evaluator, *self._tensor_threads)


    def as_type(self, backend):
        """See Tensor.as_type - always float32 here, so no reinterpret,
        just a reshape to this tensor's own shape. Refreshes first (see
        refresh()) - call refresh() directly instead, ahead of time, to
        skip the redundant extra one in a tight loop that already calls
        it explicitly (e.g. right after rebinding deferred input)."""
        self.refresh()
        return _reshape(self._output_t.as_tensor(backend), backend, list(self.shape))


def _collect_tensor_barriers(node, seen_ids, out):
    """Depth-first walk of node's own Tensor-family graph, over plain
    Python g/._tensor_evaluator.g attribute references (reliable object
    identity - unlike the GLSL __fields__-based walk
    _collect_compute_dependencies does, which only reaches one barrier
    deep for this family - see TensorBarrier's own refresh()), collecting
    every distinct (by identity) upstream TensorBarrier in dependency
    order (postorder, so an earlier stage lands before whatever reads it)."""
    if isinstance(node, TensorBarrier):
        if id(node) in seen_ids:
            return
        _collect_tensor_barriers(node._tensor_evaluator.g, seen_ids, out)
        seen_ids.add(id(node))
        out.append(node)
    elif isinstance(node, TensorBase) and 'g' in type(node).__fields__:
        _collect_tensor_barriers(node.g, seen_ids, out)


Tensor.__barrier_type__ = TensorBarrier


class TensorOperation(Component):
    """
    Self-contained, self-materializing tensor operation - each subclass
    stores whatever input tensors it needs directly, as plain
    Deferred[Buffer[FLOAT32]] fields read as flat, already-materialized
    buffers (never recursively composed the way TensorBase's own
    forward() chaining works - no G generic, no inline-vs-pointer
    ambiguity, no cooperative occupancy/uniformity bookkeeping needed:
    each op picks whatever load/shared-memory strategy actually suits its
    own computation, entirely on its own terms). Has its own (element)
    shape and a dedicated result buffer it writes into, and dispatches
    itself - its own evaluator (TensorOpEvaluator), one thread per 4x8
    tile (see BLOCK_ROWS/BLOCK_CHANNELS) - eagerly, right in __init__: by
    the time construction returns, .result already holds this op's own
    output, so whatever's built from it next just reads that buffer
    directly. No separate barrier/materialization step needed anywhere in
    a chain of these - "automatic" in the sense that there's simply
    nothing left to forget to add.
    """
    __abstract__ = True
    __precode__ = """
#define TENSOR_OP_UNROLL [[unroll]]
void forward(RDV_NODE node);
"""
    shape: Array[2, int]
    result: Buffer[Type.FLOAT32]

    _evaluator = None
    """This instance's own evaluator (see TensorOpEvaluator) - one per
    *instance*, not shared per concrete op type: dispatch() caches a
    command buffer per main_node instance with that instance's own
    device pointer baked in as a push constant at record time, so two
    different op instances (of a type with no shape-driven generic to
    tell them apart, e.g. two ReLUTensorOp stages of one pipeline) taking
    turns through one shared, mutable evaluator object - even carefully
    repointing its own .op field back to self before every dispatch -
    still each hit that same one cached recording, built for whichever
    instance evaluate() last actually reran the dispatch machinery's own
    upload path for. A dedicated evaluator per instance sidesteps that
    entirely; the underlying *pipeline* (this type's own compiled shader)
    is still shared/cached normally, just not this mutable wrapper object."""
    _shape = None
    _upstream = None
    """The TensorOperation this one directly reads from (set by whichever
    subclass's own __init__ called _as_op_input), or None if this op's own
    input was raw external data - see evaluate_chain()."""
    _extra_upstreams = ()
    """Any *additional* upstream TensorOperation(s) beyond _upstream - a
    multi-input op like ConcatTensorOp reads from two, not one; see
    evaluate_chain()/_collect_tensor_ops."""

    _TILE_ROWS = BLOCK_ROWS
    _TILE_COLS = BLOCK_CHANNELS
    """Per-tile coverage (element units) - shape must be an exact multiple
    of both. Overridable per subclass - a cooperative-matrix op (e.g.
    LinearTensorOpFast) tiles 32x16, not the default 4x8."""
    _THREADS_PER_TILE = 1
    """How many actual GPU threads cooperate on one tile - 1 for the
    default one-thread-one-tile design; a cooperative-matrix op needs the
    whole 32-lane workgroup collectively producing one tile, so 32 - see
    _dispatch_thread_count() and forward()'s own tile_id/THREADS_PER_TILE
    math for how each subclass turns that back into its own tile index."""
    _WORKGROUP_SIZE = 32
    """This op's own dispatch workgroup size (see TensorOpEvaluator's own
    __local_size__, which reads this back per bound op type) - fixed at 32
    (independent of _THREADS_PER_TILE) for every op here except
    LinearTensorOpSimple, which packs _SUBGROUPS_PER_TILE subgroups into
    one workgroup so several can cooperate on the same staged x-tile at
    once."""

    @classmethod
    def _dispatch_thread_count(cls, shape):
        tile_count = (shape[0] // cls._TILE_ROWS) * (shape[1] // cls._TILE_COLS)
        return tile_count * cls._THREADS_PER_TILE

    def __init__(self, shape: list[int]):
        if type(self).__is_generic__:
            # First of _NodeMeta.__call__'s own two construction passes -
            # self.match(...) calls a subclass's own __init__ already made
            # (e.g. LinearTensorOp's INF/OUTF) only take effect (rebuilding
            # this instance from scratch against the now-concrete type)
            # *after* __init__ fully returns, so type(self) is still the
            # bare, unspecialized class here - too early to allocate a
            # buffer or dispatch against (Generics.OP couldn't bind to a
            # still-generic type either). Skip the side effects below on
            # this pass; they run for real on the guaranteed second,
            # concrete-typed pass __call__ makes right after.
            self._shape = shape
            self.shape = shape
            return
        cls = type(self)
        assert len(shape) == 2, "TensorOperation only supports 2D (rows, channels) shapes"
        assert shape[0] % cls._TILE_ROWS == 0, f"rows ({shape[0]}) must be a multiple of {cls._TILE_ROWS}"
        assert shape[1] % cls._TILE_COLS == 0, f"channels ({shape[1]}) must be a multiple of {cls._TILE_COLS}"
        self._shape = shape
        self.shape = shape
        output_t = RDVObject(_vk.tensor(shape, Type.FLOAT32, location=MemoryLocation.DEVICE))
        self.result = output_t
        self._output_t = output_t
        self._evaluator = TensorOpEvaluator(self)
        dispatch(self._evaluator, cls._dispatch_thread_count(shape))

    def as_type(self, backend):
        """Exports this op's own already-computed result in backend,
        reshaped to its own (element) shape - always float32, so no
        reinterpret, and always already up to date (see this class's own
        docstring - no refresh()/barrier needed, unlike TensorBarrier)."""
        return _reshape(self._output_t.as_tensor(backend), backend, list(self._shape))

    def evaluate(self):
        """Re-dispatches this op's own forward() over its own already-
        allocated result buffer, via its own already-built evaluator (see
        _evaluator)/pipeline - no reallocation, no Python-side rebuild -
        the fixed-shape, changing-input-data regime: pure re-evaluation
        cost only. Does *not* also refresh whatever upstream op(s) this
        one reads from - see evaluate_chain() for that."""
        dispatch(self._evaluator, type(self)._dispatch_thread_count(self._shape))

    def evaluate_chain(self):
        """Re-dispatches this op and every upstream op it (transitively)
        reads from, in dependency order (see _collect_tensor_ops) - call
        this once, on the final op of an already-built pipeline, after
        mutating a leaf input's own buffer in place (e.g. torch's own
        in-place copy_() into the same tensor a leaf op was built from),
        instead of rebuilding any of the pipeline itself."""
        ops = []
        _collect_tensor_ops(self, set(), ops)
        for op in ops:
            op.evaluate()


class TensorOpEvaluator(Main, Generic[Generics.OP]):
    """Dispatches op's own forward(), one thread per 4x8 tile - see
    TensorOperation."""
    op: Generics.OP
    __code__ = "shaders/core/tensor_op_evaluator.h"

    @_classproperty
    def __local_size__(cls):
        # Derived from the bound op's own _WORKGROUP_SIZE (default 32)
        # rather than fixed, so an op that packs several subgroups into one
        # workgroup (e.g. LinearTensorOpSimple's own _SUBGROUPS_PER_TILE)
        # gets a workgroup sized to match.
        op_cls = cls.__generics__[Generics.OP.__name__]
        return (op_cls._WORKGROUP_SIZE, 1, 1)

    def __init__(self, op) -> None:
        self.match(Generics.OP, type(op))
        self.op = op


def _as_op_input(x):
    """Normalizes x - a TensorOperation, or raw external data - to a
    (result_buffer, shape, upstream_op) triple for another TensorOperation's
    own input field (upstream_op is x itself if it's a TensorOperation, for
    evaluate_chain() to later walk back through - see _collect_tensor_ops -
    or None if x is raw external data, the first stage of a chain). x need
    not be an rdv Node at all in that raw-data case: a TensorOperation only
    ever reads its inputs as flat, already-materialized buffers (see
    TensorOperation's own docstring), never by recursively calling into
    another node's own forward()."""
    if isinstance(x, TensorOperation):
        return x.result, list(x._shape), x
    data = x if isinstance(x, RDVObject) else wrap(x)
    return data, list(data.shape), None


def _collect_tensor_ops(node, seen_ids, out):
    """Depth-first walk of node's own upstream TensorOperation chain (see
    TensorOperation._upstream/_extra_upstreams), collecting every distinct
    (by identity) op in dependency order (postorder, so an earlier stage
    lands before whatever reads it) - see TensorOperation.evaluate_chain()."""
    if id(node) in seen_ids:
        return
    if node._upstream is not None:
        _collect_tensor_ops(node._upstream, seen_ids, out)
    for up in node._extra_upstreams:
        _collect_tensor_ops(up, seen_ids, out)
    seen_ids.add(id(node))
    out.append(node)


# ================ MAPS =======================


class Map(Component, Generic[IN, OUT]):
    __precode__ = "shaders/maps/map_precode.h"
    __evaluator__ = None
    """
    Cached evaluator for this map.
    """
    __cast_cache__ = None
    """
    Per-class cache of this (still-generic) map's own cast() to a concrete
    Map[in_size, out_size], keyed by the (in_size, out_size) tuple - cast()
    replays the original constructor (see Node.cast), which for a map like
    Constant re-uploads its data to the GPU, so casting fresh on every
    __call__ is wasteful whenever the same shape recurs (e.g. every
    iteration of a dispatch loop).
    """
    def __call__(self, *args, **kwargs):
        x, = args
        in_size = x.shape[-1]
        cls = type(self)
        if cls.__is_generic__:
            key = (in_size, OUT(cls))
            if cls.__cast_cache__ is None:
                cls.__cast_cache__ = {}
            m = cls.__cast_cache__.get(key)
            if m is None:
                m = self.cast(Map[key[0], key[1]])
                if type(m).__is_generic__:
                    raise RuntimeError("Generic arguments could not be deduced from input data. You must explicitly cast.")
                cls.__cast_cache__[key] = m
            return m(*args, **kwargs)
        x_shape = list(x.shape[:-1])
        evaluator = cls.__evaluator__
        if evaluator is None or evaluator._cached_output_shape != x_shape:
            output_t = RDVObject(_vk.tensor([*x_shape, OUT(cls)], Type.FLOAT32, location=MemoryLocation.DEVICE))
            if evaluator is None:
                evaluator = MapEvaluator(self, x.device_ptr, output_t.device_ptr)
                cls.__evaluator__ = evaluator
            evaluator._cached_output = output_t
            evaluator._cached_output_shape = x_shape
        else:
            output_t = evaluator._cached_output
        if getattr(evaluator, '_cached_map', None) is not self:
            # See the same guard in Grid.__call__.
            evaluator.__accessor__._values['map'] = self
            evaluator._cached_map = self
        evaluator.set_direct(self, x.device_ptr, output_t.device_ptr)
        if kwargs:
            DeferredLibrary.bind(**kwargs)
        dispatch(evaluator, _math.prod(x_shape))
        if kwargs:
            DeferredLibrary.unbind(**kwargs)
        return output_t

    def __rshift__(self, m):
        if isinstance(m, _AverageDeclaration):
            return Average(self, m.n)
        m = m.cast(Map[IN:OUT(self)])
        s = self.cast(Map[OUT:IN(m)])
        if isinstance(s, Identity):
            return m
        if isinstance(m, Identity):
            return s
        return Composite(m, s)

    def __lshift__(self, m):
        if not isinstance(m, Map):
            return NotImplemented
        m = m.cast(Map[OUT:IN(self)])
        s = self.cast(Map[IN:OUT(m)])
        if isinstance(s, Identity):
            return m
        if isinstance(m, Identity):
            return s
        return Composite(s, m)

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __mul__(self, other):
        return Mul(self, other)

    def __rmul__(self, other):
        return Mul(other, self)

    def __sub__(self, other):
        return Sub(self, other)

    def __rsub__(self, other):
        return Sub(other, self)

    def __truediv__(self, other):
        return Div(self, other)

    def __rtruediv__(self, other):
        return Div(other, self)

    def __floordiv__(self, other):
        return Mod(self, other)

    def __rfloordiv__(self, other):
        return Mod(other, self)

    def __or__(self, other):
        other = as_map(other)
        return ConcatMap(self, other)

    def __ror__(self, other):
        return ConcatMap(other, self)

    def __getitem__(self, indices) -> 'Map':
        """
        Picks components (ints and/or slices, resolved against this map's
        own OUT) from this map's own output - e.g. m.select(0, slice(4, 6))
        picks components 0, 4, 5. Chains naturally with then():
        m.select(...).then(...).

        Resolving a slice's actual indices needs this map's own OUT - see
        Select, which now does that resolution itself (only once it's
        actually available), so this can stay a plain, eager pass-through
        even when self is still generic.
        """
        return Select(self, indices)

    def promote(self, dim: int):
        assert OUT(self) == 1
        if dim == 1:
            return self
        return Promote(self, dim)


class MapEvaluator(Main, Generic[IN, OUT, Generics.MAP]):
    """
    input_ptr/output_ptr are plain device pointers, not Deferred - Map.__call__
    caches one MapEvaluator per concrete Map class and, on every later call,
    writes the current input/output pointers directly onto it (see
    Map.__call__) instead of going through DeferredLibrary.bind()/unbind()
    (a dict lookup plus an indirect resolve() through the deferred-pointers
    SSBO on every thread). Measured ~9% faster end to end than the
    Deferred-based version (sandbox_simple_maps.py, 5000-iteration loop,
    same device/tensors): cheaper both to set from Python (a couple of
    direct field writes - see Accessor._set_leaf) and to read in the shader
    (no resolve() indirection).
    """
    map: Generics.MAP
    input_ptr: Single[Type.UINT64]
    output_ptr: Single[Type.UINT64]

    __aliases__ = {
        Generics.IB: FixArray[Type.FLOAT32, IN],
        Generics.OB: FixArray[Type.FLOAT32, OUT],
    }

    __local_size__ = (32, 1, 1)

    __code__ = "shaders/maps/map_evaluator.h"

    def __init__(self, map, input: RDVObject | int, output: RDVObject | int) -> None:
        M = type(map)
        self.match(IN, IN(M))
        self.match(OUT, OUT(M))
        self.match(Generics.MAP, M)
        self.input_ptr = input
        self.output_ptr = output
        self.map = map

    def set_direct(self, map, input_ptr: int, output_ptr: int) -> None:
        """Writes map/input_ptr/output_ptr directly - see _direct_offsets."""
        # map_off, input_off, output_off = _direct_offsets(self, 'map', 'input_ptr', 'output_ptr')
        acc = self.__accessor__
        acc.as_uint64s[0] = resolve_ptr(map)
        acc.as_uint64s[1] = input_ptr
        acc.as_uint64s[2] = output_ptr


class _BarrierDeclaration:
    __barrier_result_type__ = None
    __barrier_kwargs__ = None

    def __call__(self, *args, **kwargs):
        assert len(args) == 0, "All arguments of a barrier must be named."
        assert self.__barrier_kwargs__ is None, "Arguments of barrier can not be set twice"
        b = _BarrierDeclaration()
        b.__barrier_result_type__ = self.__barrier_result_type__
        b.__barrier_kwargs__ = kwargs
        return b

    def __getitem__(self, item):
        assert isinstance(item, type) and hasattr(item, '__barrier_type__')
        assert self.__barrier_result_type__ is None, "Result type of a barrier can not be set twice"
        b = _BarrierDeclaration()
        b.__barrier_kwargs__ = self.__barrier_kwargs__
        b.__barrier_result_type__ = item
        return b

    def create_barrier(self, grid):
        target_type = self.__barrier_result_type__ or TensorGrid
        kwargs = self.__barrier_kwargs__ or {}
        return target_type.__barrier_type__(grid, **kwargs)

    def __rrshift__(self, other):
        if not isinstance(other, Grid):
            return NotImplemented
        return self.create_barrier(other)

    def __lshift__(self, other):
        if not isinstance(other, Grid):
            return NotImplemented
        return self.create_barrier(other)


barrier = _BarrierDeclaration()
"""
Defines a barrier to treat any grid as a tensor type. For instance:
my_sampler = grid >> barrier[GPUSampler](aligned=True)
"""


# ============== COMMON MAPS ====================


class Constant(Map):
    __code__ = "shaders/maps/constant.h"
    buf_: Deferred[FixArray[Type.FLOAT32, OUT]]

    def __init__(self, data: _typing.Any):
        if not isinstance(data, (RDVObject | Deferred[RDVObject])):
            data = wrap(data)
        shape = data.shape
        self.match(OUT, shape[-1])
        self.buf_ = data


class Identity(Map):
    """f(x) = x."""
    __code__ = "shaders/maps/identity.h"

    def __init__(self):
        # match() queues an inferred value in __pending_generics__ rather
        # than binding type(self).__generics__ synchronously (see its own
        # docstring) - so Generics.OUT(self) right after the match() call
        # just below would still read the old, unbound value, not what was
        # just inferred (only visible after __init__ returns and
        # _NodeMeta.__call__ applies the pending generics) - nothing left
        # to assert here that these two calls don't already guarantee.
        self.match(Generics.IN, Generics.OUT(self))
        self.match(Generics.OUT, Generics.IN(self))


X = Identity()  # Already a generic instance to be used to spawn operations from identity.


class RefMap(Map, Generic[Generics.M]):
    map: Generics.M
    __code__ = "shaders/maps/ref_map.h"
    def __init__(self, map: Generics.M) -> None:
        self.match(Generics.IN, Generics.IN(map))
        self.match(Generics.OUT, Generics.OUT(map))
        self.match(Generics.M, type(map))
        self.map = map


class Composite(Map, Generic[Generics.F, Generics.G]):
    """f(g(x)) - g's own OUT (V) must match f's own IN, the same way
    Capture's grid-sampled values must match its map's own IN."""
    __aliases__ = {
        Generics.V: OUT(Generics.G)
    }
    f: Generics.F
    g: Generics.G
    __code__ = "shaders/maps/composite.h"

    def __init__(self, f, g) -> None:
        # If this Composite's own IN is already known at this point (e.g.
        # a replay via cast(), triggered by an outer composition that
        # already knows what IN must be - see e.g. Mul/_map_like), but g's
        # own IN isn't, push it down into g first - same idiom
        # Select.__init__ uses for m, ConcatMap.__init__ for m1/m2.
        self_in = IN(type(self))
        if self_in is not None and IN(type(g)) is None:
            g = g.cast(Map[IN: self_in])
        F_ = type(f)
        G_ = type(g)
        assert OUT(G_) == IN(F_), \
            "g's OUT must match f's IN - g's output is f's input."
        self.match(IN, IN(G_))
        self.match(OUT, OUT(F_))
        self.match(Generics.F, F_)
        self.match(Generics.G, G_)
        self.f = f
        self.g = g


class Promote(Map, Generic[Generics.F]):
    """Repeats a scalar-valued map (f's own OUT == 1) `dimension` times:
    result(x) = [f(x), f(x), ..., f(x)] (dimension entries)."""
    f: Generics.F
    __code__ = "shaders/maps/promote.h"

    def __init__(self, f, dimension: int) -> None:
        # Same "push self's already-known IN down into a still-generic
        # child" idiom Composite/Select/Average use - see BinaryOperation
        # for why this matters (f built purely from other generic-IN
        # maps, only resolvable once an outer cast() replay reveals what
        # this Promote's own IN must be).
        self_in = IN(type(self))
        if self_in is not None and IN(type(f)) is None:
            f = f.cast(Map[IN: self_in])
        F_ = type(f)
        assert OUT(F_) == 1, "f must be scalar-valued (OUT == 1) to promote."
        self.match(IN, IN(F_))
        self.match(OUT, dimension)
        self.match(Generics.F, F_)
        self.f = f


class Average(Map, Generic[Generics.M]):
    """Evaluates the wrapped map m at the same input n times and averages
    the results - e.g. to reduce variance of a noisy, unbiased per-sample
    estimator (PBVR/PBVRT) directly in the shader, instead of needing n
    separate dispatches plus a Python-side accumulate. Each of the n calls
    still draws its own random() values (see RandomLibrary - a thread's own
    rdv_rng_state keeps advancing across calls), so they're independent
    samples, not the same one repeated. See average(), the preferred way to
    build one via `m >> average(n)`."""
    m: Generics.M
    samples: int
    __code__ = "shaders/maps/average.h"

    def __init__(self, m, n: int) -> None:
        # If this Average's own IN is already known at this point (e.g. a
        # replay via cast(), triggered by an outer composition that already
        # knows what IN must be - see Sensor/Composite), but m's own IN
        # isn't, push it down into m first - same idiom Composite/Select/
        # ConcatMap all use.
        self_in = IN(type(self))
        if self_in is not None and IN(type(m)) is None:
            m = m.cast(Map[IN: self_in])
        M_ = type(m)
        self.match(IN, IN(M_))
        self.match(OUT, OUT(M_))
        self.match(Generics.M, M_)
        self.m = m
        self.samples = n


class _AverageDeclaration:
    """`m >> average(n)`/`g >> average(n)` (see Map.__rshift__/
    Grid.__rshift__, which both check for this explicitly) builds
    Average(m, n) for a Map or GridAverage(g, n) for a Grid - lets n be
    given before the thing it applies to is known, pipeline-style.
    __rrshift__ is a fallback for the same, for any Map/Grid subclass that
    doesn't (or can't) check for this explicitly."""
    def __init__(self, n: int):
        self.n = n

    def __rrshift__(self, other):
        if isinstance(other, Grid):
            return GridAverage(other, self.n)
        if isinstance(other, Map):
            return Average(other, self.n)
        return NotImplemented


def average(n: int) -> _AverageDeclaration:
    """Defines an averaging combinator to pipeline after any Map or Grid,
    e.g. radiances = view_radiances >> average(100) - averaging a Map stays
    a Map (see Average), averaging a Grid stays a Grid (see GridAverage)."""
    return _AverageDeclaration(n)


class Select(Map, Generic[Generics.M, Generics.V]):
    """Picks a fixed list of components (by index into m's own V-sized
    output) from the wrapped map m's own output - see Map.select(), the
    preferred way to build one (resolves slices for you)."""
    m: Generics.M
    sel: Array[OUT, int]
    __code__ = "shaders/maps/select.h"

    def __init__(self, m, indices) -> None:
        """
        `indices` may be a bare int/slice or a tuple mixing both, still
        unresolved against m's own OUT (see Map.__getitem__). m can still
        be generic at this point (e.g. Select built ahead of the real
        input that will eventually fix m's shape) - this resolves as much
        as it can regardless:

        - OUT (this Select's own output count) and sel (the concrete
          picked-index list) only need the *indices themselves* to be
          resolvable - true for a bare int, or a slice with explicit,
          non-negative start/stop (e.g. X[3:6]) - never m's own OUT (V).
          These bind eagerly, so e.g. `X[3:6] >> other` can already infer
          `other`'s IN as 3 even while X itself is still fully generic.
        - V (m's own total OUT - shaders/maps/select.h needs it to size
          `values[V]`, the buffer sel indexes into) and the resulting
          range validation genuinely do need m's own OUT, so those stay
          deferred (V left unbound) until it's known.
        - An open/negative-bound slice (e.g. X[:6], X[-3:]) can't be
          resolved into concrete indices without m's own OUT either - in
          that case OUT/sel *also* stay deferred, same as V.

        Whatever's still open here is left for Node.cast() to resolve
        later, by replaying __init__ with these exact same (m, indices)
        once something makes it (or m) concrete.

        If this Select's own IN is already known when this runs (e.g. a
        replay via cast(), triggered by an outer composition that already
        knows what IN must be - see Composite.__init__), but m's own IN
        isn't, m is cast to match first - the same "push a known IN down
        into a still-generic argument" idiom ConcatMap.__init__ already uses
        between its own m1/m2. For an X (Identity) with truly nothing else
        constraining it, this is often exactly what unlocks the rest (m's
        OUT, hence v below): Identity's own __init__ ties its OUT to
        whatever IN ends up being.
        """
        if not isinstance(indices, tuple):
            indices = (indices,)
        self_in = IN(type(self))
        if self_in is not None and IN(type(m)) is None:
            m = m.cast(Map[IN: self_in])
        M_ = type(m)
        v = OUT(M_)
        self.match(IN, IN(M_))
        self.match(Generics.M, M_)
        self.m = m
        picked = []
        for idx in indices:
            if not isinstance(idx, slice):
                picked.append(idx)
                continue
            if v is not None:
                picked.extend(range(*idx.indices(v)))
                continue
            if idx.start is None or idx.stop is None or idx.start < 0 or idx.stop < 0:
                return  # needs m's own OUT to mean anything - stays deferred
            picked.extend(range(idx.start, idx.stop, idx.step if idx.step is not None else 1))
        self.match(OUT, len(picked))
        self.sel = picked
        if v is not None:
            assert all(0 <= i < v for i in picked), \
                f"Select: every index must be in [0, {v}) - m's own OUT."
            self.match(Generics.V, v)


class ConcatMap(Map, Generic[Generics.M1, Generics.M2]):
    __aliases__ = {
        Generics.M1_OUT: OUT(Generics.M1),
        Generics.M2_OUT: OUT(Generics.M2),
    }
    m1: Generics.M1
    m2: Generics.M2
    __code__ = "shaders/maps/concat.h"
    def __init__(self, m1, m2) -> None:
        m1 = m1.cast(Map[IN: IN(m2)])
        m2 = m2.cast(Map[IN: IN(m1)])
        assert IN(m1) == IN(m2)
        M1 = type(m1)
        M2 = type(m2)
        if OUT(M1) and OUT(M2):
            self.match(OUT, OUT(M1) + OUT(M2))
        self.match(IN, IN(m1))
        self.match(Generics.M1, M1)
        self.match(Generics.M2, M2)
        self.m1 = m1
        self.m2 = m2


def as_map(data):
    if isinstance(data, Map):
        return data  # aleady a Map
    if isinstance(data, (int, float)):
        return Constant((float(data),))
    return Constant(data) # try any other RDVObject as constant


def _map_like(m1, reference_map):
    if not isinstance(reference_map, Map):
        return m1  # nothing to do
    m1 = as_map(m1)
    m1 = m1.cast(Map[IN: IN(reference_map)])  # try to infer IN(reference_map)
    if OUT(m1) == 1 and OUT(reference_map) is not None:
        return m1.promote(OUT(reference_map))
    return m1.cast(Map[OUT: OUT(reference_map)])


class BinaryOperation(Map, Generic[Generics.M1, Generics.M2]):
    __abstract__ = True
    m1 : Generics.M1
    m2 : Generics.M2
    __code__ = "shaders/maps/binary.h"

    def __init__(self, m1, m2) -> None:
        # If this BinaryOperation's own IN is already known at this point
        # (e.g. a replay via cast(), triggered by an outer composition
        # that already knows what IN must be), but m1/m2's own IN isn't,
        # push it down into them first - same idiom Composite/Select/
        # Average use for their own child map(s). Without this, an
        # operand built purely from other still-generic-IN maps (e.g.
        # X[0:3] + X[3:4] * ...) can never resolve its own IN: _map_like
        # below only cross-infers IN from the *other* operand, which
        # fails too when both start out IN-generic - the class stays
        # forever unspecialized (M1/M2 shown with IN=None even once this
        # instance's own IN resolves), and dispatch later fails deep
        # inside GPU pointer resolution instead of at construction time.
        self_in = IN(type(self))
        if self_in is not None:
            if isinstance(m1, Map) and IN(type(m1)) is None:
                m1 = m1.cast(Map[IN: self_in])
            if isinstance(m2, Map) and IN(type(m2)) is None:
                m2 = m2.cast(Map[IN: self_in])
        m1 = _map_like(m1, m2)
        m2 = _map_like(m2, m1)
        # _map_like conforms m1/m2 to share the same IN/OUT (broadcasting
        # a scalar-OUT operand via promote() where needed) - but never
        # actually ties *this* instance's own IN/OUT to them, unlike e.g.
        # Composite.__init__'s self.match(IN, ...)/self.match(OUT, ...).
        # Without this, a still-generic BinaryOperation (Add/Sub/Mul/Div/
        # Mod) never resolves its own OUT even once both operands do.
        self.match(IN, IN(type(m1)))
        self.match(OUT, OUT(type(m1)))
        # Also missing: M1/M2 themselves (this instance's own generics
        # tracking m1/m2's concrete types, like Composite's F/G) were never
        # bound either - only the field *values* below were ever set, so
        # this stayed generic in M1/M2 even once IN/OUT resolved above.
        self.match(Generics.M1, type(m1))
        self.match(Generics.M2, type(m2))
        self.m1 = m1
        self.m2 = m2


class Add(BinaryOperation):
    __precode__ = "#define BINARY_OP(x, y) ((x) + (y))"
    __poscode__ = "#undef BINARY_OP"


class Sub(BinaryOperation):
    __precode__ = "#define BINARY_OP(x, y) ((x) - (y))"
    __poscode__ = "#undef BINARY_OP"


class Mul(BinaryOperation):
    __precode__ = "#define BINARY_OP(x, y) ((x) * (y))"
    __poscode__ = "#undef BINARY_OP"


class Div(BinaryOperation):
    __precode__ = "#define BINARY_OP(x, y) ((x) / (y))"
    __poscode__ = "#undef BINARY_OP"


class Mod(BinaryOperation):
    __precode__ = "#define BINARY_OP(x, y) ((x) % (y))"
    __poscode__ = "#undef BINARY_OP"


# ============== DISTRIBUTIONS ===================


class Uniform(Map):
    """Uniform noise in [0, 1) per output component - ignores its own
    input entirely (see shaders/maps/uniform.h); composed after a grid
    mainly to get one independent random draw per grid cell, relying on
    RandomLibrary's own per-thread seeding for that independence."""
    __requires__ = [RandomLibrary]
    __code__ = "shaders/maps/uniform.h"


class UniformDirection(Map):
    """Uniform direction in R^N"""
    __requires__ = [RandomLibrary]
    __code__ = "shaders/maps/uniform_direction.h"


class Normal(Map):
    pass


# ===================== Grid Samplers ===========================


class GridSampler(Map, Generic[Generics.G]):
    """
    Multi-linear (bilinear/trilinear/...) or nearest-neighbor sampling of a
    grid at continuous coordinates in [-1, 1]^IN - see shaders/samplers/grid_sampler.h.
    `linear` selects multi-linear interpolation (True, the 2^IN-corner blend)
    or nearest-neighbor (False, the single closest cell - no blending).
    `clamp` controls out-of-range corners/indices: clamped to the nearest
    valid index (True), or treated as a zero contribution so the result
    fades to 0 past the border (False). `aligned` controls how [-1, 1] maps
    to the grid: True ties -1/+1 to the centers of the first/last cell;
    False ties them to the outer edges of the grid, with the first/last
    cell centers strictly inside (-1, 1) - matching
    torch.nn.functional.grid_sample's align_corners flag either way.
    """
    __code__ = "shaders/samplers/grid_sampler.h"
    g: Generics.G
    clamp: int
    aligned: int
    linear: int

    def __init__(self, g, clamp: bool = True, aligned: bool = False, linear: bool = True) -> None:
        G = type(g)
        self.match(IN, IN(G))
        self.match(OUT, OUT(G))
        self.match(Generics.G, G)
        self.g = g
        self.clamp = 1 if clamp else 0
        self.aligned = 1 if aligned else 0
        self.linear = 1 if linear else 0


class TensorSampler(Map):
    """
    Like GridSampler, but specialized for a densely-packed Tensor[FLOAT32,
    IN+1] (as opposed to an arbitrary Grid): t's own N-1=IN leading dims are
    the contiguous, row-major spatial axes to interpolate over
    (shape[0..IN-1]) and its trailing dim is this sampler's own OUT, so
    every corner's address is computed directly via pointer arithmetic over
    t's own base pointer/strides (see shaders/samplers/tensor_sampler.h) - no
    virtual forward() dispatch per corner. The two fastest-varying (last
    spatial dimension) neighbors are always adjacent in memory (stride 1),
    so their blend is handled in one visit by scanline_interpolator instead
    of two independent corner fetches. clamp/aligned/linear mean the same
    as in GridSampler. t's own rank (Tensor's N) is expressed directly as
    Generics.IN + 1 - see GenericsMeta.__add__/_GenericArithmetic - so
    __IN__ alone drives both this sampler's own field and t's, instead of
    needing a second, separately-bound generic kept in sync by hand.
    """
    __code__ = "shaders/samplers/tensor_sampler.h"
    __aliases__ = {
        Generics.F: FixArray[Type.FLOAT32, OUT],
    }
    t: Tensor[T:Type.FLOAT32, IN:(IN + 1)]
    clamp: int
    linear: int
    aligned: int

    def __init__(self, t, clamp: bool = True, aligned: bool = False, linear: bool = True) -> None:
        n = Generics.N(t)
        self.match(IN, n - 1)
        self.match(OUT, t.shape[n - 1])
        self.t = t
        self.clamp = 1 if clamp else 0
        self.aligned = 1 if aligned else 0
        self.linear = 1 if linear else 0


class ResourceHandle:
    """
    Opaque, refcounted handle to a single resource (an Image or Sampler)
    bound into one of ImagingLibrary's registry arrays - returned by
    ImagingLibrary.create_image_1d/2d/3d/create_sampler, never constructed
    directly.

    Multiple owners can share one handle via acquire() instead of each
    creating/binding a redundant resource - see GPUSampler.clone(). The
    registry slot is only actually freed (its index queued for reuse by
    the next create_image_*/create_sampler call) and the underlying vk
    resource dropped once every owner has release()'d it (refcount
    reaches 0).
    """
    __slots__ = ('_library', '_slot', '_device_index', '_index', '_resource', '_refcount')

    def __init__(self, library, slot: int, device_index: int, index: int, resource):
        self._library = library
        self._slot = slot
        self._device_index = device_index
        self._index = index
        self._resource = resource

    @property
    def index(self) -> int:
        return self._index

    @property
    def resource(self):
        return self._resource

    def release(self) -> None:
        """Drops one owner's reference; idempotent (a second release()
        from the same owner, e.g. via __del__ after an explicit release(),
        is a no-op). Once every owner has released, frees the registry
        slot for reuse and drops the underlying vk resource."""
        if self._resource is None:
            return
        self._library._release(self._device_index, self._slot, self._index)
        self._resource = None

    def __del__(self):
        # Backstop for an owner that never calls release() explicitly -
        # relies on the refcount, so a still-shared handle is unaffected.
        try:
            self.release()
        except Exception:
            pass


class ImagingLibrary(Library):
    """
    Bindless-style image/sampler registry, shared globally (like
    DeferredLibrary): create_image_1d/2d/3d create an Image (Library's own
    responsibility, not each requiring type's - see GPUSampler) and bind
    it into the matching fixed-capacity SAMPLED_IMAGE array; create_sampler
    does the same for the SAMPLER array - both return an opaque
    ResourceHandle (see its own docstring for the ownership/sharing/
    release model) rather than a bare index. Freed slots (every handle
    release()'d) are reused by later create_image_*/create_sampler calls
    before the registry grows further - see _alloc()/_release().

    See shaders/libraries/imaging_library.h for the GLSL side (sample_image, which
    combines an image index with a sampler index). Every slot is dynamic
    (re-bound on every dispatch() call - see dynamic_resources()), so a
    newly created image/sampler becomes visible to the very next dispatch,
    without rebuilding any pipeline. MAX_IMAGES/MAX_SAMPLERS are fixed (not
    a Generic, matching DeferredLibrary's own fixed-size buffer) since
    this is one shared registry, not something meant to be specialized
    differently per requiring type - kept in sync by hand with the array
    sizes declared in imaging_library.h.
    """
    MAX_IMAGES = 1024*16
    MAX_SAMPLERS = 1024*16

    __descriptors__ = [
        [MAX_IMAGES, DescriptorType.SAMPLED_IMAGE],    # 1D images
        [MAX_IMAGES, DescriptorType.SAMPLED_IMAGE],    # 2D images
        [MAX_IMAGES, DescriptorType.SAMPLED_IMAGE],    # 3D images
        [MAX_SAMPLERS, DescriptorType.SAMPLER],        # samplers
    ]

    __resources__ = {}  # di -> [slot -> [resource or None, ...]] (one list per slot 0..3)
    __free__ = {}       # di -> [slot -> [freed index, ...]] (reusable before the registry grows)

    @classmethod
    def _ensure(cls, di: int) -> None:
        if di not in cls.__resources__:
            cls.__resources__[di] = [[], [], [], []]
            cls.__free__[di] = [[], [], [], []]

    @classmethod
    def static_resources(cls):
        return [None, None, None, None]

    @classmethod
    def dynamic_resources(cls):
        di = device_index()
        cls._ensure(di)
        # Occupied slots aren't necessarily contiguous from 0 any more
        # (release() can free one in the middle for reuse) - dispatch()
        # binds each (resource, index) pair individually for a sparse
        # list like this instead of one combined array bind (see there).
        result = []
        for registry in cls.__resources__[di]:
            occupied = [(r, idx) for idx, r in enumerate(registry) if r is not None]
            result.append(occupied or None)
        return result

    @classmethod
    def _alloc(cls, slot: int, capacity: int, resource) -> ResourceHandle:
        di = device_index()
        cls._ensure(di)
        free_list = cls.__free__[di][slot]
        registry = cls.__resources__[di][slot]
        if free_list:
            index = free_list.pop()
            registry[index] = resource
        else:
            index = len(registry)
            assert index < capacity, f'ImagingLibrary: slot {slot} is full ({capacity} max).'
            registry.append(resource)
        return ResourceHandle(cls, slot, di, index, resource)

    @classmethod
    def _release(cls, di: int, slot: int, index: int) -> None:
        cls.__resources__[di][slot][index] = None
        cls.__free__[di][slot].append(index)

    @classmethod
    def create_image_1d(cls, width, format, location=MemoryLocation.DEVICE) -> ResourceHandle:
        """Creates a 1D Image and binds it into the registry - returns its handle."""
        return cls._alloc(0, cls.MAX_IMAGES, _vk.readonly_image(width, 1, 1, format=format, location=location, dimensions=1))

    @classmethod
    def create_image_2d(cls, width, height, format, location=MemoryLocation.DEVICE) -> ResourceHandle:
        """Creates a 2D Image and binds it into the registry - returns its handle."""
        # dimensions=2 forces a genuine 2D image even when height==1 (a
        # degenerate but legitimate 2D size, e.g. a 1-row environment map) -
        # without it, _vk.readonly_image's own height-based inference
        # silently hands back a 1D image instead, which this class still
        # registers/binds as 2D (gpu_sampler.h's IN==2 always samples via
        # sampler2D) - a type mismatch that reads back as all-zero.
        return cls._alloc(1, cls.MAX_IMAGES, _vk.readonly_image(width, height, 1, format=format, location=location, dimensions=2))

    @classmethod
    def create_image_3d(cls, width, height, depth, format, location=MemoryLocation.DEVICE) -> ResourceHandle:
        """Creates a 3D Image and binds it into the registry - returns its handle."""
        return cls._alloc(2, cls.MAX_IMAGES, _vk.readonly_image(width, height, depth, format=format, location=location, dimensions=3))

    @classmethod
    def create_sampler(cls, **kwargs) -> ResourceHandle:
        """Creates a Sampler (kwargs forwarded to _vk.sampler) and binds it
        into the registry - returns its handle."""
        return cls._alloc(3, cls.MAX_SAMPLERS, _vk.sampler(**kwargs))

    __code__ = "shaders/libraries/imaging_library.h"


class GPUSampler(Map):
    """
    Like GridSampler/TensorSampler (coordinates in [-1, 1]), but samples
    through a real Vulkan Image + Sampler (registered in ImagingLibrary)
    instead of manual pointer arithmetic - `clamp` (WrapMode.CLAMP_TO_EDGE
    vs CLAMP_TO_BORDER) and `linear` (Filter.LINEAR vs NEAREST) are baked
    into the native Sampler object once, at construction time (zero
    per-sample cost, unlike GridSampler/TensorSampler's own runtime
    branches) - the shader only ever adjusts for `aligned` (real texture
    sampling has no "align corners" notion) before handing off to actual
    hardware texture filtering via textureSize()/texture().

    OUT is limited to 4 (1..4-channel float image formats -
    R32_Float/RG32_Float/RGB32_Float/RGBA32_Float); IN to 1..3 (1D/2D/3D
    images). A uint8 tensor gets an 8-bit UNorm image instead (R8/RG8/
    RGBA8_UNorm, a quarter of the memory - e.g. a label volume), sampled
    as value / 255 (exact at texel centres with linear=False). update(t) (re)loads the image from t - never done implicitly,
    so repeatedly sampling an unchanged tensor doesn't re-upload every time.

    One known hardware-level limitation: clamp=False (CLAMP_TO_BORDER)'s
    border color is always opaque black - (0,0,0,1), not configurable in
    this vk binding - so with OUT == 4 specifically, the 4th channel reads
    1.0 (not 0.0) past the border.

    The underlying image/sampler are created and registered by
    ImagingLibrary itself (see create_image_1d/2d/3d/create_sampler), not
    by this class - this instance just owns the resulting ResourceHandle
    pair, releasing them (freeing their registry slot for reuse) once
    release()'d or garbage-collected. clone() shares (acquires) those same
    handles with a new GPUSampler instance instead of allocating redundant
    ones - prefer it over repeatedly constructing GPUSamplers for the same
    data.
    """
    MAX_OUT = 4
    _FORMATS = {
        1: _vk.Format.R32_Float,
        2: _vk.Format.RG32_Float,
        # NOT RGB32_Float: a 3-component 32-bit-float format has no
        # SAMPLED_IMAGE support on virtually any GPU/driver (Vulkan's own
        # format support tables never require it - hardware texture units
        # expect power-of-2 texel sizes: 4/8/16 bytes, not 12) - confirmed
        # directly on this device (readonly_image raises
        # ErrorFormatNotSupported for RGB32_Float, same as RGB8_UNorm).
        # RGBA32_Float instead, padded to 4 channels in __init__/update()
        # below - gpu_sampler.h already only ever reads OUT (3) of the
        # sampled vec4's own components, so no shader change is needed.
        3: _vk.Format.RGBA32_Float,
        4: _vk.Format.RGBA32_Float,
    }
    _FORMATS_UINT8 = {  # uint8 tensors; same RGB -> RGBA padding as above
        1: _vk.Format.R8_UNorm,
        2: _vk.Format.RG8_UNorm,
        3: _vk.Format.RGBA8_UNorm,
        4: _vk.Format.RGBA8_UNorm,
    }

    image_index: int
    sampler_index: int
    aligned: int
    shape: Array[IN, int]

    __requires__ = [
        ImagingLibrary
    ]

    __code__ = "shaders/samplers/gpu_sampler.h"
    _wrap = None
    _image_handle = None
    _sampler_handle = None
    _pad_alpha = False  # see _FORMATS[3]'s own comment

    def __init__(self, t: _typing.Any,
                 clamp: bool = True,
                 aligned: bool = False,
                 linear: bool = True,
                 _upload: bool = True) -> None:
        self._wrap = wrap(t)
        self.shape = self._wrap.shape[:-1]
        in_size = len(self._wrap.shape) - 1
        out_size = self._wrap.shape[-1]
        assert 1 <= in_size <= 3, f"GPUSampler: only 1D/2D/3D tensors are supported, got {in_size}D."
        assert 1 <= out_size <= self.MAX_OUT, f"GPUSampler: OUT is limited to 1..{self.MAX_OUT}, got {out_size}."
        self.match(IN, in_size)
        self.match(OUT, out_size)
        if type(self).__is_generic__:
            # Called from a still-generic GPUSampler (IN/OUT weren't bound
            # yet, e.g. GPUSampler(t) rather than GPUSampler[2, 3](t)) -
            # _NodeMeta.__call__ is about to rebuild this instance from
            # scratch on the now-concrete specialization the match() calls
            # above just resolved, replaying __init__ a second time. Unlike
            # every other Node here, this one has real side effects
            # (allocating/sharing the image/sampler handles, uploading
            # data) that must happen exactly once - deferred to that
            # second, concrete call.
            return

        # Vulkan images are width-fastest-varying; tensor shape is
        # dim0-slowest (row-major, last dim fastest) - reverse the
        # spatial shape so the image's own (width, height[, depth])
        # lines up with the shader's reversed _input[] axis order.
        # self.shape (the node field) reads back through the
        # Accessor, not a plain list - use self._wrap.shape instead.
        width, height, depth = (list(reversed(self._wrap.shape[:-1])) + [1, 1])[:3]
        formats = self._FORMATS_UINT8 if self._wrap.scalar_type == Type.UINT8 else self._FORMATS
        image_format = formats[out_size]
        if out_size == 3:
            # The image itself is really RGBA32_Float (see _FORMATS' own
            # comment) - pad the uploaded data to 4 channels to match, a
            # zero alpha never read back (OUT stays 3 - see gpu_sampler.h).
            # update() (called below and on every future .update()) redoes
            # this same padding from self._wrap's own then-current content,
            # so re-uploading new 3-channel data later still works.
            self._pad_alpha = True
        create_image = (
            ImagingLibrary.create_image_1d, ImagingLibrary.create_image_2d, ImagingLibrary.create_image_3d
        )[in_size - 1]
        dims = (width, height, depth)[:in_size]
        self._image_handle = create_image(*dims, format=image_format, location=MemoryLocation.DEVICE)
        edge = _vk.WrapMode.CLAMP_TO_EDGE if clamp else _vk.WrapMode.CLAMP_TO_BORDER
        filt = _vk.Filter.LINEAR if linear else _vk.Filter.NEAREST
        self._sampler_handle = ImagingLibrary.create_sampler(
            mag_filter=filt, min_filter=filt, wrap_u=edge, wrap_v=edge, wrap_w=edge)
        self.aligned = 1 if aligned else 0
        self.image_index = self._image_handle.index
        self.sampler_index = self._sampler_handle.index
        if _upload:
            self.update()

    def release(self) -> None:
        """Releases this instance's own reference to its image/sampler
        handles - idempotent. If no other GPUSampler (e.g. from clone())
        still holds them, their registry slots are freed for reuse and
        the underlying vk resources dropped."""
        self._image_handle = None
        self._sampler_handle = None

    def update(self) -> None:
        """(Re)loads the GPU image's own content from this instance's own
        wrapped tensor (see __init__) - a plain torch/numpy tensor, an
        RDVObject, or anything else wrap() accepts, of shape
        [*spatial_shape, OUT] matching this sampler's own IN/OUT. The
        image itself can't be resized - the tensor's shape must match the
        one this sampler was constructed with.
        """
        source = self._wrap
        if self._pad_alpha:
            # The image is really RGBA32_Float (see _FORMATS[3]'s own
            # comment) - build a 4-channel view with a zero (never read
            # back) alpha, redone every update() (not cached: self._wrap's
            # own content may have changed - e.g. update() called again
            # after mutating the original tensor in place) from whatever
            # self._wrap currently holds.
            padded = self._wrap.as_tensor('torch')
            import torch as _torch
            padded4 = _torch.zeros(*padded.shape[:-1], 4, dtype=padded.dtype, device=padded.device)
            padded4[..., :3] = padded
            source = wrap(padded4)
        with _vk.device(self.device_index):
            with _vk.transfer() as cb:
                cb.transfer(source.memory, self._image_handle.resource)


class GPUSamplerBarrier(GPUSampler):
    """
    GPUSampler.__barrier_type__ (see _BarrierDeclaration / `grid >>
    barrier[GPUSampler](...)`): materializes grid into a real Vulkan Image
    sampled through GPUSampler, instead of a plain tensor (see
    TensorGridBarrier) - fills a staging tensor with grid's own dispatch,
    then transfers that tensor into this sampler's own Image. Two hazards,
    hence two barriers: compute write (the grid's own dispatch) -> transfer
    read (the tensor -> image copy), then transfer write -> texture read
    (whatever samples this image next, recorded right after in the same
    command buffer - see dispatch()). _upload=False on the super().__init__
    call below skips GPUSampler's own eager image upload (the staging
    tensor is still uninitialized garbage at construction time) - this
    instance's own append_computation is what actually fills it, lazily,
    the first time it's ever dispatched.
    """

    def __init__(self, grid, clamp: bool = True, aligned: bool = False, linear: bool = True) -> None:
        staging = RDVObject(_vk.tensor([*grid._shape, OUT(type(grid))], Type.FLOAT32, location=MemoryLocation.DEVICE))
        super().__init__(staging, clamp=clamp, aligned=aligned, linear=linear, _upload=False)
        if type(self).__is_generic__:
            return
        self._grid_evaluator = GridEvaluator(grid, staging.device_ptr)
        self._grid_threads = (_math.prod(grid._shape), 1, 1)

    def append_computation(self, cmd) -> None:
        append_dispatch(cmd, self._grid_evaluator, self._grid_threads)
        cmd.barrier()
        cmd.transfer(self._wrap.memory, self._image_handle.resource)
        cmd.barrier()


GPUSampler.__barrier_type__ = GPUSamplerBarrier


class SingleImageLibrary(Library, Generic[Generics.ID, IN]):
    """
    One image + one sampler for one generic ID: SingleImageLibrary[ID: k, IN: d]
    owns exactly two plain (non-array) descriptor slots - a SAMPLED_IMAGE
    (d-dimensional) and a SAMPLER - unlike ImagingLibrary's bindless
    registry arrays, which WebGPU can't express. Different IDs are different
    specializations, so they get different slots and coexist in the same
    megakernel; SingleGPUSampler[ID: k]/SingleGPUGrid[ID: k] require (and so
    read from) the one with their own ID (see shaders/libraries/
    single_image_library.h).

    The bound image/sampler belong to the ID, not to a node instance: the
    last SingleGPUSampler/SingleGPUGrid constructed with a given ID (on a
    given device) is the one bound - construct one per ID per data set. The
    registry also keeps the source tensor and sampler settings, which the
    WebGPU export uses to recreate the same texture/sampler.
    """
    __abstract__ = True
    __code__ = "shaders/libraries/single_image_library.h"
    __descriptors__ = [
        DescriptorType.SAMPLED_IMAGE,
        DescriptorType.SAMPLER,
    ]
    __bound__ = {}  # (ID, device index) -> dict(image, sampler, source, clamp, linear, padded)

    @classmethod
    def _key(cls):
        return Generics.ID(cls), device_index()

    @classmethod
    def bind(cls, image, sampler, **info) -> None:
        cls.__bound__[cls._key()] = dict(image=image, sampler=sampler, **info)

    @classmethod
    def bound(cls) -> dict | None:
        return cls.__bound__.get(cls._key())

    @classmethod
    def static_resources(cls):
        b = cls.bound()
        return [None, None] if b is None else [b['image'], b['sampler']]

    @classmethod
    def dynamic_resources(cls):
        # rebound every dispatch: a later constructor may have bound a new
        # image/sampler to this same ID after the pipeline was built
        return cls.static_resources()


def _bind_single_image(node, t, clamp: bool, linear: bool):
    """Shared by SingleGPUSampler/SingleGPUGrid: creates node's own image +
    sampler from t ([*spatial, C] - C in 1..4 channels, spatial 1..3D),
    uploads t and binds both to SingleImageLibrary[ID, IN]. Returns the
    spatial shape. Same formats/padding as GPUSampler (3 channels are
    stored as RGBA32_Float)."""
    w = wrap(t)
    in_size, out_size = len(w.shape) - 1, w.shape[-1]
    assert 1 <= in_size <= 3, f"{type(node).__name__}: only 1D/2D/3D tensors are supported, got {in_size}D."
    assert 1 <= out_size <= 4, f"{type(node).__name__}: 1..4 channels are supported, got {out_size}."
    node.match(IN, in_size)
    node.match(OUT, out_size)
    if type(node).__is_generic__:
        return w.shape[:-1]  # __init__ is replayed on the concrete class - create resources only then
    width, height, depth = (list(reversed(w.shape[:-1])) + [1, 1])[:3]
    image = _vk.readonly_image(width, height, depth, format=GPUSampler._FORMATS[out_size],
                               location=MemoryLocation.DEVICE, dimensions=in_size)
    edge = _vk.WrapMode.CLAMP_TO_EDGE if clamp else _vk.WrapMode.CLAMP_TO_BORDER
    filt = _vk.Filter.LINEAR if linear else _vk.Filter.NEAREST
    sampler = _vk.sampler(mag_filter=filt, min_filter=filt, wrap_u=edge, wrap_v=edge, wrap_w=edge)
    source = w
    if out_size == 3:  # RGBA32_Float image - see GPUSampler._FORMATS
        padded = w.as_tensor('torch')
        import torch as _torch
        padded4 = _torch.zeros(*padded.shape[:-1], 4, dtype=padded.dtype, device=padded.device)
        padded4[..., :3] = padded
        source = wrap(padded4)
    with _vk.device(node.device_index):
        with _vk.transfer() as cb:
            cb.transfer(source.memory, image)
    library = SingleImageLibrary[Generics.ID(node), in_size]
    library.bind(image, sampler, source=source, shape=list(w.shape[:-1]), channels=out_size,
                 clamp=clamp, linear=linear)
    return w.shape[:-1]


class SingleGPUSampler(Map, Generic[Generics.ID]):
    """
    GPUSampler (same [-1, 1] coordinates, clamp/aligned/linear meaning and
    1..4-channel, 1..3D formats), but its image/sampler are the single
    resources bound for its generic ID (SingleImageLibrary[ID]) instead of
    entries of ImagingLibrary's bindless arrays - which makes it exportable
    to WebGPU (there: one texture + one sampler binding per ID).
    SingleGPUSampler[ID: 0](cloud) and SingleGPUSampler[ID: 1](other) sample
    different images; constructing a second one with the same ID rebinds
    that ID (see SingleImageLibrary).
    """
    __requires__ = [SingleImageLibrary[Generics.ID, IN]]
    __code__ = "shaders/samplers/single_gpu_sampler.h"
    aligned: int
    shape: Array[IN, int]

    def __init__(self, t: _typing.Any, clamp: bool = True, aligned: bool = False, linear: bool = True) -> None:
        shape = _bind_single_image(self, t, clamp, linear)
        self.shape = list(shape)
        self.aligned = 1 if aligned else 0


class SingleGPUGrid(Grid, Generic[Generics.ID]):
    """
    A Grid over the single image bound for its generic ID
    (SingleImageLibrary[ID]): forward(index) is the exact texel at that
    integer index (texelFetch - no filtering, values bit-exact), Grid's own
    index convention (index[0] slowest). E.g. a majorant grid kept in a
    texture: VoxelMajorantField(SingleGPUGrid[ID: 1](majorant_tensor)).
    """
    __requires__ = [SingleImageLibrary[Generics.ID, IN]]
    __code__ = "shaders/grids/single_gpu_grid.h"

    def __init__(self, t: _typing.Any) -> None:
        shape = _bind_single_image(self, t, clamp=True, linear=False)
        super().__init__(list(shape))
