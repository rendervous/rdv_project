from ..core import *
from .. import maths as _maps
import typing as _typing
import math as _math


# ============= RENDERING CAMERA ==================


class Cameras(_maps.Grid[3, 6]):
    __code__ = "shaders/rendering/cameras.h"
    __requires__ = [
        RandomLibrary,
        MathLibrary
    ]
    __aliases__ = {
        Generics.P: FixArray[Type.VEC3, 3],
    }
    poses: Deferred[Type.UINT64]
    aspect_ratio: float
    fov: float
    pixel_jitter: int

    def __init__(self,
                 width: int,
                 height: int,
                 poses: _typing.Any | RDVObject,
                 aspect_ratio: float | None = None,
                 fov: float | None = None,
                 jitter: bool = False) -> None:
        if aspect_ratio is None:
            aspect_ratio = width / height
        if fov is None:
            fov = _math.pi / 4
        if not isinstance(poses, RDVObject):
            poses = wrap(poses)
        assert poses.shape[1] == 9, "poses must be a [num_cameras, 9] tensor (ori | target | up)"
        super().__init__([poses.shape[0], height, width])
        self.poses = poses
        self.aspect_ratio = aspect_ratio
        self.fov = fov
        self.pixel_jitter = 1 if jitter else 0


class CameraReproject(_maps.Grid, Generic[Generics.S, Generics.PD, Generics.D]):
    """
    Reprojects samples a Camera previously accumulated at a different pose/
    fov onto a NEW camera's own pixel grid, using the new camera's own
    depth map (parametric distance along its own primary ray, e.g.
    VolumeProfile's "depth 0" output channel) to reconstruct each new
    pixel's world-space hit point, then looking that point up - continuous-
    coordinate, via an internally-built GridSampler (see `clamp`/`aligned`/
    `linear`) - in `grid` (the previous camera's own accumulated
    Grid[3, K], indexed [cam, y, x] same as Cameras' own convention) at the
    coordinate the previous camera would itself have projected that same
    point to. Meant for one camera moving over time (not a batch of
    different cameras), so `grid`'s own cam axis is always sampled dead-
    center - `depth`'s own cam axis (this map's own shape[0]) can still be
    whatever it needs to be, e.g. multiple viewports all reprojecting
    against the same single previous frame.

    `depth` must be a Grid[3, 1] matching this map's own IN=3 (cam, y, x)
    index space (the NEW camera's own pixel grid) and own shape (the new
    camera's own resolution) - shape is taken from `depth` directly, same
    as e.g. MajorantGrid takes its own shape from its `shape` argument.

    `prev_depth` is a Grid[3, 1] matching `grid`'s own indexing (the
    PREVIOUS camera's own pixel grid) - the depth that was actually
    recorded at each of the previous camera's own pixels when `grid` was
    built. Sampled at the same reprojected coordinate as `grid` itself, so
    it can be compared against the new point's own depth *as seen from the
    previous camera* (`dz` below - not `depth`, which is along the NEW
    camera's own ray): if they roughly agree, the same surface was visible
    from both cameras and the reprojected sample is trustworthy; a large
    gap means whatever the previous camera actually saw along that
    direction was a different surface entirely (e.g. a disocclusion, or a
    thin/foreground occluder only one of the two cameras can see) - a
    purely geometric check, independent of the frustum/behind-camera one
    already covered by the validity mask.

    OUT is `grid`'s own OUT + 2: [0:OUT(grid)] is the reprojected sample,
    [OUT(grid)] is a validity mask (1.0 if the reprojected point falls
    inside the previous camera's own frustum and in front of it, 0.0
    otherwise - e.g. newly disoccluded regions, or a background/no-hit new
    pixel with depth <= 0), [OUT(grid) + 1] is the depth discrepancy
    described above (world-space distance units, 0.0 wherever invalid).
    What either should mean for accumulation (skip the update entirely,
    fall back to a fresh sample, scale a per-pixel blend weight, etc.) is
    left to the caller, not guessed here.

    `pose`/`previous_pose` are each an (origin, target, up) vec3-like
    triple, same convention as Cameras' own `poses` tensor rows (just as
    plain fields here, not a tensor, since this is a single camera's own
    two points in time, not a batch).
    """
    __code__ = "shaders/rendering/camera_reproject.h"
    prev_sampler: Generics.S
    prev_depth_sampler: Generics.PD
    depth: Generics.D
    origin: vec3
    target: vec3
    up: vec3
    fov: float
    aspect_ratio: float
    previous_origin: vec3
    previous_target: vec3
    previous_up: vec3
    previous_fov: float
    previous_aspect_ratio: float

    def __init__(self, grid, depth, prev_depth,
                 pose, fov: float, aspect_ratio: float,
                 previous_pose, previous_fov: float, previous_aspect_ratio: float,
                 clamp: bool = True, aligned: bool = False, linear: bool = True) -> None:
        assert Generics.IN(type(grid)) == 3, "CameraReproject expects a Grid[3, ...] previous grid (cam, y, x)."
        D_ = type(depth)
        assert Generics.IN(D_) == 3 and Generics.OUT(D_) == 1, \
            "CameraReproject expects a Grid[3, 1] depth map (cam, y, x), matching its own shape."
        PD_ = type(prev_depth)
        assert Generics.IN(PD_) == 3 and Generics.OUT(PD_) == 1, \
            "CameraReproject expects a Grid[3, 1] prev_depth map (cam, y, x), matching grid's own shape."
        prev_sampler = _maps.GridSampler(grid, clamp=clamp, aligned=aligned, linear=linear)
        prev_depth_sampler = _maps.GridSampler(prev_depth, clamp=clamp, aligned=aligned, linear=linear)
        self.match(Generics.S, type(prev_sampler))
        self.match(Generics.PD, type(prev_depth_sampler))
        self.match(Generics.D, D_)
        self.match(Generics.IN, 3)
        self.match(Generics.OUT, Generics.OUT(type(prev_sampler)) + 2)
        super().__init__(depth._shape)
        self.prev_sampler = prev_sampler
        self.prev_depth_sampler = prev_depth_sampler
        self.depth = depth

        def _vec3(v):
            return v if isinstance(v, vec3) else vec3(*v)

        o, t, u = pose
        self.origin, self.target, self.up = _vec3(o), _vec3(t), _vec3(u)
        self.fov = fov
        self.aspect_ratio = aspect_ratio
        po, pt, pu = previous_pose
        self.previous_origin, self.previous_target, self.previous_up = _vec3(po), _vec3(pt), _vec3(pu)
        self.previous_fov = previous_fov
        self.previous_aspect_ratio = previous_aspect_ratio


# ============= PROJECTIONS =======================


class XRProject(_maps.Map[3, 2]):
    __code__ = "shaders/rendering/xr_project.h"
    __requires__ = [
        MathLibrary
    ]


dir2xr = XRProject()
"""
Projects a 3D direction into its 2D equirectangular coordinate.
"""

class XRUnproject(_maps.Map[2, 3]):
    __code__ = "shaders/rendering/xr_unproject.h"
    __requires__ = [
        MathLibrary
    ]


xr2dir = XRUnproject()
"""
Unprojects a 2D equirectangular coordinate to its spherical position.
"""


def _build_xr_density_quadtree(image, levels: int):
    """
    image: [H, W, C] (e.g. an environment map's own RGB data, already on
    rdv - a plain tensor is wrap()ed). Returns a flat float32 tensor: for
    tree node n (root n=0), buf[n*4:n*4+4] holds the *cumulative*
    probabilities of descending into each of that node's own 4 children
    (buf[n*4+3] ~= 1, in leaf-index order 0=(left,top), 1=(right,top),
    2=(left,bottom), 3=(right,bottom) - matching the (u0,u1)/(v0,v1)
    bookkeeping in xr_environment_sampler.h) - stored cumulative, not
    each child's own bare probability, so a single random draw can walk
    the whole path (see that file's own header comment for why/how -
    a per-level draw would work too, but spawning LEVELS randoms per
    sample was deliberately avoided). Child c's own next node is
    n*4+1+c, the standard 1-indexed "complete k-ary tree as a flat
    array" scheme (root is node 0, its own children are nodes 1..4,
    their own children are 5..20, ...) - plain n*4+c would collapse any
    "always pick child 0" path back to node 0 at every depth (leading-
    zero ambiguity in base-4 positional numbering).

    Solid-angle-weighted for the XR (equirectangular, alpha in [-pi,pi] x
    beta in [-pi/2,pi/2]) parametrization xr2dir/dir2xr already use, not
    raw image pixel counts - a pixel near a pole covers much less solid
    angle than one near the equator for the same (du, dv) extent, so its
    own sampling probability must be scaled down to match (row weight
    below), or directions there would be *oversampled* relative to how
    much they can ever actually contribute.
    """
    import torch
    res = 1 << levels
    # Resized on-GPU via rdv's own linear resampler (RegularGrid's own
    # per-cell center coordinate >> a linear GridSampler over image,
    # wrapped as a plain TensorGrid) - no torch.nn.functional/interpolate
    # involved.
    resized = (_maps.RegularGrid([res, res], mode=1) >> _maps.GridSampler(_maps.TensorGrid(image), clamp=True, aligned=False, linear=True))()
    density = resized.as_tensor('torch').sum(-1).clamp_min(0)

    # row r spans v in [r/res, (r+1)/res] -> beta in [(v*2-1)*pi/2] - solid
    # angle per row (integrating uniform alpha over the row, dropping
    # constants that cancel out once normalized) is proportional to
    # sin(beta1) - sin(beta0).
    rows = torch.arange(res, dtype=torch.float32)
    v0, v1 = rows / res, (rows + 1) / res
    beta0, beta1 = (v0 * 2 - 1) * (_math.pi / 2), (v1 * 2 - 1) * (_math.pi / 2)
    row_weight = torch.sin(beta1) - torch.sin(beta0)
    density = (density.cpu() * row_weight[:, None]).clamp_min(0)
    density = density / density.sum().clamp_min(1e-12)

    # mass pyramid: masses[0] is the full (leaf) resolution, masses[d] is
    # masses[d-1] 2x2-block-summed (one level coarser); masses[levels] is
    # the 1x1 root (~total mass 1). Reversed after, so masses[d] ends up
    # holding the resolution-2^d grid for tree depth d (masses[0] = root).
    masses = [density]
    cur = density
    for _ in range(levels):
        cur = cur.reshape(cur.shape[0] // 2, 2, cur.shape[1] // 2, 2).sum(dim=(1, 3))
        masses.append(cur)
    masses = masses[::-1]

    total_slots = 4 * ((4 ** levels - 1) // 3)
    buf = torch.zeros(total_slots, dtype=torch.float32)

    def morton_flatten(t, d):
        # t: (2**d, 2**d). Node id at depth d, k-th in *Morton* (Z-order,
        # recursive-quadtree) order, is base(d)+k, base(d) = (4**d-1)//3
        # - not row-major k=pr*2**d+pc - because the old loop's own id
        # formula (id = pid*4+1+c, applied depth by depth from the root)
        # expands to k = sum_i c_i * 4**(d-1-i), i.e. each level's own
        # 2-bit child selector c_i = 2*row_bit_i+col_bit_i becomes one
        # base-4 digit of k, MSB (root-adjacent) first - exactly Morton
        # order interleaving (row_bit_i, col_bit_i) pairs from the most
        # to the least significant bit of (pr, pc). Splitting each axis
        # into its own d bits (row-major reshape - see the mass-pyramid
        # comment above for why that's the MSB-first order) then
        # interleaving row/col bit-pairs and flattening reproduces that
        # exact ordering, vectorized.
        if d == 0:
            return t.reshape(1)
        t = t.reshape(*([2] * d), *([2] * d))
        perm = [i for pair in zip(range(d), range(d, 2 * d)) for i in pair]
        return t.permute(*perm).reshape(4 ** d)

    # Vectorized per level instead of a per-node Python loop (the latter
    # does up to 5*(4**levels-1)/3 individual .item() calls for
    # levels=8 - ~110k calls, ~1.5-3s of pure interpreter/indexing
    # overhead dwarfing the actual mass-pyramid reduction above, which is
    # already vectorized and takes ~1-2ms). Each depth's own nodes occupy
    # one contiguous buf slice (base(d)*4 .. base(d)*4+4*4**d), and in
    # Morton order a parent's 4 children are always contiguous at the
    # next depth (appending one more base-4 digit c to Morton index k
    # gives child index 4k+c) - so child_flat.reshape(n, 4) already
    # groups each parent's own 4 children in c=0..3 order with no extra
    # reshuffling needed.
    base = 0
    for d in range(levels):
        parent_flat = morton_flatten(masses[d], d)
        children = morton_flatten(masses[d + 1], d + 1).reshape(4 ** d, 4)
        has_mass = (parent_flat > 1e-12).unsqueeze(-1)
        per_child = torch.where(has_mass, children / parent_flat.clamp_min(1e-12).unsqueeze(-1), torch.full_like(children, 0.25))
        cum = torch.cumsum(per_child, dim=-1)
        cum[:, 3] = 1.0  # forced exactly 1 on the last child - see this function's own docstring
        n = 4 ** d
        buf[4 * base: 4 * base + 4 * n] = cum.reshape(-1)
        base += n
    return buf


class XREnvironmentSampler(_maps.Map, Generic[Generics.ENVIRONMENT, Generics.LEVELS]):
    """
    Importance-samples a direction from an XR (equirectangular)
    environment map using a linearized quadtree of solid-angle-weighted
    pixel densities (see _build_xr_density_quadtree), instead of drawing
    a uniform direction and hoping it lands somewhere bright (see
    sandbox_pbvr.py's own previous `UniformDirection() >> (X | ...)`
    sampler) - noticeably lower variance for environments with
    concentrated bright regions (a sun, a window, ...), since directions
    toward bright regions get drawn far more often, each correspondingly
    down-weighted by 1/pdf so the estimator stays unbiased either way.

    Matches EnvironmentSamplerTrait's own Map[6, 3+SPECTRAL_DIM+1]
    interface directly and self-contained (its own IN - shading point/
    outgoing direction - is unused; environment sampling here doesn't
    depend on either) - drop-in for wherever `environment_sampler` is
    expected (e.g. PBVR_ENV's own constructor), no extra composition
    needed on the caller's own side.

    environment: an already-built Map[3, spectral_dim] (e.g.
    `XRProject() >> GPUSampler(environment_data, ...)`) - evaluated
    directly, at the sampled direction, to get this sample's own
    radiance. image: the *same* data environment itself samples from
    (own [H, W, spectral_dim] tensor) - used only once, up front, to
    build the quadtree; not read again afterward.
    """
    __code__ = "shaders/rendering/xr_environment_sampler.h"
    __requires__ = [RandomLibrary, MathLibrary]
    environment: Generics.ENVIRONMENT
    densities: Buffer[Type.FLOAT32]

    def __init__(self, environment, image, levels: int = 8) -> None:
        env_in = Generics.IN(type(environment))
        env_out = Generics.OUT(type(environment))
        assert env_in == 3, "environment must be a Map[3, spectral_dim] (takes a direction)"
        assert env_out is not None, "environment's own OUT (spectral dimension) must be resolvable"
        self.match(Generics.IN, 6)
        self.match(Generics.OUT, 3 + env_out + 1)
        self.match(Generics.ENVIRONMENT, type(environment))
        self.match(Generics.LEVELS, levels)
        self.environment = environment
        self.densities = wrap(_build_xr_density_quadtree(image, levels))


class OctProject(_maps.Map[3, 2]):
    __code__ = "shaders/rendering/oct_project.h"
    __requires__ = [
        MathLibrary
    ]


dir2oct = OctProject()
"""
Projects a 3D direction into its 2D octahedral coordinate.
"""


class OctUnproject(_maps.Map[2, 3]):
    __code__ = "shaders/rendering/oct_unproject.h"
    __requires__ = [
        MathLibrary
    ]


oct2dir = OctUnproject()
"""
Unprojects a 2D octahedral coordinate into its spherical position.
"""


class USCProject(_maps.Map[3, 2]):
    __code__ = "shaders/rendering/usc_project.h"
    __requires__ = [
        MathLibrary
    ]


dir2usc = USCProject()
"""
Projects a direction into its universal spherical coordinates.
"""


class USCUnproject(_maps.Map[2, 3]):
    __code__ = "shaders/rendering/usc_unproject.h"
    __requires__ = [
        MathLibrary
    ]


usc2dir = USCUnproject()
"""
Unprojects a 2D universal spherical coordinate into a 3D sphere position.
"""


def grid3d_fit_box(resolution: _typing.Tuple[int, int, int], align_corners: bool = True) -> tuple[vec3, vec3]:
    """
    Returns the bmin, bmax corners to fit a 3D grid of given resolution into the unit cube [-1,1]^3.
    """
    max_dim = max(resolution)
    if align_corners:
        scale = (vec3(
            resolution[2],
            resolution[1],
            resolution[0])-vec3(1.0))/vec3(float(max_dim-1))
    else:
        scale = vec3(
            resolution[2],
            resolution[1],
            resolution[0]) / vec3(float(max_dim))
    return -scale, scale


def unit2box(bmin: vec3, bmax: vec3) -> tuple[vec3, vec3]:
    """
    Returns the scale and translation to transform a cube [-1,1]^3 to the given bounding box.
    :param bmin: minimum corner of the box
    :param bmax: maximum corner of the box
    :return: scale and translation to transform [-1,1]^3 to the box defined by bmin and bmax
    """
    scale = (bmax - bmin)*0.5
    translate = (bmax + bmin)*0.5
    return scale, translate


def box2unit(bmin: vec3, bmax: vec3) -> tuple[vec3, vec3]:
    """
    Returns the scale and translation to transform a box defined by the given corners to the cube [-1,1]^3.
    :param bmin: minimum corner of the box
    :param bmax: maximum corner of the box
    :return: scale and translation to transform the box defined by bmin and bmax to [-1,1]^3
    """
    size = (bmax - bmin)
    scale = vec3(2.0) / size
    translate = -(bmax + bmin)/size
    return scale, translate


class AffineTransform3D(_maps.Map[3, 3]):
    __code__ = "shaders/rendering/affine_transform_3d.h"
    transform: mat4x3
    def __init__(self, transform):
        # transform's own field is mat4x3 (the affine 3 rows of a 4x4
        # transform), but a mat4 (e.g. straight from trs()) is a common,
        # convenient thing to pass here too - mat4x3(a_mat4) crops it to
        # shape (see _MatBase.__init__), dropping the trailing [0,0,0,1]
        # row a plain affine transform doesn't need. A mat4x3 passed in
        # directly round-trips through this unchanged.
        self.transform = mat4x3(transform)


# class Transform3DSampler(_maps.Map[Generics.IN: 3], Generic[Generics.S]):
#     """Applies an affine transform to a 3D input point, then samples the
#     wrapped `inner` map (whose own IN must be 3) at the transformed point -
#     e.g. rotating/translating/scaling a volumetric GridSampler/
#     TensorSampler/GPUSampler's coordinate space without touching its data."""
#     __code__ = "shaders/rendering/transform3d_prev_sampler.h"
#     inner: Generics.S
#     transform: mat4x3
#
#     def __init__(self, prev_sampler: Generics.S, transform: mat4x3 | None = None):
#         if transform is None:
#             transform = mat4x3(1.0, 0, 0, 0, 1, 0, 0.0, 0., 1., 0.0, 0.0, 0.0)
#         prev_sampler = prev_sampler.cast(_maps.Map[Generics.IN: 3])
#         self.match(Generics.OUT, Generics.OUT(prev_sampler))
#         self.match(Generics.S, type(prev_sampler))
#         self.inner = prev_sampler
#         self.transform = transform
