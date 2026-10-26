from ..core import *
from .. import maths as _maps
# from .rendering import Transform3DSampler
from .rendering import AffineTransform3D


class ExtinctionTrait(Code, Generic[Generics.EXTINCTION]):
    """
    Trait. Represents maps holding an extinction attribute with a Map[3,1].
    allows:
    float sigma_t = extinction(node, x);
    """
    __precode__ = "shaders/volumes/extinction_trait.h"
    extinction: Generics.EXTINCTION
    def __init__(self, *, extinction: Generics.EXTINCTION, **kwargs):
        extinction = extinction.cast(_maps.Map[3, 1])
        self.match(Generics.EXTINCTION, type(extinction))
        self.extinction = extinction
        super().__init__(**kwargs)


class EnvironmentSamplerTrait(Code, Generic[Generics.ENVIRONMENT_SAMPLER, Generics.SPECTRAL_DIM]):
    """
    Trait. Represents maps holding an environment_sampler with a Map[6, 3 + SPECTRAL_DIM + 1]
    allows:
    float weighted_environment[SPECTRAL_DIM];
    vec3 dir = environment_sampler(node, x, w, weighted_environment);
    """
    environment_sampler: Generics.ENVIRONMENT_SAMPLER
    __precode__ = "shaders/volumes/environment_sampler_trait.h"
    def __init__(self, *, environment_sampler: Generics.ENVIRONMENT_SAMPLER, **kwargs):
        spectral_dim = Generics.SPECTRAL_DIM(self)
        if spectral_dim is None:
            es_out = Generics.OUT(environment_sampler)
            if es_out is not None:
                spectral_dim = es_out - 4
        environment_sampler = environment_sampler.cast(_maps.Map[6, None if spectral_dim is None else spectral_dim + 4])
        self.match(Generics.ENVIRONMENT_SAMPLER, type(environment_sampler))
        self.match(Generics.SPECTRAL_DIM, spectral_dim)
        self.environment_sampler = environment_sampler
        super().__init__(**kwargs)


class EnvironmentTrait(Code, Generic[Generics.ENVIRONMENT, Generics.SPECTRAL_DIM]):
    """
    Trait. Represents maps holding an environment with a Map[3, SPECTRAL_DIM]
    allows:
    float env[SPECTRAL_DIM];
    environment(node, w, env);
    """
    environment: Generics.ENVIRONMENT
    __precode__ = "shaders/volumes/environment_trait.h"
    def __init__(self, *, environment: Generics.ENVIRONMENT, **kwargs):
        spectral_dim = Generics.SPECTRAL_DIM(self)
        if spectral_dim is None:
            environment_out = Generics.OUT(environment)
            if environment_out is not None:
                spectral_dim = environment_out
        environment = environment.cast(_maps.Map[3, spectral_dim])
        self.match(Generics.ENVIRONMENT, type(environment))
        self.match(Generics.SPECTRAL_DIM, spectral_dim)
        self.environment = environment
        super().__init__(**kwargs)


class ScatteringAlbedoTrait(Code, Generic[Generics.SCATTERING_ALBEDO, Generics.SPECTRAL_DIM]):
    """
    Trait. Represents maps holding a scattering_albedo with a Map[3, SPECTRAL_DIM]
    allows:
    float sa[SPECTRAL_DIM];
    scattering_albedo(node, x, sa);
    """
    scattering_albedo: Generics.SCATTERING_ALBEDO
    __precode__ = "shaders/volumes/scattering_albedo_trait.h"
    def __init__(self, *, scattering_albedo: Generics.SCATTERING_ALBEDO, **kwargs):
        spectral_dim = Generics.SPECTRAL_DIM(self)
        if spectral_dim is None:
            scattering_albedo_out = Generics.OUT(scattering_albedo)
            if scattering_albedo_out is not None:
                spectral_dim = scattering_albedo_out
        scattering_albedo = scattering_albedo.cast(_maps.Map[3, spectral_dim])
        self.match(Generics.SCATTERING_ALBEDO, type(scattering_albedo))
        self.match(Generics.SPECTRAL_DIM, spectral_dim)
        self.scattering_albedo = scattering_albedo
        super().__init__(**kwargs)


class MajorantTrait(Code, Generic[Generics.MAJORANT]):
    """
    Trait. Represents maps holding a majorant field with a Map[6, 2]
    allows:
    float majorant_distance;
    float majorant_value = majorant(node, x, w, majorant_distance);
    """
    majorant: Generics.MAJORANT
    __precode__ = "shaders/volumes/majorant_trait.h"
    def __init__(self, *, majorant: Generics.MAJORANT, **kwargs):
        majorant = _as_majorant_field(majorant)
        majorant = majorant.cast(_maps.Map[6, 2])
        self.match(Generics.MAJORANT, type(majorant))
        self.majorant = majorant
        super().__init__(**kwargs)


class RatiotrackingTransmittanceTrait(ExtinctionTrait, MajorantTrait):
    """
    Trait. Groups Extinction and Majorant traits.
    allows:
    float T = transmittance_rt (node, x, w, distance, density_scale);
    """
    __precode__ = "shaders/volumes/ratiotracking_transmittance_trait.h"


class RaymarchingTransmittanceTrait(ExtinctionTrait):
    """
        Trait. Groups Extinction traits.
        allows:
        float T = transmittance_rm (node, x, w, distance, step_size, density_scale);
        """
    __precode__ = "shaders/volumes/raymarching_transmittance_trait.h"



class RaymarchingTransmittance(_maps.Map[6, 1], RaymarchingTransmittanceTrait):
    __code__ = "shaders/volumes/raymarching_transmittance.h"
    __requires__ = [
        MathLibrary,
    ]
    step_size: float
    transform: mat4x3
    density_scale: float

    def __init__(self, extinction: Generics.EXTINCTION, transform: mat4x3 | None = None, step_size: float = 0.002,
                 density_scale: float = 1.0):
        super().__init__(extinction=extinction)
        self.match(Generics.IN, 6)
        self.match(Generics.OUT, 1)
        self.step_size = step_size
        self.transform = mat4x3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0) \
            if transform is None else mat4x3(transform)
        self.density_scale = density_scale


class RatiotrackingTransmittance(_maps.Map[6, 1], RatiotrackingTransmittanceTrait):
    """Like RaymarchingTransmittance, but via majorant-based ratio tracking
    (transmittance_rt - see RatiotrackingTransmittanceTrait) instead of a
    fixed-step raymarch: unbiased and typically cheaper for a sparse/empty-
    heavy medium (skips empty space via `majorant`), at the cost of some
    per-dispatch noise (see the Russian-roulette cutoff in
    shaders/volumes/ratiotracking_transmittance_trait.h)."""
    __code__ = "shaders/volumes/ratiotracking_transmittance.h"
    __requires__ = [
        RandomLibrary,
        MathLibrary,
    ]
    transform: mat4x3
    density_scale: float

    def __init__(self, extinction: Generics.EXTINCTION, majorant: Generics.MAJORANT,
                 transform: mat4x3 | None = None, density_scale: float = 1.0):
        super().__init__(extinction=extinction, majorant=majorant)
        self.match(Generics.IN, 6)
        self.match(Generics.OUT, 1)
        self.transform = mat4x3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0) \
            if transform is None else mat4x3(transform)
        self.density_scale = density_scale


# class MajorantField(Component):
#     """
#     A majorant field provides, for an object-space position `pos` and
#     (unnormalized) direction `dir`, both a majorant - an upper bound on the
#     extinction over some neighborhood of `pos` along `dir` - and, via
#     `max_step`, the (object-space, parametric) distance along `dir` for
#     which that bound is guaranteed to hold: the caller (delta-tracking
#     free-flight sampling - see PBVR/PBVRT) must re-query once it travels
#     (at least) that far without a collision. See
#     shaders/volumes/majorant_field_precode.h for the exact contract
#     (`local_majorant`) every implementation defines.
#
#     VoxelMajorantField (below) is the voxel-grid implementation - wraps a
#     Grid[3, 1] and walks it as an axis-aligned macrocell DDA. A field
#     bounding the medium with, say, a hierarchy of spheres instead (cheaper
#     to query, coarser away from the medium's actual shape) would be
#     another implementation of this same contract, and PBVR/PBVRT would not
#     need to change at all to use it.
#     """
#     __precode__ = "shaders/volumes/majorant_field_precode.h"


class VoxelMajorantField(_maps.Map[6, 2], Generic[Generics.G]):
    """
    Voxel-grid MajorantField: `grid` is a Grid[3, 1] (in local [-1, 1]^3
    box space) whose every cell must be an upper bound on the extinction
    over that cell's own sub-box, or the estimator is biased - a
    single-cell (shape [1, 1, 1]) grid holding one global value recovers
    the classic global-majorant algorithm; a real MajorantGrid (see
    rdv.maths), built from the density field at a coarser resolution,
    instead lets the tracker skip empty space. See
    shaders/volumes/voxel_majorant_field.h.
    """
    __code__ = "shaders/volumes/voxel_majorant_field.h"
    grid: Generics.G
    def __init__(self, grid):
        grid = grid.cast(_maps.Grid[3, 1])
        self.match(Generics.G, type(grid))
        self.grid = grid


class SphereMajorantField(_maps.Map[6, 2], Generic[Generics.G]):
    """
    MajorantField built from a SphereRadiusField grid (rdv.maths) - `grid`
    is a Grid[3, 2] holding, per voxel, (majorant, radius) where radius is
    in voxel-index units (see SphereRadiusField's own docstring for how
    it's built and its default level-selection heuristic).

    At query time (object-space position x, direction w), looks up the
    nearest voxel's own (majorant, radius) pair and returns, as the free-
    flight step cap, the remaining distance from x to that voxel's own
    safe-box boundary along w - not just to the voxel's own edge (see
    VoxelMajorantField) - so the tracker can skip potentially many voxels
    in a single step wherever the precomputed radius is large, instead of
    re-querying at every voxel boundary crossed. SphereRadiusField's own
    guarantee is really that `majorant` bounds a BOX of half-width radius
    (voxel-index units) centered on the voxel - a sphere of that radius is
    just a looser subset of it - so the exit distance is computed against
    that box directly, scaled per-axis by this grid's own voxel size, not
    collapsed to an isotropic sphere via e.g. the smallest per-axis voxel
    size (far too conservative once voxels aren't cubic - see
    shaders/volumes/sphere_majorant_field.h for why that matters).
    """
    __code__ = "shaders/volumes/sphere_majorant_field.h"
    grid: Generics.G
    def __init__(self, grid):
        grid = grid.cast(_maps.Grid[3, 2])
        self.match(Generics.G, type(grid))
        self.grid = grid


def _as_majorant_field(majorant):
    """
    PBVR/PBVRT's own `majorant` argument: already a MajorantField, use as
    given; otherwise (a bare Grid[3, 1], e.g. a plain TensorGrid or
    MajorantGrid - the previous, pre-MajorantField calling convention)
    wrap it as the voxel implementation, so existing callers built around
    a plain grid keep working unchanged.
    """
    if isinstance(majorant, _maps.Map):
        return majorant  # already a field as a map
    if isinstance(majorant, VoxelMajorantField):
        return majorant
    if isinstance(majorant, _maps.Grid) and Generics.OUT(majorant) == 1.0:
        return VoxelMajorantField(majorant)
    raise NotImplementedError()


class PBVR(_maps.Map[6, 3], Generic[Generics.S, Generics.E, Generics.MG]):
    """
    Physically based (unbiased, spectral/RGB) volume rendering via delta
    tracking (null-collision, spatial-majorant free-flight sampling) - see
    shaders/volumes/pbvr.h. Ported from rendervous.project's own
    volume_rendering/pbvr.h, minus next-event-estimation (no importance-
    sampled environment term - just a plain environment(direction) lookup
    for escaped rays): noisier per single dispatch than a biased
    ray-marcher, but unbiased - meant to be averaged over many dispatches
    (see the progressive accumulation in sandbox_pbvr.py), not read as a
    single sample.

    `s` is the extinction (density) field, a plain vec3 -> float map in
    local [-1, 1]^3 box space (no transform baked in - e.g. a bare
    GPUSampler, not one composed with AffineTransform3D); `transform` is
    this volume's own object-to-world matrix, applied once per ray here
    instead. `env` is a vec3 (world-space direction) -> vec3 (RGB radiance)
    map, e.g. XRProject() >> GPUSampler(environment_data). `albedo`/`g`
    (Henyey-Greenstein anisotropy) are spatially-uniform (a single tinted,
    single-phase-function medium).

    `majorant` is a MajorantField (also in local [-1, 1]^3 box space)
    providing an upper bound on `s` local to any queried position/direction
    - see MajorantField/VoxelMajorantField. A bare Grid[3, 1] (e.g. a
    single-cell one holding one global value, recovering the classic
    global-majorant algorithm; or a real MajorantGrid built from `s`'s own
    density field at a coarser resolution, letting the tracker skip empty
    space) is also accepted and wrapped as VoxelMajorantField for
    convenience.
    """
    __code__ = "shaders/volumes/pbvr.h"
    __requires__ = [
        RandomLibrary,
        MathLibrary,
    ]

    s: Generics.S
    env: Generics.E
    mg: Generics.MG
    transform: mat4x3
    albedo: vec3
    g: float
    max_bounces: int

    def __init__(self,
                 extinction: Generics.S,
                 environment: Generics.E,
                 majorant: Generics.MG,
                 transform: mat4x3 | None = None,
                 albedo=(1.0, 1.0, 1.0),
                 g: float = 0.0,
                 max_bounces: int = 32) -> None:
        extinction = extinction.cast(_maps.Map[3, 1])
        environment = environment.cast(_maps.Map[3, 3])
        majorant = _as_majorant_field(majorant)
        self.match(Generics.S, type(extinction))
        self.match(Generics.E, type(environment))
        self.match(Generics.MG, type(majorant))
        self.match(Generics.IN, 6)
        self.match(Generics.OUT, 3)
        self.s = extinction
        self.env = environment
        self.mg = majorant
        self.transform = mat4x3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0) \
            if transform is None else mat4x3(transform)
        self.albedo = vec3(*albedo) if not isinstance(albedo, vec3) else albedo
        self.g = g
        self.max_bounces = max_bounces


class PBVRT(_maps.Map, RatiotrackingTransmittanceTrait, ScatteringAlbedoTrait, EnvironmentTrait):
    """
    Like PBVR, but with an explicit transmittance/alpha channel and next-
    event-estimation - see shaders/volumes/pbvrt.h, ported from
    rendervous.project's own volume_rendering/pbvrt.h. NEE there samples an
    environment *importance sampler* (direction + value/pdf); here it's
    against a single fixed-direction light instead (light_direction/
    light_intensity) - a delta light needs no importance sampling/pdf, so
    its contribution is just throughput * light_intensity * phase * shadow
    transmittance (see transmittance_rt in the shader).

    OUT is SPECTRAL_DIM + 1: [0..SPECTRAL_DIM) is radiance, [SPECTRAL_DIM]
    is a transmittance/alpha flag - 1.0 while the ray hasn't scattered yet
    (still fully "background", meant for external alpha-compositing with a
    separately rendered backdrop), reset to 0.0 the moment it does (from
    then on this map has fully resolved the ray's own radiance itself,
    including whatever it sees of `env` after escaping post-scatter).

    `s`/`transform`/`g`/`max_bounces` mean the same as in PBVR. `majorant`
    is a MajorantField provider, same as PBVR's own (see its docstring) -
    used both by the main free-flight loop and by the NEE shadow ray
    (transmittance_rt). `albedo`/`light_intensity` are
    SPECTRAL_DIM-length (a generic spectral dimension - RGB is just
    SPECTRAL_DIM=3, not baked in).
    """
    __code__ = "shaders/volumes/pbvrt.h"
    __requires__ = [
        RandomLibrary,
        MathLibrary,
    ]
    transform: mat4x3
    light_direction: vec3
    light_intensity: Array[Generics.SPECTRAL_DIM, float]
    g: float
    density_scale: float
    max_bounces: int

    def __init__(self,
                 extinction: Generics.EXTINCTION,
                 majorant: Generics.MAJORANT,
                 environment: Generics.ENVIRONMENT,
                 scattering_albedo: Generics.SCATTERING_ALBEDO,
                 transform: mat4x3 | None = None,
                 light_direction=(0.0, 1.0, 0.0),
                 light_intensity=None,
                 density_scale: float = 1.0,
                 g: float = 0.0,
                 max_bounces: int = 32) -> None:
        super().__init__(extinction=extinction, majorant=majorant, environment=environment, scattering_albedo=scattering_albedo)
        self.match(Generics.IN, 6)
        spectral_dim: int = Generics.SPECTRAL_DIM(self)
        self.match(Generics.OUT, None if spectral_dim is None else spectral_dim + 1)
        self.transform = mat4x3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0) \
            if transform is None else mat4x3(transform)
        self.light_direction = vec3(*light_direction) if not isinstance(light_direction, vec3) else light_direction
        if spectral_dim is not None:
            self.light_intensity = [1.0] * spectral_dim if light_intensity is None else list(light_intensity)
        self.g = g
        self.density_scale = density_scale
        self.max_bounces = max_bounces


class PBVR_NEE(_maps.Map, RatiotrackingTransmittanceTrait, ScatteringAlbedoTrait):
    """
    Like PBVR, but with an explicit transmittance/alpha channel and next-
    event-estimation - see shaders/volumes/pbvrt.h, ported from
    rendervous.project's own volume_rendering/pbvrt.h. NEE there samples an
    environment *importance sampler* (direction + value/pdf); here it's
    against a single fixed-direction light instead (light_direction/
    light_intensity) - a delta light needs no importance sampling/pdf, so
    its contribution is just throughput * light_intensity * phase * shadow
    transmittance (see transmittance_rt in the shader).

    OUT is SPECTRAL_DIM + 1: [0..SPECTRAL_DIM) is radiance, [SPECTRAL_DIM]
    is a transmittance/alpha flag - 1.0 while the ray hasn't scattered yet
    (still fully "background", meant for external alpha-compositing with a
    separately rendered backdrop), reset to 0.0 the moment it does (from
    then on this map has fully resolved the ray's own radiance itself,
    including whatever it sees of `env` after escaping post-scatter).

    `s`/`transform`/`g`/`max_bounces` mean the same as in PBVR. `majorant`
    is a MajorantField provider, same as PBVR's own (see its docstring) -
    used both by the main free-flight loop and by the NEE shadow ray
    (transmittance_rt). `albedo`/`light_intensity` are
    SPECTRAL_DIM-length (a generic spectral dimension - RGB is just
    SPECTRAL_DIM=3, not baked in).
    """
    __code__ = "shaders/volumes/pbvr_nee.h"
    __requires__ = [
        RandomLibrary,
        MathLibrary,
    ]
    transform: mat4x3
    light_direction: vec3
    g: float
    density_scale: float
    max_bounces: int

    def __init__(self,
                 extinction: Generics.EXTINCTION,
                 majorant: Generics.MAJORANT,
                 scattering_albedo: Generics.SCATTERING_ALBEDO,
                 transform: mat4x3 | None = None,
                 light_direction=(0.0, 1.0, 0.0),
                 density_scale: float = 1.0,
                 g: float = 0.0,
                 max_bounces: int = 32) -> None:
        super().__init__(extinction=extinction, majorant=majorant, scattering_albedo=scattering_albedo)
        self.match(Generics.IN, 6)
        spectral_dim: int = Generics.SPECTRAL_DIM(self)
        self.match(Generics.OUT, spectral_dim)
        self.transform = mat4x3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0) \
            if transform is None else mat4x3(transform)
        self.light_direction = vec3(*light_direction) if not isinstance(light_direction, vec3) else light_direction
        self.g = g
        self.density_scale = density_scale
        self.max_bounces = max_bounces


class PBVR_ENV(_maps.Map, RatiotrackingTransmittanceTrait, ScatteringAlbedoTrait, EnvironmentSamplerTrait):
    """
    Like PBVR, but with an explicit transmittance/alpha channel and next-
    event-estimation - see shaders/volumes/pbvrt.h, ported from
    rendervous.project's own volume_rendering/pbvrt.h. NEE there samples an
    environment *importance sampler* (direction + value/pdf); here it's
    against a single fixed-direction light instead (light_direction/
    light_intensity) - a delta light needs no importance sampling/pdf, so
    its contribution is just throughput * light_intensity * phase * shadow
    transmittance (see transmittance_rt in the shader).

    OUT is SPECTRAL_DIM + 1: [0..SPECTRAL_DIM) is radiance, [SPECTRAL_DIM]
    is a transmittance/alpha flag - 1.0 while the ray hasn't scattered yet
    (still fully "background", meant for external alpha-compositing with a
    separately rendered backdrop), reset to 0.0 the moment it does (from
    then on this map has fully resolved the ray's own radiance itself,
    including whatever it sees of `env` after escaping post-scatter).

    `s`/`transform`/`g`/`max_bounces` mean the same as in PBVR. `majorant`
    is a MajorantField provider, same as PBVR's own (see its docstring) -
    used both by the main free-flight loop and by the NEE shadow ray
    (transmittance_rt). `albedo`/`light_intensity` are
    SPECTRAL_DIM-length (a generic spectral dimension - RGB is just
    SPECTRAL_DIM=3, not baked in).
    """
    __code__ = "shaders/volumes/pbvr_env.h"
    __requires__ = [
        RandomLibrary,
        MathLibrary,
    ]
    transform: mat4x3
    environment_sampler: Generics.ENVIRONMENT_SAMPLER
    g: float
    density_scale: float
    max_bounces: int

    def __init__(self,
                 extinction: Generics.EXTINCTION,
                 majorant: Generics.MAJORANT,
                 scattering_albedo: Generics.SCATTERING_ALBEDO,
                 environment_sampler: Generics.ENVIRONMENT_SAMPLER,
                 transform: mat4x3 | None = None,
                 density_scale: float = 1.0,
                 g: float = 0.0,
                 max_bounces: int = 32) -> None:
        super().__init__(extinction=extinction, majorant=majorant, scattering_albedo=scattering_albedo, environment_sampler=environment_sampler)
        self.match(Generics.IN, 6)
        spectral_dim: int = Generics.SPECTRAL_DIM(self)
        self.match(Generics.OUT, spectral_dim)
        self.transform = mat4x3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0) \
            if transform is None else mat4x3(transform)
        self.g = g
        self.density_scale = density_scale
        self.max_bounces = max_bounces



class VolumeProfile(_maps.Map,
                    RatiotrackingTransmittanceTrait, RaymarchingTransmittanceTrait,
                    ScatteringAlbedoTrait, EnvironmentTrait,
                    Generic[Generics.DEPTH_SAMPLES], # also SPECTRAL_DIM inherited from traits
                    ):
    __code__ = "shaders/volumes/volume_profile.h"
    __requires__ = [
        RandomLibrary,
        MathLibrary,
    ]
    density_scale: float
    transform: mat4x3
    light_direction: vec3
    light_intensity: Array[Generics.SPECTRAL_DIM, float]
    g: float
    transmittance_threshold: Array[Generics.DEPTH_SAMPLES, float]
    sample_index: int
    environment_samples: int
    step_size: float
    nee_near_distance: float

    def __init__(self,
                 extinction: Generics.EXTINCTION,
                 majorant: Generics.MAJORANT,
                 environment: Generics.ENVIRONMENT,
                 scattering_albedo: Generics.SCATTERING_ALBEDO,
                 transmittance_threshold: list[float],
                 density_scale: float = 1.0,
                 transform: mat4x3 | None = None,
                 light_direction=(0.0, 1.0, 0.0),
                 light_intensity=None,
                 g: float = 0.0,
                 step_size: float = 0.001,
                 environment_samples: int = 4,
                 sample_index: int = 0,
                 nee_near_distance: float = 0.05) -> None:
        super().__init__(extinction=extinction, majorant=majorant, environment=environment,
                         scattering_albedo=scattering_albedo)
        depth_samples = len(transmittance_threshold)
        self.match(Generics.IN, 6)
        self.match(Generics.DEPTH_SAMPLES, depth_samples)
        spectral_dim: int = Generics.SPECTRAL_DIM(self)
        self.match(Generics.OUT, None if spectral_dim is None else 4 + spectral_dim*3 + depth_samples)
        self.transform = mat4x3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0) \
            if transform is None else mat4x3(transform)
        self.light_direction = vec3(*light_direction) if not isinstance(light_direction, vec3) else light_direction
        if spectral_dim is not None:
            self.light_intensity = [1.0] * spectral_dim if light_intensity is None else list(light_intensity)
        self.density_scale = density_scale
        self.g = g
        self.step_size = step_size
        self.environment_samples = environment_samples
        self.sample_index = sample_index
        self.transmittance_threshold = transmittance_threshold
        self.nee_near_distance = nee_near_distance


class VolumeProfileDDA(VolumeProfile, Generic[Generics.MG]):
    """Like VolumeProfile, but the primary (viewing) ray's own tau/depth-
    threshold marching walks `majorant_grid` cell-by-cell via a real
    Amanatides-Woo DDA (see shaders/volumes/volume_profile_dda.h) instead
    of VolumeProfile's own repeated majorant()/MajorantField re-queries
    (which only ever trim the ray's two ends - see that file's own comment
    on why interleaving it into every fixed-step iteration too was tried
    and rolled back). `majorant` (the MajorantField, still required - see
    VolumeProfile.__init__) is unchanged and still used for everything
    else (NEE/shadow-ray transmittance); `majorant_grid` is an *additional*,
    separate index -> max_value Grid[3, 1] - e.g. the raw output of
    rdv.MajorantGrid(...) wrapped in a TensorGrid - purely for the primary
    ray's own DDA traversal. Meant to be compared side by side against
    plain VolumeProfile on the same scene (see
    sandbox_volume_profile_interactive.py)."""
    __code__ = "shaders/volumes/volume_profile_dda.h"
    majorant_grid: Generics.MG

    def __init__(self,
                 extinction: Generics.EXTINCTION,
                 majorant: Generics.MAJORANT,
                 majorant_grid: Generics.MG,
                 environment: Generics.ENVIRONMENT,
                 scattering_albedo: Generics.SCATTERING_ALBEDO,
                 transmittance_threshold: list[float],
                 density_scale: float = 1.0,
                 transform: mat4x3 | None = None,
                 light_direction=(0.0, 1.0, 0.0),
                 light_intensity=None,
                 g: float = 0.0,
                 step_size: float = 0.001,
                 environment_samples: int = 4,
                 sample_index: int = 0,
                 nee_near_distance: float = 0.05) -> None:
        super().__init__(
            extinction=extinction, majorant=majorant, environment=environment,
            scattering_albedo=scattering_albedo, transmittance_threshold=transmittance_threshold,
            density_scale=density_scale, transform=transform, light_direction=light_direction,
            light_intensity=light_intensity, g=g, step_size=step_size,
            environment_samples=environment_samples, sample_index=sample_index,
            nee_near_distance=nee_near_distance)
        majorant_grid = majorant_grid.cast(_maps.Grid[3, 1])
        self.match(Generics.MG, type(majorant_grid))
        self.majorant_grid = majorant_grid

