"""
Curated public surface: only the concrete classes/functions used directly
to compose a graph (and the vk primitives needed alongside them), not every
internal symbol `rdv.core`/`rdv.math`/`rdv.domain.*` happen to define.
Anything else is still reachable through those modules directly (e.g.
`rdv.math.GridEvaluator`, `rdv.domain.volumes.ExtinctionLibrary`).
"""

from .core import (
    # Node-graph base classes, for authoring custom nodes.
    # Node,
    Main,
    Component,
    # Code,
    Library,
    Struct,
    Single,
    Array,
    Generic,
    # GenericParam,
    Generics,
    # RDVType,
    RDVObject,
    Pointer,
    Buffer,
    FixArray,
    Deferred,
    deferreds,
    # Library dependencies commonly listed in a custom node's __requires__.
    MathLibrary,
    RandomLibrary,
    GridLibrary,
    CooperativeLibrary,
    DeferredLibrary,
    # Wrapping/interop utilities.
    wrap,
    ensure_on_rdv,
    is_on_rdv,
    # Dispatch/profiling.
    dispatch,
    profiling,
    reset_submit_times,
    profiling_submit_times,
    mat2, mat3, mat4, mat3x4, mat4x3, vec2, vec3, vec4, ivec2, ivec3, ivec4, uvec2, uvec3,uvec4, bvec2, bvec3, bvec4
)

from .maths import (
    Grid,
    RefGrid,
    TensorGrid,
    RegularGrid,
    MajorantGrid,
    DilatedMaxPoolGrid,
    sphere_radius_field,
    TensorBase,
    Tensor,
    tensor,
    TensorOperation,
    Map,
    RefMap,
    Constant,
    Identity,
    X,
    # Composite,
    # Promote,
    # Select,
    Uniform,
    UniformDirection,
    Normal,
    GridSampler,
    TensorSampler,
    GPUSampler,
    SingleImageLibrary,
    SingleGPUSampler,
    SingleGPUGrid,
    barrier,
    average,
    pi,
    e,
    pi_over_2,
    pi_over_4,
    four_pi,
    one_over_four_pi,
    as_map
)

from .domain.rendering import (
    Cameras,
    XRProject,
    XRUnproject,
    XREnvironmentSampler,
    OctProject,
    OctUnproject,
    USCProject,
    USCUnproject,
    AffineTransform3D,
    grid3d_fit_box,
    unit2box,
    box2unit,
xr2dir,
dir2xr,
oct2dir,
dir2oct,
usc2dir,
dir2usc
)

from .domain.volumes import (
    RaymarchingTransmittance,
    RatiotrackingTransmittance,
    PBVR,
    PBVR_NEE,
    PBVR_ENV,
    PBVRT,
    VoxelMajorantField,
    SphereMajorantField,
    VolumeProfile,
    VolumeProfileDDA,
    ExtinctionTrait,
    RatiotrackingTransmittanceTrait,
    MajorantTrait,
    EnvironmentTrait,
    RaymarchingTransmittanceTrait,
    ScatteringAlbedoTrait,
    EnvironmentSamplerTrait
)

from .domain.nn import (
    Linear,
    LinearFast,
    linear,
    ReLU,
    relu,
    Conv2dGrid,
    conv2d,
    Conv2D,
    conv2d_o,
    Downsample2x,
    Upsample2x,
    ReLUTensor,
    LinearTensor,
    ReLUTensorOp,
    LinearTensorOp,
    LinearTensorOpFast,
    LinearTensorOpMine,
    LinearTensorOpSimple,
    LinearTensorOpSmall,
    Conv2DTensorOp,
    Conv2DTensorOpFast,
    Conv2DTensorOpSimple,
    Conv2DTensorOpSimple3x3,
    Conv2DTensorOpWindow3x3,
    Conv2DTensorOpWinograd3x3,
    Downsample2xTensorOp,
    Upsample2xTensorOp,
    ConcatTensorOp,
)

from vk.math3d import *
from vk import (
    MemoryLocation, Type, Format,
    window, Window, WindowMode,
    FloatState, IntState, BoolState, ColorState, StringState, CanvasState, MouseState,
    device, device_index, caps, DescriptorType,
)
from vk.vk import EngineType
