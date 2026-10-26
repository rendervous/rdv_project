"""rdv.Tensor.cast() reinterpreting the same underlying buffer as a different
scalar type (byte-for-byte, not value conversion), verified round-trip
through an actual GPU dispatch. Needs a Vulkan device."""
import pytest

torch = pytest.importorskip("torch")
import rdv

pytestmark = pytest.mark.gpu

T = rdv.Generics.T
N = rdv.Generics.N


class ReinterpretCopy(rdv.Main, rdv.Generic[T, N]):
    """Copies src (a Tensor[T, N]) into dst (a flat Buffer[T]) element by
    element, with no conversion."""
    src: rdv.Tensor[rdv.Generics.T:T, rdv.Generics.IN:N]
    dst: rdv.Buffer[T]
    __local_size__ = (32, 1, 1)
    __code__ = """
void rdv_main(RDV_NODE node, uvec3 idx) {
    int indices[N];
    global_to_indices(deref(node).src.shape, int(idx.x), indices);
    deref(deref(node).dst)[idx.x] = get_value(deref(node).src, indices);
}
"""

    def __init__(self, src, dst):
        src_type = type(src)
        self.match(T, src_type.generic_value(T))
        self.match(N, src_type.generic_value(rdv.Generics.IN))
        self.src = src
        self.dst = dst


def _gpu_reinterpret(torch_data, target_type, target_dtype):
    data = rdv.RDVObject(torch_data)
    generic = rdv.Tensor[rdv.Generics.IN:torch_data.dim()](data, list(torch_data.shape))
    view = generic.cast(rdv.Tensor[rdv.Generics.T:target_type])
    out_data = rdv.RDVObject(torch.zeros(torch_data.numel(), dtype=target_dtype, device="cuda"))
    out_buf = rdv.Buffer[target_type](out_data)
    rdv.dispatch(ReinterpretCopy(view, out_buf), torch_data.numel())
    return out_data.as_tensor("torch")


def test_float_to_int_reinterpret_matches_torch_view():
    float_data = torch.arange(4, dtype=torch.float32, device="cuda") + 1.0
    gpu_result = _gpu_reinterpret(float_data, rdv.Type.INT32, torch.int32)
    assert torch.equal(gpu_result, float_data.view(torch.int32))


def test_int_to_float_reinterpret_matches_torch_view():
    float_data = torch.arange(4, dtype=torch.float32, device="cuda") + 1.0
    int_data = float_data.view(torch.int32).clone()
    gpu_result = _gpu_reinterpret(int_data, rdv.Type.FLOAT32, torch.float32)
    assert torch.equal(gpu_result, int_data.view(torch.float32))


def test_tensor_as_type_reinterprets_not_just_relabels():
    torch.manual_seed(0)
    t = torch.randn(8, device="cuda")
    wrapped = rdv.wrap(t)
    generic_tensor = rdv.Tensor[rdv.Generics.IN:1](wrapped, shape=[8])

    as_float = generic_tensor.cast(rdv.Tensor[T:rdv.Type.FLOAT32]).as_type("torch")
    as_int = generic_tensor.cast(rdv.Tensor[T:rdv.Type.INT32]).as_type("torch")

    assert torch.equal(as_float, t)
    assert torch.equal(as_int, t.view(torch.int32))
