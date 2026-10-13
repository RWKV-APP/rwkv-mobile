"""Optional FP32 recurrent WKV kernel for 64-element RWKV7 heads."""

import torch
from coreai_torch import MetalParameter, TorchMetalKernel


def wkv_reference(state: torch.Tensor, decay: torch.Tensor, aa: torch.Tensor,
                  bb: torch.Tensor, vv: torch.Tensor, kk: torch.Tensor,
                  rr: torch.Tensor, steps: int, heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = []
    current = state.clone()
    for i in range(steps):
        current = current * decay[i:i+1] + current @ aa[i:i+1] @ bb[i:i+1] + vv[i:i+1] @ kk[i:i+1]
        outputs.append(current @ rr[i:i+1])
    return torch.cat(outputs, dim=0), current


def wkv_flat_reference(state: torch.Tensor, decay: torch.Tensor, aa: torch.Tensor,
                       bb: torch.Tensor, vv: torch.Tensor, kk: torch.Tensor,
                       rr: torch.Tensor, steps: int, heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    out, updated = wkv_reference(
        state.reshape(1, heads, 64, 64), decay.reshape(steps, heads, 1, 64),
        aa.reshape(steps, heads, 64, 1), bb.reshape(steps, heads, 1, 64),
        vv.reshape(steps, heads, 64, 1), kk.reshape(steps, heads, 1, 64),
        rr.reshape(steps, heads, 64, 1), steps, heads)
    return out.flatten(), updated.flatten()


# One SIMD group owns a state row for the whole chunk. Each lane retains two
# state values; reductions use simd_sum and memory accesses are contiguous.
WKV_KERNEL = TorchMetalKernel(
    name="rwkv7_wkv_row64_fp32",
    input_names=["state", "decay", "aa", "bb", "vv", "kk", "rr", "steps", "heads"],
    result_names=["out", "next_state"],
    torch_defn=wkv_flat_reference,
    metal_params=[MetalParameter("gid", "uint", "thread_position_in_grid")],
    src="""
    if (gid >= uint(heads * 64 * 32)) return;
    uint lane = gid % 32;
    uint row_id = gid / 32;
    uint head = row_id / 64;
    uint row = row_id % 64;
    uint base = row_id * 64;
    float s0 = state[base + lane];
    float s1 = state[base + lane + 32];
    for (int t = 0; t < steps; ++t) {
        uint vbase = (uint(t) * uint(heads) + head) * 64;
        uint j0 = vbase + lane;
        uint j1 = j0 + 32;
        float dot_a = simd_sum(s0 * aa[j0] + s1 * aa[j1]);
        float value = vv[vbase + row];
        s0 = s0 * decay[j0] + dot_a * bb[j0] + value * kk[j0];
        s1 = s1 * decay[j1] + dot_a * bb[j1] + value * kk[j1];
        float dot_r = simd_sum(s0 * rr[j0] + s1 * rr[j1]);
        if (lane == 0) out[vbase + row] = dot_r;
    }
    next_state[base + lane] = s0;
    next_state[base + lane + 32] = s1;
    """,
)


class CoreAIWKV(torch.nn.Module):
    def forward(self, state, decay, aa, bb, vv, kk, rr):
        steps, heads = decay.shape[:2]
        out, updated = WKV_KERNEL(
            state.flatten(), decay.flatten(), aa.flatten(), bb.flatten(),
            vv.flatten(), kk.flatten(), rr.flatten(), steps, heads,
            threads_per_grid=(heads * 64 * 32, 1, 1),
            threads_per_thread_group=(128, 1, 1),
            result_shapes=[[steps * heads * 64], [heads * 64 * 64]],
        )
        return out.reshape(steps, heads, 64, 1), updated.reshape(1, heads, 64, 64)


def use_fused_wkv(model, decode=False):
    for block in model.blocks:
        if block.att.head_size != 64:
            raise ValueError("Fused WKV currently requires head_size=64")
        block.att.coreai_wkv = CoreAIWKV()
        block.att.coreai_fused_decode = decode
