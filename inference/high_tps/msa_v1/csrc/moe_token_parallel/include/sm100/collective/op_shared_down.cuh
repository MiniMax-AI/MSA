#pragma once
#include "include/sm100/collective/expert_body.cuh"
namespace mega_moe::token_parallel {
// The producer acquires every shared up/gate publication before it issues the
// SharedMid TMA. The FULL mbarrier acquire carries that ordering to consumers,
// so repeating all sixteen global flag polls here is redundant.
__device__ inline void op_shared_down(Context &c, Args const &a, int role) {
  consume_shared_down_tile(c, a, role);
}
} // namespace mega_moe::token_parallel
