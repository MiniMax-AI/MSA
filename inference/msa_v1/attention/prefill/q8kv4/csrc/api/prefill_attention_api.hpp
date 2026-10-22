#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace minimax::msa_v1::attention::prefill::q8kv4 {

void prefill_run(torch::Tensor q, torch::Tensor packed_k, torch::Tensor packed_v,
                 torch::Tensor k_scale, torch::Tensor v_scale, torch::Tensor page_table,
                 torch::Tensor cu_seqlens_q, torch::Tensor cu_seqlens_k, torch::Tensor k2q_row_ptr,
                 torch::Tensor qsplit_indices, torch::Tensor scheduler_metadata,
                 torch::Tensor work_count, torch::Tensor o_partial, torch::Tensor lse_partial,
                 double softmax_scale,
                 c10::optional<torch::Tensor> combine_tile_counters = c10::nullopt);

// Memory locality domains of a device (Rubin uGPU, Blackwell's dies; one elsewhere).
int64_t locality_domain_count(int64_t device);
// Allocation granularity of domain-localized memory, the page at which a striped buffer
// alternates domains.
int64_t locality_page_bytes(int64_t device);
// The locality domain of every SM id (%smid), -1 where unknown.
std::vector<int64_t> sm_locality_domains(int64_t device);
// A uint8 device buffer whose pages alternate between the locality domains: page p (of
// locality_page_bytes) lives in domain p % locality_domain_count. bytes must be a multiple of
// the page size times the domain count.
torch::Tensor alloc_locality_striped(int64_t bytes, int64_t device);

} // namespace minimax::msa_v1::attention::prefill::q8kv4
