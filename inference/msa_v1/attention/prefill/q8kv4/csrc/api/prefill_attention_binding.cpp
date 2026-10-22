#include <torch/extension.h>

#include "prefill_attention_api.hpp"

namespace q8kv4 = minimax::msa_v1::attention::prefill::q8kv4;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("run", &q8kv4::prefill_run, "SM100 Q8KV4 paged sparse-prefill K1", py::arg("q"),
             py::arg("packed_k"), py::arg("packed_v"), py::arg("k_scale"), py::arg("v_scale"),
             py::arg("page_table"), py::arg("cu_seqlens_q"), py::arg("cu_seqlens_k"),
             py::arg("k2q_row_ptr"), py::arg("qsplit_indices"), py::arg("scheduler_metadata"),
             py::arg("work_count"), py::arg("o_partial"), py::arg("lse_partial"),
             py::arg("softmax_scale"), py::arg("combine_tile_counters") = py::none());
  module.def("locality_domain_count", &q8kv4::locality_domain_count,
             "Memory locality domains of a device", py::arg("device"));
  module.def("locality_page_bytes", &q8kv4::locality_page_bytes,
             "Allocation granularity of domain-localized memory", py::arg("device"));
  module.def("sm_locality_domains", &q8kv4::sm_locality_domains,
             "Locality domain of every SM id, -1 where unknown", py::arg("device"));
  module.def("alloc_locality_striped", &q8kv4::alloc_locality_striped,
             "Device buffer whose pages alternate between the locality domains", py::arg("bytes"),
             py::arg("device"));
}
