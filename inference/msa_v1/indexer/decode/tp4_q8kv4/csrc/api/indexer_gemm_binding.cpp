#include <torch/extension.h>

#include "indexer_gemm_api.hpp"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  using namespace minimax::msa_v1::indexer::decode::tp4_q8kv4;
  module.def("_workspace_size", &indexer_gemm_workspace_size,
             py::arg("batch_size"),
             "Return opaque scheduler workspace bytes");
  module.def("_plan", &indexer_gemm_plan, py::arg("page_table"),
             py::arg("seq_lens"), py::arg("workspace"),
             "Prepare decode indexer GEMM scheduler state");
  module.def("_run", &indexer_gemm_run, py::arg("q"),
             py::arg("packed_k_cache"), py::arg("k_scale"),
             py::arg("page_table"), py::arg("seq_lens"),
             py::arg("workspace"), py::arg("sm_count"), py::arg("out"),
             "Run SM100-family Q8KV4 decode indexer GEMM");
}
