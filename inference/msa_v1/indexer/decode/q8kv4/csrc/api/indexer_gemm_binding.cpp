#include <torch/extension.h>

#include "indexer_gemm_api.hpp"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  using namespace minimax::msa_v1::indexer::decode::q8kv4;
  module.def("_run", &indexer_gemm_run, py::arg("q"), py::arg("packed_k_cache"), py::arg("k_scale"),
             py::arg("page_table"), py::arg("seq_lens"), py::arg("workspace"), py::arg("sm_count"),
             py::arg("out"), "Run SM100-family Q8KV4 decode indexer GEMM");
}
