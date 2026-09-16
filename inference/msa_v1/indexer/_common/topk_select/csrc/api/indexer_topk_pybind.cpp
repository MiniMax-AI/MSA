#include <torch/extension.h>

#include "indexer_topk_api.hpp"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  using namespace minimax::msa_v1::indexer::topk;
  module.def("_run", &indexer_topk_run, py::arg("scores"),
             py::arg("lengths"), py::arg("out"),
             "Run the standalone SM100-family indexer TopK");
}
