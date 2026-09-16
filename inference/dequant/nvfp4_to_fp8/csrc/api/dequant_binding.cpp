#include <torch/extension.h>

#include "dequant_api.hpp"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  using namespace minimax::inference::dequant;
  module.def("_run_dense", &dequantize_nvfp4_to_fp8, py::arg("packed_nvfp4"), py::arg("scale"),
             py::arg("out"), "Convert contiguous NVFP4 rows to E4M3 FP8");
  module.def("_run_sparse", &dequantize_sparse_paged_nvfp4_to_fp8, py::arg("packed_k"),
             py::arg("packed_v"), py::arg("k_scale"), py::arg("v_scale"), py::arg("pair_keys"),
             py::arg("out_k"), py::arg("out_v"),
             "Convert selected physical-page and KV-head pairs to E4M3 FP8");
  module.def("_compiled_backend", &compiled_backend,
             "Return the CUDA-version-selected implementation");
}
