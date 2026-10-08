// PyBind interface for Q8KV4 paged sparse decode attention.

#include "decode_attention_api.hpp"

#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace py = pybind11;
using namespace minimax::msa_v1::attention::decode::q8kv4;

PYBIND11_MODULE(_minimax_msa_v1_attention_decode_q8kv4_cpp, module) {
  module.doc() = "SM100 Q8KV4 paged sparse decode attention backend";

  py::class_<PlanInfo, std::unique_ptr<PlanInfo>>(module, "_PlanHandle");

  module.def("plan_decode", &make_decode_plan, py::arg("qo_segment_lens"),
             py::arg("kv_segment_lens"), py::arg("num_qo_heads"), py::arg("num_kv_heads"),
             py::arg("num_kv_splits"), py::arg("page_size"), py::arg("topk"),
             py::arg("usable_sm_count"), py::arg("device") = py::none(),
             py::arg("split_mode") = "streamk");

  module.def("run_decode", &run_decode, py::arg("q"), py::arg("k"), py::arg("v"), py::arg("plan"),
             py::arg("seq_lens"), py::arg("page_table"), py::arg("topk_indices"),
             py::arg("k_scale"), py::arg("v_scale"), py::arg("out"), py::arg("out_scale"),
             py::arg("sm_scale"));
}
