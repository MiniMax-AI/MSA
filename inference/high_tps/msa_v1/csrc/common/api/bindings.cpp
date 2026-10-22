// m3_decode_pybind.cpp - pybind11 bindings for the M3 decode kernels.
//
// Exposes PlanInfo and the M3 decode entry points to Python.
// The shared Python backend loads this module as _msa_v1_m3_decode_cpp.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "attn/api/api.hpp"
#include "common/api/support.hpp"
#include "common/runtime/tp4_nvls_workspace.hpp"

namespace py = pybind11;
using namespace mega_atten::m3_decode;

static PlanInfo py_m3_decode_plan(at::Tensor qo_segment_lens, at::Tensor kv_segment_lens,
                                  int64_t topk, std::optional<int> device) {
  return m3_decode_plan(qo_segment_lens, kv_segment_lens, topk, device);
}

static py::tuple py_m3_decode_mega_atten(
    at::Tensor q, at::Tensor k, at::Tensor v, PlanInfo &plan_info, at::Tensor kv_indices,
    std::optional<at::Tensor> kv_block_indexes, at::Tensor k_scale_tensor,
    at::Tensor v_scale_tensor, at::Tensor output_projection_weight,
    at::Tensor output_projection_weight_sf, int64_t tp_unicast_ptr, int64_t tp_multicast_ptr,
    int64_t tp_rank, at::Tensor residual_in, at::Tensor attention_residual,
    at::Tensor post_attention_norm, at::Tensor rms_weight, std::optional<at::Tensor> indexer_q,
    std::optional<at::Tensor> indexer_k, std::optional<at::Tensor> indexer_k_scale,
    std::optional<at::Tensor> indexer_page_table,
    std::optional<at::Tensor> attention_projection_input_norm_weight,
    std::optional<at::Tensor> attention_projection_weight,
    std::optional<at::Tensor> attention_projection_weight_sf,
    std::optional<at::Tensor> attention_projection_workspace_data,
    std::optional<at::Tensor> attention_projection_workspace_scale,
    std::optional<at::Tensor> attention_projection_grid_epoch,
    std::optional<at::Tensor> attention_projection_special_partial,
    std::optional<at::Tensor> attention_projection_special_epochs,
    std::optional<at::Tensor> attention_projection_positions,
    std::optional<at::Tensor> attention_projection_slot_mapping,
    std::optional<at::Tensor> attention_projection_q_norm_weight,
    std::optional<at::Tensor> attention_projection_k_norm_weight,
    std::optional<at::Tensor> attention_projection_qi_norm_weight,
    std::optional<at::Tensor> attention_projection_ki_norm_weight,
    std::optional<at::Tensor> attention_projection_cos_sin_cache,
    double attention_projection_input_norm_eps, double attention_projection_post_norm_eps,
    double attention_projection_ki_dequant_scale, int64_t topk, std::optional<at::Tensor> out,
    std::optional<at::Tensor> out_sf, std::optional<float> sm_scale, std::optional<float> q_scale,
    std::optional<float> k_scale, std::optional<float> v_scale, std::optional<float> o_scale,
    double rms_eps, int64_t wait_timeout_ns, bool enable_placement) {
  auto tensor_or_empty = [](std::optional<at::Tensor> const &tensor) {
    return tensor.has_value() ? *tensor : at::Tensor{};
  };
  AttentionProjectionInfo const attention_projection{
      tensor_or_empty(attention_projection_input_norm_weight),
      tensor_or_empty(attention_projection_weight),
      tensor_or_empty(attention_projection_weight_sf),
      tensor_or_empty(attention_projection_workspace_data),
      tensor_or_empty(attention_projection_workspace_scale),
      tensor_or_empty(attention_projection_grid_epoch),
      tensor_or_empty(attention_projection_special_partial),
      tensor_or_empty(attention_projection_special_epochs),
      tensor_or_empty(attention_projection_positions),
      tensor_or_empty(attention_projection_slot_mapping),
      tensor_or_empty(attention_projection_q_norm_weight),
      tensor_or_empty(attention_projection_k_norm_weight),
      tensor_or_empty(attention_projection_qi_norm_weight),
      tensor_or_empty(attention_projection_ki_norm_weight),
      tensor_or_empty(attention_projection_cos_sin_cache),
      attention_projection_input_norm_eps,
      attention_projection_post_norm_eps,
      attention_projection_ki_dequant_scale};
  run_m3_decode_mega_atten_impl(
      q, k, v, plan_info, topk, kv_indices,
      kv_block_indexes.has_value() ? *kv_block_indexes : at::Tensor{}, k_scale_tensor,
      v_scale_tensor, out, out_sf, sm_scale, q_scale, k_scale, v_scale, o_scale,
      output_projection_weight, output_projection_weight_sf, tp_unicast_ptr, tp_multicast_ptr,
      tp_rank, residual_in, attention_residual, post_attention_norm, rms_weight, indexer_q,
      indexer_k, indexer_k_scale, indexer_page_table, attention_projection, rms_eps,
      wait_timeout_ns, enable_placement);
  return py::make_tuple(attention_residual, post_attention_norm);
}

PYBIND11_MODULE(_msa_v1_m3_decode_cpp, m) {
  m.doc() = "SM100/SM103 M3 decode C++ backend";

  py::class_<PlanInfo>(m, "PlanInfo")
      .def_property_readonly("kv_segment_lens",
                             [](PlanInfo const &plan) { return plan.kv_segment_lens; })
      .def_property_readonly("output_projection_barrier",
                             [](PlanInfo const &plan) { return plan.output_projection_barrier; })
      .def_property_readonly("kv_block_num", [](PlanInfo const &plan) { return plan.kv_block_num; })
      .def_property_readonly("kv_lens_cpu", [](PlanInfo const &plan) { return plan.kv_lens_cpu; })
      .def_property_readonly("is_valid", &PlanInfo::is_valid);

  py::class_<Tp4NvlsWorkspace, std::shared_ptr<Tp4NvlsWorkspace>>(m, "Tp4NvlsWorkspace")
      .def(py::init<int, size_t, int, bool>(), py::arg("device"), py::arg("requested_size"),
           py::arg("world_size"), py::arg("create_multicast"))
      .def("export_multicast_fd", &Tp4NvlsWorkspace::export_multicast_fd)
      .def("import_multicast_fd", &Tp4NvlsWorkspace::import_multicast_fd)
      .def("add_device", &Tp4NvlsWorkspace::add_device)
      .def("map_and_bind", &Tp4NvlsWorkspace::map_and_bind)
      .def("zero", &Tp4NvlsWorkspace::zero)
      .def("close", &Tp4NvlsWorkspace::close)
      .def_property_readonly("device", &Tp4NvlsWorkspace::device)
      .def_property_readonly("world_size", &Tp4NvlsWorkspace::world_size)
      .def_property_readonly("size", &Tp4NvlsWorkspace::size)
      .def_property_readonly("unicast_ptr", &Tp4NvlsWorkspace::unicast_ptr)
      .def_property_readonly("multicast_ptr", &Tp4NvlsWorkspace::multicast_ptr)
      .def_property_readonly("next_hidden", &Tp4NvlsWorkspace::next_hidden)
      .def_property_readonly("ready", &Tp4NvlsWorkspace::ready);

  m.def("_m3_decode_plan", &py_m3_decode_plan, py::arg("qo_segment_lens"),
        py::arg("kv_segment_lens"), py::arg("topk") = 16, py::arg("device") = py::none());

  m.def("m3_decode_mega_atten", &py_m3_decode_mega_atten, py::arg("q"), py::arg("k"), py::arg("v"),
        py::arg("plan_info"), py::arg("kv_indices"), py::arg("kv_block_indexes"),
        py::arg("k_scale_tensor"), py::arg("v_scale_tensor"), py::arg("output_projection_weight"),
        py::arg("output_projection_weight_sf"), py::arg("tp_unicast_ptr"),
        py::arg("tp_multicast_ptr"), py::arg("tp_rank"), py::arg("residual_in"),
        py::arg("attention_residual"), py::arg("post_attention_norm"), py::arg("rms_weight"),
        py::arg("indexer_q") = py::none(), py::arg("indexer_k") = py::none(),
        py::arg("indexer_k_scale") = py::none(), py::arg("indexer_page_table") = py::none(),
        py::arg("attention_projection_input_norm_weight") = py::none(),
        py::arg("attention_projection_weight") = py::none(),
        py::arg("attention_projection_weight_sf") = py::none(),
        py::arg("attention_projection_workspace_data") = py::none(),
        py::arg("attention_projection_workspace_scale") = py::none(),
        py::arg("attention_projection_grid_epoch") = py::none(),
        py::arg("attention_projection_special_partial") = py::none(),
        py::arg("attention_projection_special_epochs") = py::none(),
        py::arg("attention_projection_positions") = py::none(),
        py::arg("attention_projection_slot_mapping") = py::none(),
        py::arg("attention_projection_q_norm_weight") = py::none(),
        py::arg("attention_projection_k_norm_weight") = py::none(),
        py::arg("attention_projection_qi_norm_weight") = py::none(),
        py::arg("attention_projection_ki_norm_weight") = py::none(),
        py::arg("attention_projection_cos_sin_cache") = py::none(),
        py::arg("attention_projection_input_norm_eps") = 0.0,
        py::arg("attention_projection_post_norm_eps") = 0.0,
        py::arg("attention_projection_ki_dequant_scale") = 1.0, py::arg("topk") = 16,
        py::arg("out") = py::none(), py::arg("out_sf") = py::none(),
        py::arg("sm_scale") = py::none(), py::arg("q_scale") = py::none(),
        py::arg("k_scale") = py::none(), py::arg("v_scale") = py::none(),
        py::arg("o_scale") = py::none(), py::arg("rms_eps") = 1e-6, py::arg("wait_timeout_ns") = 0,
        py::arg("enable_placement") = false);
}
