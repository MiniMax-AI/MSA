#pragma once

#include <cuda_runtime.h>

#include "cutlass/cutlass.h"

namespace cutlass::fmha::device {

template <class Kernel_>
class FMHA {
 public:
  using Kernel = Kernel_;
  using Arguments = typename Kernel::Arguments;
  using Params = typename Kernel::Params;

  static int const kThreadCount = Kernel::MaxThreadsPerBlock;

 private:
  Params params_;

 public:
  Params const& params() const {
    return params_;
  }

  static cutlass::Status can_implement(Arguments const& args) {
    return Kernel::can_implement(args) ? cutlass::Status::kSuccess
                                       : cutlass::Status::kInvalid;
  }

  static size_t get_workspace_size(Arguments const& args) {
    return Kernel::get_workspace_size(args);
  }

  static dim3 get_grid_shape(Params const& params) {
    return Kernel::get_grid_shape(params);
  }

  cutlass::Status initialize(Arguments const& args,
                             void* workspace = nullptr,
                             cudaStream_t stream = nullptr) {
    cutlass::Status status =
        Kernel::initialize_workspace(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }
    params_ = Kernel::to_underlying_arguments(args, workspace);
    return cutlass::Status::kSuccess;
  }

  cutlass::Status update(Arguments const& args, void* workspace = nullptr) {
    params_ = Kernel::to_underlying_arguments(args, workspace);
    return cutlass::Status::kSuccess;
  }

  static cutlass::Status run(Params& params, cudaStream_t stream = nullptr) {
    cudaError_t status = Kernel::run(params, stream);
    return status == cudaSuccess ? cutlass::Status::kSuccess
                                 : cutlass::Status::kErrorInternal;
  }

  cutlass::Status run(Arguments const& args,
                      void* workspace = nullptr,
                      cudaStream_t stream = nullptr) {
    cutlass::Status status = initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }
    return run(params_, stream);
  }

  cutlass::Status operator()(Arguments const& args,
                             void* workspace = nullptr,
                             cudaStream_t stream = nullptr) {
    return run(args, workspace, stream);
  }

  cutlass::Status run(cudaStream_t stream = nullptr) {
    return run(params_, stream);
  }

  cutlass::Status operator()(cudaStream_t stream = nullptr) {
    return run(params_, stream);
  }
};

}  // namespace cutlass::fmha::device
