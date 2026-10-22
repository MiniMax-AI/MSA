#pragma once

#include <cuda.h>
#include <torch/torch.h>

#include <cstddef>
#include <cstdint>

namespace mega_atten::m3_decode {

class Tp4NvlsWorkspace {
public:
  Tp4NvlsWorkspace(int device, size_t requested_size, int world_size, bool create_multicast);
  Tp4NvlsWorkspace(Tp4NvlsWorkspace const &) = delete;
  Tp4NvlsWorkspace &operator=(Tp4NvlsWorkspace const &) = delete;
  ~Tp4NvlsWorkspace();

  int export_multicast_fd() const;
  void import_multicast_fd(int fd);
  void add_device();
  void map_and_bind();
  void zero();
  void close();

  int device() const { return device_; }
  int world_size() const { return world_size_; }
  size_t size() const { return size_; }
  uintptr_t unicast_ptr() const { return unicast_va_; }
  uintptr_t multicast_ptr() const { return multicast_va_; }
  at::Tensor next_hidden() const;
  bool ready() const { return bound_ && !closed_; }

private:
  void set_access(CUdeviceptr address);
  void release_resources() noexcept;

  int device_ = -1;
  int world_size_ = 0;
  size_t size_ = 0;
  CUdevice cu_device_ = 0;
  CUmulticastObjectProp multicast_prop_{};
  CUmemGenericAllocationHandle unicast_handle_ = 0;
  CUmemGenericAllocationHandle multicast_handle_ = 0;
  CUdeviceptr unicast_va_ = 0;
  CUdeviceptr multicast_va_ = 0;
  bool owns_created_multicast_ = false;
  bool device_added_ = false;
  bool bound_ = false;
  bool closed_ = false;
};

} // namespace mega_atten::m3_decode
