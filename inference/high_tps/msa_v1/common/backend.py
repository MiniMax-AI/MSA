"""C++ extension loading and FFI registration."""

import os


def _jit_compile_cpp_backend():
    """JIT-compile the C++ backend using torch.utils.cpp_extension.load()."""
    import tvm_ffi
    from torch.utils.cpp_extension import CUDA_HOME, load

    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csrc_dir = os.path.join(pkg_dir, "csrc")
    tvm_ffi_dir = os.path.dirname(tvm_ffi.__file__)

    extra_include = [
        csrc_dir,
        os.path.join(tvm_ffi_dir, "include"),
    ]
    if CUDA_HOME:
        extra_include.append(os.path.join(CUDA_HOME, "include"))
    cpp = load(
        name="_msa_v1_m3_decode_cpp",
        sources=[
            os.path.join(csrc_dir, name)
            for name in [
                "attn/api/attn_api.cpp",
                "attn/api/attn_validation.cpp",
                "common/api/bindings.cpp",
                "common/api/common_validation.cpp",
                "common/api/support.cpp",
                "common/runtime/m3_decode_variant_manager.cpp",
                "common/runtime/tp4_nvls_workspace.cpp",
            ]
        ],
        extra_include_paths=extra_include,
        extra_ldflags=[
            f"-L{os.path.join(tvm_ffi_dir, 'lib')}",
            "-ltvm_ffi",
            f"-Wl,-rpath,{os.path.join(tvm_ffi_dir, 'lib')}",
            "-lcuda",
        ],
        extra_cflags=["-std=c++20", "-O2"],
        verbose=False,
    )
    return cpp


_cpp = None


def _get_cpp():
    global _cpp
    if _cpp is None:
        _cpp = _jit_compile_cpp_backend()
    return _cpp


try:
    import tvm_ffi

    from . import jit as _jit


    def _jit_get_m3_decode_mega_atten_variant(wait_deadline=False, enable_placement=False):
        return _jit.get_m3_decode_mega_atten_variant(
            wait_deadline=bool(wait_deadline), enable_placement=bool(enable_placement)
        )._fn

    tvm_ffi.register_global_func(
        "msa_v1_m3_decode.jit_get_m3_decode_mega_atten_variant",
        _jit_get_m3_decode_mega_atten_variant,
        override=True,
    )
except ImportError:
    pass
