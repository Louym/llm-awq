from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import torch
def has_sm_120():
    try:
        device_name = torch.cuda.get_device_name(0).lower()
        return "blackwell" in device_name or "b100" in device_name or "b200" in device_name or "5090" in device_name
    except:
        return False

extra_compile_args = {
    "cxx": ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"],
    "nvcc": [
        "-O3",
        "-std=c++17",
        "-DENABLE_BF16",  # TODO
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "--threads=8",
    ],
}
if has_sm_120():
    print("*"*80)
    print("Adding sm_120 support, ensuring that your cuda version is at least 12.8!")
    extra_compile_args["nvcc"].append("-gencode=arch=compute_120,code=sm_120")
    print("*"*80)
setup(
    name="awq_inference_engine",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="awq_inference_engine",
            sources=[
                "csrc/pybind.cpp",
                "csrc/quantization/gemm_cuda_gen.cu",
                "csrc/quantization/gemv_cuda.cu",
                "csrc/quantization_new/gemv/gemv_cuda.cu",
                "csrc/quantization_new/gemm/gemm_cuda.cu",
                "csrc/layernorm/layernorm.cu",
                "csrc/position_embedding/pos_encoding_kernels.cu",
                "csrc/attention/ft_attention.cpp",
                "csrc/attention/decoder_masked_multihead_attention.cu",
                "csrc/rope_new/fused_rope_with_pos.cu",
                "csrc/w8a8/w8a8_gemm_cuda.cu",
                "csrc/w8a8/quantization.cu",
                "csrc/w8a8/act.cu",
                "csrc/w8a8/norm.cu"
            ],
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"],
)
