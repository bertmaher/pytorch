#ifdef ENABLE_LLVM
#pragma once

//#include <torch/csrc/jit/tensorexpr/kernel.h>
#include <string>
#include <unordered_map>

namespace torch {
namespace jit {
class Value;
namespace tensorexpr {
class Tensor;
class TensorExprKernel;
} // namespace tensorexpr
} // namespace jit
} // namespace torch

using TensorCreator = std::function<torch::jit::tensorexpr::Tensor*(
    torch::jit::tensorexpr::TensorExprKernel*,
    const torch::jit::Value* v)>;
std::unordered_map<std::string, std::pair<void*, TensorCreator>>&
getNativeFunctionRegistry();

struct RegisterNativeFunction {
  template <typename T>
  RegisterNativeFunction(std::string name, T* fn, TensorCreator cv) {
    getNativeFunctionRegistry()[name] =
        std::make_pair(reinterpret_cast<void*>(fn), cv);
  }
};

#endif // ENABLE_LLVM
