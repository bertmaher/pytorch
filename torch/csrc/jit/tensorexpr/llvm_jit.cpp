#ifdef ENABLE_LLVM

#include "torch/csrc/jit/tensorexpr/llvm_jit.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include "llvm/ExecutionEngine/Orc/LLJIT.h"

namespace llvm {
namespace orc {

// Lightly modified implementation from LLVM's Kaleidoscope JIT tutorial:
// https://llvm.org/docs/tutorial/BuildingAJIT1.html
class TORCH_API PytorchLLVMJITImpl {
 private:
  std::unique_ptr<LLJIT> LLJ;

 public:
  PytorchLLVMJITImpl() : LLJ(cantFail(LLJITBuilder().create())) {
    // Handle type-overloaded std:: functions
    using ffptr = float (*)(float);

    // Handle platform-specific symbol mangling
    MangleAndInterner Mangle(LLJ->getExecutionSession(), LLJ->getDataLayout());

    // Register implementations of intrinsics
    cantFail(LLJ->defineAbsolute(
        *Mangle("log10_float"),
        {llvm::pointerToJITTargetAddress(ffptr(&std::log10)), {}}));
  }

  Error addModule(ThreadSafeModule M) {
    if (auto Err = LLJ->addIRModule(std::move(M))) {
      return Err;
    }
    return Error::success();
  }

  JITSymbol findSymbol(const std::string Name) {
    return cantFail(LLJ->lookup(Name));
  }

  const DataLayout& getDataLayout() {
    return LLJ->getDataLayout();
  }
};

PytorchLLVMJIT::PytorchLLVMJIT()
    : impl_(std::make_unique<PytorchLLVMJITImpl>()) {}

PytorchLLVMJIT::~PytorchLLVMJIT() = default;

Error PytorchLLVMJIT::addModule(ThreadSafeModule M) {
  return impl_->addModule(std::move(M));
}

JITSymbol PytorchLLVMJIT::findSymbol(const std::string Name) {
  return impl_->findSymbol(std::move(Name));
}

const DataLayout& PytorchLLVMJIT::getDataLayout() {
  return impl_->getDataLayout();
}

} // end namespace orc
} // end namespace llvm

#endif // ENABLE_LLVM
