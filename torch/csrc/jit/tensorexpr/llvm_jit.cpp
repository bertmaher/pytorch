#ifdef ENABLE_LLVM

#include "torch/csrc/jit/tensorexpr/llvm_jit.h"
#include "torch/csrc/jit/tensorexpr/native.h"

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
  MangleAndInterner Mangle;

 public:
  PytorchLLVMJITImpl()
      : LLJ(cantFail(LLJITBuilder().create())),
        Mangle(LLJ->getExecutionSession(), LLJ->getDataLayout()) {
    auto ProcSymbolsGenerator =
        cantFail(DynamicLibrarySearchGenerator::GetForCurrentProcess(
            LLJ->getDataLayout().getGlobalPrefix()));
    LLJ->getMainJITDylib().setGenerator(std::move(ProcSymbolsGenerator));
    // Handle platform-specific symbol mangling

    for (auto kv : getNativeFunctionRegistry()) {
      auto str = kv.first;
      auto func = kv.second.first;
      cantFail(LLJ->defineAbsolute(
          mangle(str), {llvm::pointerToJITTargetAddress(func), {}}));
    }

    // Register implementations of intrinsics
    cantFail(LLJ->defineAbsolute(
        mangle("log10f"), {llvm::pointerToJITTargetAddress(&log10f), {}}));
    cantFail(LLJ->defineAbsolute(
        mangle("logf"), {llvm::pointerToJITTargetAddress(&logf), {}}));
    cantFail(LLJ->defineAbsolute(
        mangle("log2f"), {llvm::pointerToJITTargetAddress(&log2f), {}}));
    cantFail(LLJ->defineAbsolute(
        mangle("expf"), {llvm::pointerToJITTargetAddress(&expf), {}}));
    cantFail(LLJ->defineAbsolute(
        mangle("erff"), {llvm::pointerToJITTargetAddress(&erff), {}}));
    cantFail(LLJ->defineAbsolute(
        mangle("cosf"), {llvm::pointerToJITTargetAddress(&cosf), {}}));
    cantFail(LLJ->defineAbsolute(
        mangle("sinf"), {llvm::pointerToJITTargetAddress(&sinf), {}}));
    cantFail(LLJ->defineAbsolute(
        mangle("tanf"), {llvm::pointerToJITTargetAddress(&tanf), {}}));
    cantFail(LLJ->defineAbsolute(
        mangle("acosf"), {llvm::pointerToJITTargetAddress(&acosf), {}}));
    cantFail(LLJ->defineAbsolute(
        mangle("asinf"), {llvm::pointerToJITTargetAddress(&asinf), {}}));
    cantFail(LLJ->defineAbsolute(
        mangle("atanf"), {llvm::pointerToJITTargetAddress(&atanf), {}}));
    cantFail(LLJ->defineAbsolute(
        mangle("coshf"), {llvm::pointerToJITTargetAddress(&coshf), {}}));
    cantFail(LLJ->defineAbsolute(
        mangle("sinhf"), {llvm::pointerToJITTargetAddress(&sinhf), {}}));
    cantFail(LLJ->defineAbsolute(
        mangle("tanhf"), {llvm::pointerToJITTargetAddress(&tanhf), {}}));
    cantFail(LLJ->defineAbsolute(
        mangle("sqrtf"), {llvm::pointerToJITTargetAddress(&sqrtf), {}}));
    cantFail(LLJ->defineAbsolute(
        mangle("fabsf"), {llvm::pointerToJITTargetAddress(&fabsf), {}}));
    cantFail(LLJ->defineAbsolute(
        mangle("floorf"), {llvm::pointerToJITTargetAddress(&floorf), {}}));
    cantFail(LLJ->defineAbsolute(
        mangle("ceilf"), {llvm::pointerToJITTargetAddress(&ceilf), {}}));
    cantFail(LLJ->defineAbsolute(
        mangle("roundf"), {llvm::pointerToJITTargetAddress(&roundf), {}}));
    cantFail(LLJ->defineAbsolute(
        mangle("truncf"), {llvm::pointerToJITTargetAddress(&truncf), {}}));
    cantFail(LLJ->defineAbsolute(
        mangle("atan2f"), {llvm::pointerToJITTargetAddress(&atan2f), {}}));
    cantFail(LLJ->defineAbsolute(
        mangle("fmodf"), {llvm::pointerToJITTargetAddress(&fmodf), {}}));
    cantFail(LLJ->defineAbsolute(
        mangle("remainderf"),
        {llvm::pointerToJITTargetAddress(&remainderf), {}}));
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

  StringRef mangle(std::string S) {
    return *Mangle(S);
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

StringRef PytorchLLVMJIT::mangle(std::string S) {
  return impl_->mangle(S);
}

const DataLayout& PytorchLLVMJIT::getDataLayout() {
  return impl_->getDataLayout();
}

} // end namespace orc
} // end namespace llvm

#endif // ENABLE_LLVM
