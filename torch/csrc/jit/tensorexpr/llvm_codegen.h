#pragma once

#define ENABLE_LLVM 1
#ifdef ENABLE_LLVM
#include <torch/csrc/WindowsTorchApiMacro.h>

#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "torch/csrc/jit/tensorexpr/ir.h"
#include "torch/csrc/jit/tensorexpr/ir_visitor.h"
#include "torch/csrc/jit/tensorexpr/llvm_jit.h"

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Verifier.h>
#include <unordered_map>
#include <vector>

#define DEBUG_PRINT 1

#if DEBUG_PRINT
#include <llvm/IR/LegacyPassManager.h>
#endif

namespace torch {
namespace jit {
namespace compiler {

class TORCH_API LLVMCodeGen : public IRVisitor {
 private:
  llvm::orc::ThreadSafeContext context_;
  llvm::IRBuilder<> irb_;
  std::unique_ptr<llvm::TargetMachine> TM;
  std::unique_ptr<llvm::orc::PytorchLLVMJIT> jit_;
  std::unique_ptr<llvm::Module> module_;
  llvm::Function* fn_;
  llvm::BasicBlock* bb_;
  llvm::Value* value_;

  llvm::Type* int32Ty_;
  llvm::Type* floatTy_;

  std::unordered_map<const BaseExprNode*, int> varToArg_;
  std::unordered_map<const Variable*, llvm::Value*> varToVal_;

 public:
  explicit LLVMCodeGen(const std::vector<Buffer*>& args, Dtype dtype = kInt32);
  LLVMCodeGen();

  void visit(const Add* v) override;
  void visit(const Sub* v) override;
  void visit(const Mul* v) override;
  void visit(const Div* v) override;
  void visit(const Max* v) override;
  void visit(const Min* v) override;
  void visit(const CompareSelect* v) override;
  void visit(const IntImm* v) override;
  void visit(const FloatImm* v) override;
  void visit(const Cast* v) override;
  void visit(const Variable* v) override;
  void visit(const Let* v) override;
  void visit(const Ramp* v) override;
  void visit(const Load* v) override;
  void visit(const For* v) override;
  void visit(const Block* v) override;
  void visit(const Store* v) override;
  void visit(const Broadcast* v) override;

  llvm::Value* emitMaskedLoad(
      llvm::Value* addr,
      llvm::Value* idx,
      llvm::Value* mask);
  void emitMaskedStore(
      llvm::Value* base,
      llvm::Value* idx,
      llvm::Value* mask,
      llvm::Value* val);

  void optimize(llvm::Module& M);

  template <typename T>
  T value() {
    std::vector<void*> args;
    return value<T>(args);
  }

  template <typename T>
  llvm::Value* getConstant(T t);

  template <typename T>
  T value(std::vector<void*>& args) {
    if (!value_) {
      value_ = getConstant(static_cast<T>(0));
    }
    irb_.CreateRet(value_);
#if DEBUG_PRINT
    llvm::errs() << *module_;
#endif
    CHECK(!llvm::verifyFunction(*fn_, &llvm::outs()))
        << "Function verification failed";
    optimize(*module_);

#if DEBUG_PRINT
    llvm::errs() << *module_;
    llvm::SmallVector<char, 0> asmBuffer;
    llvm::raw_svector_ostream asmStream(asmBuffer);
    llvm::legacy::PassManager PM;
    TM->addPassesToEmitFile(
        PM,
        asmStream,
        nullptr,
        llvm::TargetMachine::CodeGenFileType::CGFT_AssemblyFile);
    PM.run(*module_);
    llvm::errs() << asmStream.str();
#endif

    cantFail(jit_->addModule(
        llvm::orc::ThreadSafeModule(std::move(module_), context_)));
    auto sym = jit_->findSymbol("wrapper");
    auto addr = sym.getAddress();
    assert(addr);
    T (*fp)(void**) = (T(*)(void**))addr.get();
    T rv = fp(args.data());
    return rv;
  }
};

  template<>
  inline llvm::Value* LLVMCodeGen::getConstant<float>(float t) {
    return llvm::ConstantFP::get(floatTy_, t);
  }

  template<>
  inline llvm::Value* LLVMCodeGen::getConstant<int32_t>(int32_t t) {
    return llvm::ConstantInt::getSigned(int32Ty_, t);
  }


} // namespace compiler
} // namespace jit
} // namespace torch

#endif // ENABLE_LLVM
