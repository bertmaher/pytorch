#pragma once

#ifdef ENABLE_LLVM
#include <torch/csrc/WindowsTorchApiMacro.h>

#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "torch/csrc/jit/tensorexpr/codegen.h"
#include "torch/csrc/jit/tensorexpr/ir.h"
#include "torch/csrc/jit/tensorexpr/ir_visitor.h"
#include "torch/csrc/jit/tensorexpr/llvm_jit.h"

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Verifier.h>
#include <unordered_map>
#include <vector>

#define DEBUG_PRINT 0

#if DEBUG_PRINT
#include <llvm/IR/LegacyPassManager.h>
#endif

namespace torch {
namespace jit {
namespace tensorexpr {

class TORCH_API LLVMCodeGen : public CodeGen, public IRVisitor {
 private:
  llvm::orc::ThreadSafeContext context_;
  llvm::IRBuilder<> irb_;
  std::unique_ptr<llvm::TargetMachine> TM_;
  std::unique_ptr<llvm::orc::PytorchLLVMJIT> jit_;
  std::unique_ptr<llvm::Module> module_;
  llvm::Function* fn_;
  llvm::BasicBlock* bb_;
  llvm::Value* value_;
  llvm::JITTargetAddress kernelAddress_;

  llvm::Type* int32Ty_;
  llvm::Type* floatTy_;

  std::unordered_map<const BaseExprNode*, int> varToArg_;
  std::unordered_map<const Variable*, llvm::Value*> varToVal_;

  std::vector<void*> args_;

 private:
  explicit LLVMCodeGen(
      const IRNode* node,
      const std::vector<BufferArg>& args,
      Dtype dtype = kInt32);

  llvm::LLVMContext& getContext();
  llvm::Type* dtypeToLLVM(Dtype dtype);
  llvm::Type* dtypeToLLVMPtr(Dtype dtype);
  void emitWrapper(const std::vector<llvm::Type*>& params);
  void emitKernel(const IRNode* node, const std::vector<llvm::Type*>& params);

 public:
  explicit LLVMCodeGen(
      const Stmt& stmt,
      const std::vector<BufferArg>& args,
      Dtype dtype = kInt32);
  explicit LLVMCodeGen(const Stmt& stmt);
  explicit LLVMCodeGen(
      const Expr& expr,
      const std::vector<BufferArg>& args,
      Dtype dtype = kInt32);
  explicit LLVMCodeGen(const Expr& expr);

  ~LLVMCodeGen() override {}

  TORCH_API void call(const std::vector<CallArg>& args) override;

  void postorder_visit(const Add* v) override;
  void postorder_visit(const Sub* v) override;
  void postorder_visit(const Mul* v) override;
  void postorder_visit(const Div* v) override;
  void postorder_visit(const Mod* v) override;
  void postorder_visit(const Max* v) override;
  void postorder_visit(const Min* v) override;
  void postorder_visit(const CompareSelect* v) override;
  void postorder_visit(const IntImm* v) override;
  void postorder_visit(const FloatImm* v) override;
  void postorder_visit(const Cast* v) override;
  void postorder_visit(const Variable* v) override;
  void postorder_visit(const Let* v) override;
  void postorder_visit(const Ramp* v) override;
  void postorder_visit(const Load* v) override;
  void postorder_visit(const For* v) override;
  void postorder_visit(const Block* v) override;
  void postorder_visit(const Store* v) override;
  void postorder_visit(const Broadcast* v) override;
  void postorder_visit(const IfThenElse* v) override;
  void postorder_visit(const BaseCallNode* v) override;
  void postorder_visit(const Intrinsics* v) override;
  void postorder_visit(const FunctionCall* v) override;
  void postorder_visit(const Allocate* v) override;
  void postorder_visit(const Free* v) override;
  void postorder_visit(const Cond* v) override;

  llvm::Value* emitUnmaskedLoad(llvm::Value* addr, llvm::Value* idx);
  llvm::Value* emitMaskedLoad(
      llvm::Value* addr,
      llvm::Value* idx,
      llvm::Value* mask);
  void emitUnmaskedStore(llvm::Value* base, llvm::Value* idx, llvm::Value* val);
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
  T value(std::vector<void*>& args) {
    T (*fp)(void**) = (T(*)(void**))kernelAddress_;
    T rv = fp(args.data());
    return rv;
  }
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch

#endif // ENABLE_LLVM
