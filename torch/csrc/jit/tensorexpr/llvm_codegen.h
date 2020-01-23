#pragma once

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

#define DEBUG_PRINT 0

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
  llvm::JITTargetAddress kernelAddress_;

  llvm::Type* int32Ty_;
  llvm::Type* floatTy_;

  std::unordered_map<const BaseExprNode*, int> varToArg_;
  std::unordered_map<const Variable*, llvm::Value*> varToVal_;

 private:
  explicit LLVMCodeGen(
      const IRNode* node,
      const std::vector<Buffer*>& args,
      Dtype dtype = kInt32);

 public:
  explicit LLVMCodeGen(
      const Stmt& stmt,
      const std::vector<Buffer*>& args,
      Dtype dtype = kInt32);
  explicit LLVMCodeGen(const Stmt& stmt);
  explicit LLVMCodeGen(
      const Expr& expr,
      const std::vector<Buffer*>& args,
      Dtype dtype = kInt32);
  explicit LLVMCodeGen(const Expr& expr);

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
  virtual void visit(const BaseCallNode* v);
  virtual void visit(const Intrinsics* v);
  virtual void visit(const FunctionCall* v);
  virtual void visit(const Allocate* v);
  virtual void visit(const Free* v);

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
  T value(std::vector<void*>& args) {
    T (*fp)(void**) = (T(*)(void**))kernelAddress_;
    T rv = fp(args.data());
    return rv;
  }
};

} // namespace compiler
} // namespace jit
} // namespace torch

#endif // ENABLE_LLVM
