#pragma once

#include "torch/csrc/jit/tensorexpr/ir_visitor.h"

#include <asmjit/asmjit.h>
#include <memory>

namespace torch {
namespace jit {
namespace tensorexpr {

class TORCH_API ASMJITCodeGen : public IRVisitor {
 private:
  std::unique_ptr<asmjit::JitRuntime> jit_;
  std::unique_ptr<asmjit::CodeHolder> code_;
  std::unique_ptr<asmjit::x86::Compiler> cc_;
  asmjit::x86::Reg value_;

 public:
  ASMJITCodeGen();
  void postorder_visit(const Add* v) override;
  void postorder_visit(const Sub* v) override;
  void postorder_visit(const Mul* v) override;
  void postorder_visit(const Div* v) override;
  void postorder_visit(const IntImm* v) override;
  void postorder_visit(const FloatImm* v) override;
  int value();
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
