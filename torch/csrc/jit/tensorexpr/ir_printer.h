#pragma once

#include <iostream>

#include "torch/csrc/jit/tensorexpr/ir.h"
#include "torch/csrc/jit/tensorexpr/ir_visitor.h"
#include "torch/csrc/jit/tensorexpr/unique_name_manager.h"

namespace torch {
namespace jit {
namespace tensorexpr {

class TORCH_API IRPrinter : public IRVisitor {
 public:
  explicit IRPrinter(std::ostream& os) : printer_os_(this, os) {}

  void print(Expr);
  void print(Stmt);
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
  void postorder_visit(const Allocate* v) override;
  void postorder_visit(const Free* v) override;
  void postorder_visit(const Cond* v) override;

  std::ostream& os() {
    return printer_os_;
  }

  class PrinterStream : public std::ostream {
   public:
    PrinterStream(IRPrinter* printer, std::ostream& os)
        : std::ostream(os.rdbuf()), printer_(printer) {}

    IRPrinter* printer() {
      return printer_;
    }

   private:
    IRPrinter* printer_ = nullptr;
  };

 protected:
  UniqueNameManager* name_manager() {
    return &name_manager_;
  }

 private:
  std::ostream& raw_os() {
    return printer_os_;
  }

  PrinterStream printer_os_;
  UniqueNameManager name_manager_;
};

TORCH_API std::ostream& operator<<(std::ostream& stream, const Expr&);
TORCH_API std::ostream& operator<<(std::ostream& stream, const Stmt&);

} // namespace tensorexpr
} // namespace jit
} // namespace torch

namespace std {

using torch::jit::tensorexpr::Expr;
using torch::jit::tensorexpr::Stmt;

inline std::string to_string(const Expr& expr) {
  std::ostringstream oss;
  oss << expr;
  return oss.str();
}

inline std::string to_string(const Stmt& stmt) {
  std::ostringstream oss;
  oss << stmt;
  return oss.str();
}

}; // namespace std
