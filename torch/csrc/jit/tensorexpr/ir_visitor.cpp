#include "torch/csrc/jit/tensorexpr/ir_visitor.h"

#include "torch/csrc/jit/tensorexpr/ir.h"
#include "torch/csrc/jit/tensorexpr/tensor.h"

namespace torch {
namespace jit {
namespace tensorexpr {

template <typename Op>
static void traverse_binary_op(const BinaryOpNode<Op>* v, IRVisitor* visitor) {
  v->lhs().accept(visitor);
  v->rhs().accept(visitor);
}

void IRVisitor::traverse(const Add* v) {
  traverse_binary_op(v, this);
}

void IRVisitor::traverse(const Sub* v) {
  traverse_binary_op(v, this);
}

void IRVisitor::traverse(const Mul* v) {
  traverse_binary_op(v, this);
}

void IRVisitor::traverse(const Div* v) {
  traverse_binary_op(v, this);
}

void IRVisitor::traverse(const Mod* v) {
  traverse_binary_op(v, this);
}

void IRVisitor::traverse(const Max* v) {
  traverse_binary_op(v, this);
}

void IRVisitor::traverse(const Min* v) {
  traverse_binary_op(v, this);
}

void IRVisitor::traverse(const CompareSelect* v) {
  traverse_binary_op(v, this);
}

void IRVisitor::traverse(const IntImm* v) {}
void IRVisitor::traverse(const FloatImm* v) {}
void IRVisitor::traverse(const Cast* v) {
  v->src_value().accept(this);
}
void IRVisitor::traverse(const Variable* v) {}
void IRVisitor::traverse(const Let* v) {
  v->var().accept(this);
  v->value().accept(this);
  v->body().accept(this);
}

void IRVisitor::traverse(const Ramp* v) {
  v->base().accept(this);
  v->stride().accept(this);
}

void IRVisitor::traverse(const Load* v) {
  v->base_handle().accept(this);
  v->index().accept(this);
  v->mask().accept(this);
}

void IRVisitor::traverse(const Store* v) {
  v->base_handle().accept(this);
  v->index().accept(this);
  v->value().accept(this);
  v->mask().accept(this);
}

void IRVisitor::traverse(const Block* v) {
  for (int i = 0; i < v->nstmts(); i++) {
    v->stmt(i).accept(this);
  }
}

void IRVisitor::traverse(const For* v) {
  v->var().accept(this);
  v->start().accept(this);
  v->stop().accept(this);
  v->body().accept(this);
}

void IRVisitor::traverse(const Broadcast* v) {
  v->value().accept(this);
}

void IRVisitor::traverse(const IfThenElse* v) {
  v->condition().accept(this);
  v->true_value().accept(this);
  v->false_value().accept(this);
}

void IRVisitor::traverse(const BaseCallNode* v) {
  for (int i = 0; i < v->nparams(); i++) {
    v->param(i).accept(this);
  }
}

void IRVisitor::traverse(const Intrinsics* v) {
  const BaseCallNode* base = v;
  this->traverse(base);
}

void IRVisitor::traverse(const FunctionCall* v) {
  const BaseCallNode* base = v;
  this->traverse(base);
}

void IRVisitor::traverse(const Allocate* v) {
  Var buffer_var = v->buffer_var();
  buffer_var.accept(this);
  std::vector<Expr> dims = v->dims();
  for (Expr& dim : dims) {
    dim.accept(this);
  }
}

void IRVisitor::traverse(const Free* v) {
  Var buffer_var = v->buffer_var();
  buffer_var.accept(this);
}

void IRVisitor::traverse(const Cond* v) {
  Expr condition = v->condition();
  Stmt true_stmt = v->true_stmt();
  Stmt false_stmt = v->false_stmt();
  condition.accept(this);
  true_stmt.accept(this);
  false_stmt.accept(this);
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
