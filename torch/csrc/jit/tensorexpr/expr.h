/**
 * This file implements the core classes for Tensor Expressions.
 *
 * The structure of the expressions is inspired by Halide/TVM IR.
 */
#pragma once

#include "torch/csrc/jit/tensorexpr/ir_mutator.h"
#include "torch/csrc/jit/tensorexpr/ir_visitor.h"
#include "torch/csrc/jit/tensorexpr/types.h"
#include "torch/csrc/jit/tensorexpr/mem_arena.h"

namespace torch {
namespace jit {
namespace tensorexpr {

// The common base between all expression node.
class ExprHandler;
class Expr : public KernelScopedObject {
 public:
  explicit Expr(Dtype dtype) : dtype_(dtype) {}
  Dtype dtype() const {
    return dtype_;
  }
  TORCH_API virtual void accept(IRVisitor* visitor) const = 0;
  virtual const Expr* accept_mutator(IRMutator* mutator) const = 0;

 private:
  Dtype dtype_;
};

// The common base between all statement node.
class Stmt : public KernelScopedObject {
 public:
  Stmt() {}
  TORCH_API virtual void accept(IRVisitor* visitor) const = 0;
  virtual Stmt* accept_mutator(IRMutator* mutator) = 0;
};

// A CRTP pattern to accept visitors for children class,
// and dispatch back to the children.
template <class Op, class Base = Expr>
class ExprNode : public Base {
 public:
  using ExprNodeBase = ExprNode<Op>;
  void accept(IRVisitor* visitor) const override {
    visitor->visit(static_cast<const Op*>(this));
  }
  const Expr* accept_mutator(IRMutator* mutator) const override;
  // pass the constructor to the base class
  using Base::Base;
};

template <class Op>
class StmtNode : public Stmt {
 public:
  using StmtNodeBase = StmtNode<Op>;
  void accept(IRVisitor* visitor) const override {
    visitor->visit(static_cast<const Op*>(this));
  }
  Stmt* accept_mutator(IRMutator* mutator) override;
  StmtNode() {}
};

// A wrapper object to the underlying ExprNode.
// Also serves the primary way to build and operate on other expressions.
class TORCH_API ExprHandler {
 public:
  ExprHandler() {}
  explicit ExprHandler(const Expr* node)
      : base_expr_node_(const_cast<Expr*>(node)) {}

  Expr* node() {
    return base_expr_node_;
  }

  const Expr* node() const {
    return base_expr_node_;
  }

  bool empty() const {
    return base_expr_node_ == nullptr;
  }

  ExprHandler(int v);
  ExprHandler(float v);

  template <class Op>
  Op* AsNode() {
    return dynamic_cast<Op*>(this->node());
  }

  template <class Op>
  const Op* AsNode() const {
    return const_cast<ExprHandler*>(this)->AsNode<Op>();
  }

  Dtype dtype() const {
    return node()->dtype();
  }

  // Handling the math operators.
  ExprHandler operator+(const ExprHandler& other) const;
  ExprHandler operator-(const ExprHandler& other) const;
  ExprHandler operator*(const ExprHandler& other) const;
  ExprHandler operator/(const ExprHandler& other) const;
  ExprHandler operator==(const ExprHandler& other) const;
  ExprHandler operator!=(const ExprHandler& other) const;
  ExprHandler operator>(const ExprHandler& other) const;
  ExprHandler operator>=(const ExprHandler& other) const;
  ExprHandler operator<(const ExprHandler& other) const;
  ExprHandler operator<=(const ExprHandler& other) const;

 private:
  Expr* base_expr_node_ = nullptr;
};

template <class Op, class Base>
const Expr* ExprNode<Op, Base>::accept_mutator(IRMutator* mutator) const {
  ExprNode* this_mutable = const_cast<ExprNode*>(this);
  return mutator->mutate(static_cast<Op*>(this_mutable));
}

template <class Op>
Stmt* StmtNode<Op>::accept_mutator(IRMutator* mutator) {
  StmtNode* this_mutable = const_cast<StmtNode*>(this);
  return mutator->mutate(static_cast<Op*>(this_mutable));
}

inline bool same_node(const ExprHandler& expr1, const ExprHandler& expr2) {
  return expr1.AsNode<Expr>() == expr2.AsNode<Expr>();
}

inline bool same_node(Stmt* stmt1, Stmt* stmt2) {
  return stmt1 == stmt2;
}

TORCH_API ExprHandler sin(const ExprHandler& v);
TORCH_API ExprHandler cos(const ExprHandler& v);
TORCH_API ExprHandler tan(const ExprHandler& v);
TORCH_API ExprHandler asin(const ExprHandler& v);
TORCH_API ExprHandler acos(const ExprHandler& v);
TORCH_API ExprHandler atan(const ExprHandler& v);
TORCH_API ExprHandler sinh(const ExprHandler& v);
TORCH_API ExprHandler cosh(const ExprHandler& v);
TORCH_API ExprHandler tanh(const ExprHandler& v);
TORCH_API ExprHandler exp(const ExprHandler& v);
TORCH_API ExprHandler expm1(const ExprHandler& v);
TORCH_API ExprHandler fabs(const ExprHandler& v);
TORCH_API ExprHandler log(const ExprHandler& v);
TORCH_API ExprHandler log2(const ExprHandler& v);
TORCH_API ExprHandler log10(const ExprHandler& v);
TORCH_API ExprHandler log1p(const ExprHandler& v);
TORCH_API ExprHandler erf(const ExprHandler& v);
TORCH_API ExprHandler erfc(const ExprHandler& v);
TORCH_API ExprHandler sqrt(const ExprHandler& v);
TORCH_API ExprHandler rsqrt(const ExprHandler& v);
TORCH_API ExprHandler ceil(const ExprHandler& v);
TORCH_API ExprHandler floor(const ExprHandler& v);
TORCH_API ExprHandler round(const ExprHandler& v);
TORCH_API ExprHandler trunc(const ExprHandler& v);
TORCH_API ExprHandler frac(const ExprHandler& v);
TORCH_API ExprHandler lgamma(const ExprHandler& v);
TORCH_API ExprHandler atan2(const ExprHandler& v1, const ExprHandler& v2);
TORCH_API ExprHandler pow(const ExprHandler& v1, const ExprHandler& v2);
TORCH_API ExprHandler fmod(const ExprHandler& v1, const ExprHandler& v2);
TORCH_API ExprHandler remainder(const ExprHandler& v1, const ExprHandler& v2);

TORCH_API ExprHandler ifThenElse(const ExprHandler& c, const ExprHandler& t, const ExprHandler& f);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
