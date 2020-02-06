#pragma once

#include "torch/csrc/jit/tensorexpr/ir_mutator.h"
#include "torch/csrc/jit/tensorexpr/ir_visitor.h"
#include "torch/csrc/jit/tensorexpr/types.h"

namespace torch {
namespace jit {
namespace tensorexpr {

class KernelScopedObject;
// An arena that manages all the underlying kernel-scoped objects.
class KernelArena {
 public:
  static KernelArena& GetCurrentKernelArena();
  TORCH_API KernelArena() {}
  TORCH_API ~KernelArena();

 private:
  KernelArena(const KernelArena&) = delete;
  KernelArena& operator=(const KernelArena&) = delete;
  friend class KernelScopedObject;
  std::vector<KernelScopedObject*> kernel_objects_; // owned
};

// A RAII convenience wrapper on top of a kernel.
// It either creates a Kernel, or take another existing Kernel, and sets it as
// the current Kernel, as long as this KernelScope object is alive.
class KernelScope {
 public:
  TORCH_API KernelScope();
  TORCH_API explicit KernelScope(KernelArena& kernel_arena);
  TORCH_API ~KernelScope() noexcept(false);

 private:
  KernelScope(const KernelScope&) = delete;
  KernelScope& operator=(const KernelScope&) = delete;
  bool owning_kernel_arena_ = false;
  KernelArena* kernel_arena_ =
      nullptr; // possibly owned, if owning_kernel_arena_ == true
};

// The base object managed by the Kernel.
// The object must be created through "new", and when the Kernel is destroyed,
// All its registered objects are destroyed through "delete".
class TORCH_API KernelScopedObject {
 public:
  TORCH_API KernelScopedObject();
  TORCH_API virtual ~KernelScopedObject();

 private:
  KernelScopedObject(const KernelScopedObject&) = delete;
  KernelScopedObject& operator=(const KernelScopedObject&) = delete;
};

// The commomn class between all IR nodes.
class IRNode : public KernelScopedObject {
 public:
  TORCH_API virtual void accept(IRVisitor* visitor) const = 0;
  TORCH_API virtual ~IRNode() {}
};

// The common base between all expression node.
class Expr;
class BaseExprNode : public IRNode {
 public:
  explicit BaseExprNode(Dtype dtype) : dtype_(dtype) {}
  Dtype dtype() const {
    return dtype_;
  }
  virtual Expr accept_mutator(IRMutator* mutator) = 0;

 private:
  Dtype dtype_;
};

// The common base between all statement node.
class BaseStmtNode : public IRNode {
 public:
  BaseStmtNode() {}
  virtual Stmt accept_mutator(IRMutator* mutator) = 0;
};

// A CRTP pattern to accept visitors for children class,
// and dispatch back to the children.
template <class Op, class Base = BaseExprNode>
class ExprNode : public Base {
 public:
  using ExprNodeBase = ExprNode<Op>;
  void accept(IRVisitor* visitor) const override {
    visitor->visit(static_cast<const Op*>(this));
  }
  Expr accept_mutator(IRMutator* mutator) override;
  // pass the constructor to the base class
  using Base::Base;
};

template <class Op>
class StmtNode : public BaseStmtNode {
 public:
  using StmtNodeBase = StmtNode<Op>;
  void accept(IRVisitor* visitor) const override {
    visitor->visit(static_cast<const Op*>(this));
  }
  Stmt accept_mutator(IRMutator* mutator) override;
  StmtNode() {}
};

// A wrapper object to the underlying ExprNode.
// Also serves the primary way to build and operate on other expressions.
class TORCH_API Expr {
 public:
  Expr() {}
  explicit Expr(const BaseExprNode* node)
      : base_expr_node_(const_cast<BaseExprNode*>(node)) {}

  BaseExprNode* node() {
    return base_expr_node_;
  }

  const BaseExprNode* node() const {
    return base_expr_node_;
  }

  bool empty() const {
    return base_expr_node_ == nullptr;
  }

  void accept(IRVisitor* visitor) const {
    // TODO: Consider implement this without using recursion. Otherwise,
    // if the expression tree is degenerate and too long, it could cause a
    // stack overflow.
    if (node() == nullptr) {
      return;
    }
    node()->accept(visitor);
  }

  Expr accept_mutator(IRMutator* mutator) {
    if (node() == nullptr) {
      return Expr();
    }
    return node()->accept_mutator(mutator);
  }

  Expr(int v);
  Expr(float v);

  template <class Op>
  Op* AsNode() {
    return dynamic_cast<Op*>(this->node());
  }

  template <class Op>
  const Op* AsNode() const {
    return const_cast<Expr*>(this)->AsNode<Op>();
  }

  Dtype dtype() const {
    return node()->dtype();
  }

  // Handling the math operators.
  Expr operator+(const Expr& other) const;
  Expr operator-(const Expr& other) const;
  Expr operator*(const Expr& other) const;
  Expr operator/(const Expr& other) const;
  Expr operator==(const Expr& other) const;
  Expr operator!=(const Expr& other) const;
  Expr operator>(const Expr& other) const;
  Expr operator>=(const Expr& other) const;
  Expr operator<(const Expr& other) const;
  Expr operator<=(const Expr& other) const;

 private:
  BaseExprNode* base_expr_node_ = nullptr;
};

class Stmt {
 public:
  Stmt() {}
  explicit Stmt(const BaseStmtNode* node)
      : base_stmt_node_(const_cast<BaseStmtNode*>(node)) {}

  BaseStmtNode* node() {
    return base_stmt_node_;
  }

  const BaseStmtNode* node() const {
    return base_stmt_node_;
  }

  void accept(IRVisitor* visitor) const {
    if (node() == nullptr) {
      return;
    }
    node()->accept(visitor);
  }

  Stmt accept_mutator(IRMutator* mutator) {
    if (node() == nullptr) {
      return Stmt();
    }
    return node()->accept_mutator(mutator);
  }

  bool empty() const {
    return node() == nullptr;
  }

  template <class Op>
  const Op* AsNode() const {
    return dynamic_cast<const Op*>(this->node());
  }

 private:
  BaseStmtNode* base_stmt_node_ = nullptr;
};

template <class Op, class Base>
Expr ExprNode<Op, Base>::accept_mutator(IRMutator* mutator) {
  ExprNode* this_mutable = const_cast<ExprNode*>(this);
  return mutator->mutate(static_cast<Op*>(this_mutable));
}

template <class Op>
Stmt StmtNode<Op>::accept_mutator(IRMutator* mutator) {
  StmtNode* this_mutable = const_cast<StmtNode*>(this);
  return mutator->mutate(static_cast<Op*>(this_mutable));
}

inline bool same_node(const Expr& expr1, const Expr& expr2) {
  return expr1.AsNode<BaseExprNode>() == expr2.AsNode<BaseExprNode>();
}

inline bool same_node(const Stmt& stmt1, const Stmt& stmt2) {
  return stmt1.AsNode<BaseStmtNode>() == stmt2.AsNode<BaseStmtNode>();
}

TORCH_API Expr sin(const Expr& v);
TORCH_API Expr cos(const Expr& v);
TORCH_API Expr tan(const Expr& v);
TORCH_API Expr asin(const Expr& v);
TORCH_API Expr acos(const Expr& v);
TORCH_API Expr atan(const Expr& v);
TORCH_API Expr sinh(const Expr& v);
TORCH_API Expr cosh(const Expr& v);
TORCH_API Expr tanh(const Expr& v);
TORCH_API Expr exp(const Expr& v);
TORCH_API Expr fabs(const Expr& v);
TORCH_API Expr log(const Expr& v);
TORCH_API Expr log2(const Expr& v);
TORCH_API Expr log10(const Expr& v);
TORCH_API Expr erf(const Expr& v);
TORCH_API Expr sqrt(const Expr& v);
TORCH_API Expr rsqrt(const Expr& v);
TORCH_API Expr ceil(const Expr& v);
TORCH_API Expr floor(const Expr& v);
TORCH_API Expr round(const Expr& v);
TORCH_API Expr trunc(const Expr& v);
TORCH_API Expr pow(const Expr& v1, const Expr& v2);
TORCH_API Expr fmod(const Expr& v1, const Expr& v2);
TORCH_API Expr remainder(const Expr& v1, const Expr& v2);

TORCH_API Expr ifThenElse(const Expr& c, const Expr& t, const Expr& f);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
