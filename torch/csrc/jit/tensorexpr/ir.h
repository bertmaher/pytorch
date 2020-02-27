#pragma once

#include <string>
#include <vector>

#include "torch/csrc/jit/tensorexpr/expr.h"
#include "torch/csrc/jit/tensorexpr/stmt.h"

namespace torch {
namespace jit {
namespace tensorexpr {

enum IRNodeType {
  kAdd,
  kSub,
  kMul,
  kDiv,
  kMod,
  kMax,
  kMin,
  kAnd,
  kLshift,
  kRshift,
  kXor,
  kCompareSelect,
};

enum CompareSelectOperation {
  kEQ,
  kGT,
  kGE,
  kLT,
  kLE,
  kNE,
};

class Buffer;

class Cast : public ExprNode<Cast> {
 public:
  const Expr* src_value() const {
    return src_value_;
  }
  static ExprHandle make(Dtype dtype, const ExprHandle& src_value) {
    return ExprHandle(new Cast(dtype, src_value.node()));
  }
  Cast(Dtype dtype, const Expr* src_value)
      : ExprNodeBase(dtype), src_value_(src_value) {}

 private:
  const Expr* src_value_;
};

template <typename T>
ExprHandle cast(const ExprHandle& src_value) {
  return Cast::make(Dtype(ToDtype<T>(), src_value.dtype().lanes()), src_value);
}

// Represent the expression node for binary operators.
// A CRTP pattern to share common code among the operators.
template <typename Op>
class BinaryOpNode : public ExprNode<Op> {
 public:
  const Expr* lhs() const {
    return this->lhs_;
  }
  const Expr* rhs() const {
    return this->rhs_;
  }
  IRNodeType expr_type() const {
    return expr_type_;
  }

  static ExprHandle make(const ExprHandle& lhs, const ExprHandle& rhs) {
    return ExprHandle(new Op(lhs.node(), rhs.node()));
  }

  BinaryOpNode(
      const Expr* lhs_v,
      const Expr* rhs_v,
      IRNodeType expr_type,
      ReturnType ret_type = ReturnType::knone)
      : ExprNode<Op>(BinaryOpDtype(lhs_v->dtype(), rhs_v->dtype(), ret_type)),
        lhs_(CastIfNeeded(lhs_v, ExprNode<Op>::dtype())),
        rhs_(CastIfNeeded(rhs_v, ExprNode<Op>::dtype())),
        expr_type_(expr_type) {}

 private:
  static const Expr* CastIfNeeded(const Expr* expr, Dtype dst_dtype) {
    if (expr->dtype() == dst_dtype) {
      return expr;
    }
    return Cast::make(dst_dtype, ExprHandle(expr)).node();
  }

  const Expr* lhs_;
  const Expr* rhs_;
  IRNodeType expr_type_;
};

class Add : public BinaryOpNode<Add> {
 public:
  Add(const Expr* lhs, const Expr* rhs)
      : BinaryOpNode(lhs, rhs, IRNodeType::kAdd) {}
};

class Sub : public BinaryOpNode<Sub> {
 public:
  Sub(const Expr* lhs, const Expr* rhs)
      : BinaryOpNode(lhs, rhs, IRNodeType::kSub) {}
};

class Mul : public BinaryOpNode<Mul> {
 public:
  Mul(const Expr* lhs, const Expr* rhs)
      : BinaryOpNode(lhs, rhs, IRNodeType::kMul) {}
};

class Div : public BinaryOpNode<Div> {
 public:
  Div(const Expr* lhs, const Expr* rhs)
      : BinaryOpNode(lhs, rhs, IRNodeType::kDiv) {}
};

class Mod : public BinaryOpNode<Mod> {
 public:
  Mod(const Expr* lhs, const Expr* rhs)
      : BinaryOpNode(lhs, rhs, IRNodeType::kMod) {}
};

class And : public BinaryOpNode<And> {
 public:
  And(const Expr* lhs, const Expr* rhs)
      : BinaryOpNode(lhs, rhs, IRNodeType::kAnd) {
    CHECK_EQ(lhs->dtype().scalar_type(), kInt32);
    CHECK_EQ(lhs->dtype(), rhs->dtype());
  }
};

class Xor : public BinaryOpNode<Xor> {
 public:
  Xor(const Expr* lhs, const Expr* rhs)
      : BinaryOpNode(lhs, rhs, IRNodeType::kXor) {
    CHECK_EQ(lhs->dtype().scalar_type(), kInt32);
    CHECK_EQ(lhs->dtype(), rhs->dtype());
  }
};

class Lshift : public BinaryOpNode<Lshift> {
 public:
  Lshift(const Expr* lhs, const Expr* rhs)
      : BinaryOpNode(lhs, rhs, IRNodeType::kLshift) {
    CHECK_EQ(lhs->dtype().scalar_type(), kInt32);
    CHECK_EQ(lhs->dtype(), rhs->dtype());
  }
};

class Rshift : public BinaryOpNode<Rshift> {
 public:
  Rshift(const Expr* lhs, const Expr* rhs)
      : BinaryOpNode(lhs, rhs, IRNodeType::kRshift) {
    CHECK_EQ(lhs->dtype().scalar_type(), kInt32);
    CHECK_EQ(lhs->dtype(), rhs->dtype());
  }
};

class Max : public BinaryOpNode<Max> {
 private:
  bool propagate_nans_;

 public:
  Max(const Expr* lhs, const Expr* rhs, bool propagate_nans)
      : BinaryOpNode(lhs, rhs, IRNodeType::kMax),
        propagate_nans_(propagate_nans) {}

  bool propagate_nans() const {
    return propagate_nans_;
  }

  static ExprHandle make(const ExprHandle& lhs, const ExprHandle& rhs) = delete;
  static ExprHandle make(const ExprHandle& lhs, const ExprHandle& rhs, bool propagate_nans) {
    return ExprHandle(new Max(lhs.node(), rhs.node(), propagate_nans));
  }
};

class Min : public BinaryOpNode<Min> {
 private:
  bool propagate_nans_;

 public:
  Min(const Expr* lhs, const Expr* rhs, bool propagate_nans)
      : BinaryOpNode(lhs, rhs, IRNodeType::kMin),
        propagate_nans_(propagate_nans) {}

  bool propagate_nans() const {
    return propagate_nans_;
  }

  static ExprHandle make(const ExprHandle& lhs, const ExprHandle& rhs) = delete;
  static ExprHandle make(const ExprHandle& lhs, const ExprHandle& rhs, bool propagate_nans) {
    return ExprHandle(new Min(lhs.node(), rhs.node(), propagate_nans));
  }
};

// Encode an integer immediate value.
class IntImm : public ExprNode<IntImm> {
 public:
  int value() const {
    return value_;
  }
  static ExprHandle make(int value) {
    return ExprHandle(new IntImm(value));
  }
  IntImm(int value) : ExprNodeBase(kInt32), value_(value) {}

 private:
  int value_;
};

// Encode an fp32 immediate value.
class FloatImm : public ExprNode<FloatImm> {
 public:
  float value() const {
    return value_;
  }
  static ExprHandle make(float value) {
    return ExprHandle(new FloatImm(value));
  }

 private:
  FloatImm(float value) : ExprNodeBase(kFloat32), value_(value) {}
  float value_;
};

// Bind the value to the var and evaluate the body.
class Let : public ExprNode<Let> {
 public:
  const Expr* var() const {
    return var_;
  }
  const Expr* value() const {
    return value_;
  }
  const Expr* body() const {
    return body_;
  }

  static ExprHandle make(const ExprHandle& var, const ExprHandle& value, const ExprHandle& body) {
    return ExprHandle(new Let(var.node(), value.node(), body.node()));
  }

  Let(const Expr* var, const Expr* value, const Expr* body)
      : ExprNodeBase(body->dtype()), var_(var), value_(value), body_(body) {}

 private:
  const Expr* var_;
  const Expr* value_;
  const Expr* body_;
};

// Represents a ramp vector node:
//     [base, base + 1 * stride, ... , base + (lanes - 1) * stride]
class Ramp : public ExprNode<Ramp> {
 public:
  const Expr* base() const {
    return base_;
  }
  const Expr* stride() const {
    return stride_;
  }
  static ExprHandle make(const ExprHandle& base, const ExprHandle& stride, int lanes) {
    return ExprHandle(new Ramp(base.node(), stride.node(), lanes));
  }
  int lanes() const {
    return lanes_;
  }

  Ramp(const Expr* base, const Expr* stride, int lanes)
      : ExprNodeBase(Dtype(base->dtype(), lanes)),
        base_(base),
        stride_(stride),
        lanes_(lanes) {
    CHECK_EQ(stride->dtype(), base->dtype());
  }

 private:
  const Expr* base_;
  const Expr* stride_;
  int lanes_;
};

class TORCH_API Load : public ExprNode<Load> {
 public:
  const Var* base_handle() const {
    return base_handle_;
  }
  const Expr* index() const {
    return index_;
  }
  const Expr* mask() const {
    return mask_;
  }
  static ExprHandle make(const Buffer& buffer, const ExprHandle& index, const ExprHandle& mask) {
    return ExprHandle(new Load(buffer, index.node(), mask.node()));
  }
  static ExprHandle make(
      Dtype dtype,
      const VarHandle& base_handle,
      const ExprHandle& index,
      const ExprHandle& mask) {
    return ExprHandle(new Load(dtype, base_handle.node(), index.node(), mask.node()));
  }

  Load(const Buffer& buffer, const Expr* index, const Expr* mask);
  Load(
      Dtype dtype,
      const Var* base_handle,
      const Expr* index,
      const Expr* mask);

 private:
  const Var* base_handle_;
  const Expr* index_;
  const Expr* mask_;
};

class Broadcast : public ExprNode<Broadcast> {
 public:
  const Expr* value() const {
    return value_;
  }
  int lanes() const {
    return lanes_;
  }
  static ExprHandle make(const ExprHandle& value, int lanes) {
    return ExprHandle(new Broadcast(value.node(), lanes));
  }
  Broadcast(const Expr* value, int lanes)
      : ExprNodeBase(Dtype(value->dtype(), lanes)),
        value_(value),
        lanes_(lanes) {}

 private:
  const Expr* value_;
  int lanes_;
};

class IfThenElse : public ExprNode<IfThenElse> {
 public:
  const Expr* condition() const {
    return condition_;
  }

  // Lazily evaluated only if condition is true
  const Expr* true_value() const {
    return true_;
  }

  // Lazily evaluated only if condition is false
  const Expr* false_value() const {
    return false_;
  }

  static ExprHandle make(const ExprHandle& c, const ExprHandle& t, const ExprHandle& f) {
    return ExprHandle(new IfThenElse(c.node(), t.node(), f.node()));
  }

  IfThenElse(const Expr* c, const Expr* t, const Expr* f)
      : ExprNodeBase(t->dtype()), condition_(c), true_(t), false_(f) {
    CHECK_EQ(c->dtype().scalar_type(), kInt32);
    CHECK_EQ(c->dtype().lanes(), 1);
    CHECK_EQ(t->dtype(), f->dtype());
  }

 private:
  const Expr* condition_;
  const Expr* true_;
  const Expr* false_;
};

class BaseCallNode : public Expr {
 public:
  enum CallType {
    kIntrinsics,
    kFunctionCall,
  };

  int nparams() const {
    return params_.size();
  }

  const Expr* param(int index) const {
    return params_[index];
  }
  const std::vector<const Expr*>& params() const {
    return params_;
  }

  virtual std::string func_name() const = 0;

  CallType call_type() const {
    return call_type_;
  }

 protected:
  BaseCallNode(Dtype dtype, CallType call_type, const std::vector<const Expr*>& params)
      : Expr(dtype), call_type_(call_type), params_(params) {}

 private:
  // The handler for the default ir_mutator to make a copy of this node with new
  // params.
  virtual const Expr* DefaultMutator(const std::vector<const Expr*>& new_params) const = 0;

  template <class U, class B>
  friend class ExprNode;
  friend class IRMutator;

  CallType call_type_;
  std::vector<const Expr*> params_;
};

template <typename Op>
class CallNode : public ExprNode<Op, BaseCallNode> {
 public:
  using BaseClass = ExprNode<Op, BaseCallNode>;
  using BaseClass::BaseClass;
};

class TORCH_API CompareSelect : public ExprNode<CompareSelect> {
 public:
  CompareSelectOperation compare_select_op() const {
    return compare_op_;
  }
  const Expr* lhs() const {
    return this->lhs_;
  }
  const Expr* rhs() const {
    return this->rhs_;
  }
  const Expr* ret_val1() const {
    return this->ret_val1_;
  }
  const Expr* ret_val2() const {
    return this->ret_val2_;
  }

  static ExprHandle make(
      const ExprHandle& lhs,
      const ExprHandle& rhs,
      CompareSelectOperation cmp_op) {
    CHECK_EQ(lhs.dtype(), rhs.dtype());
    return ExprHandle(new CompareSelect(
        lhs.node(),
        rhs.node(),
        IntImm::make(1).node(),
        IntImm::make(0).node(),
        cmp_op));
  }

  static ExprHandle make(
      const ExprHandle& lhs,
      const ExprHandle& rhs,
      const ExprHandle& ret_val1,
      const ExprHandle& ret_val2,
      CompareSelectOperation cmp_op) {
    CHECK_EQ(lhs.dtype(), rhs.dtype());
    CHECK_EQ(ret_val1.dtype(), ret_val2.dtype());
    return ExprHandle(new CompareSelect(
        lhs.node(), rhs.node(), ret_val1.node(), ret_val2.node(), cmp_op));
  }

 private:
  const Expr* lhs_;
  const Expr* rhs_;
  const Expr* ret_val1_;
  const Expr* ret_val2_;
  CompareSelectOperation compare_op_;
  CompareSelect(
      const Expr* lhs,
      const Expr* rhs,
      const Expr* ret_val1,
      const Expr* ret_val2,
      CompareSelectOperation cmp_op)
      : ExprNodeBase(ToDtype<int>()),
        lhs_(lhs),
        rhs_(rhs),
        ret_val1_(ret_val1),
        ret_val2_(ret_val2),
        compare_op_(cmp_op) {}
};

enum IntrinsicsOp {
  kSin,
  kCos,
  kTan,
  kAsin,
  kAcos,
  kAtan,
  kAtan2,
  kSinh,
  kCosh,
  kTanh,
  kExp,
  kExpm1,
  kFabs,
  kLog,
  kLog2,
  kLog10,
  kLog1p,
  kErf,
  kErfc,
  kSqrt,
  kRsqrt,
  kPow,
  kCeil,
  kFloor,
  kRound,
  kTrunc,
  kFmod,
  kRemainder,
  kLgamma,
  kFrac,
  kRand, // We need more discussions on this. Should we consider stateful?
};

class Intrinsics : public CallNode<Intrinsics> {
 public:
  static ExprHandle make(IntrinsicsOp op_type, const ExprHandle& v1) {
    return ExprHandle(new Intrinsics(op_type, v1.node()));
  }

  static ExprHandle make(IntrinsicsOp op_type, const ExprHandle& v1, const ExprHandle& v2) {
    return ExprHandle(new Intrinsics(op_type, v1.node(), v2.node()));
  }

  static ExprHandle make(IntrinsicsOp op_type, const std::vector<ExprHandle>& params) {
    std::vector<const Expr*> params_nodes(params.size());
    for (size_t i = 0; i < params.size(); i++) {
      params_nodes[i] = params[i].node();
    }
    return ExprHandle(new Intrinsics(op_type, params_nodes));
  }

  static ExprHandle make(IntrinsicsOp op_type, Dtype dtype) {
    return ExprHandle(new Intrinsics(op_type, dtype));
  }

  IntrinsicsOp op_type() const {
    return op_type_;
  }

  std::string func_name() const override {
    switch (op_type()) {
      case kSin:
        return "sin";
      case kCos:
        return "cos";
      case kTan:
        return "tan";
      case kAsin:
        return "asin";
      case kAcos:
        return "acos";
      case kAtan:
        return "atan";
      case kAtan2:
        return "atan2";
      case kSinh:
        return "sinh";
      case kCosh:
        return "cosh";
      case kTanh:
        return "tanh";
      case kExp:
        return "exp";
      case kFabs:
        return "fabs";
      case kLog:
        return "log";
      case kLog2:
        return "log2";
      case kLog10:
        return "log10";
      case kLog1p:
        return "log1p";
      case kErf:
        return "erf";
      case kSqrt:
        return "sqrt";
      case kRsqrt:
        return "rsqrt";
      case kPow:
        return "pow";
      case kCeil:
        return "ceil";
      case kFloor:
        return "floor";
      case kRound:
        return "round";
      case kTrunc:
        return "trunc";
      case kRand:
        return "rand";
      case kFmod:
        return "fmod";
      case kRemainder:
        return "remainder";
      case kLgamma:
        return "lgamma";
      case kExpm1:
        return "expm1";
      case kErfc:
        return "erfc";
      case kFrac:
        return "frac";
      default:
        throw std::runtime_error(
            "invalid op_type: " + std::to_string(op_type()));
    }
  }
  using BaseClass = CallNode<Intrinsics>;

  Intrinsics(IntrinsicsOp op_type, Dtype dtype)
      : BaseClass(IntrinsicsDtype(op_type, dtype), kIntrinsics, {}),
        op_type_(op_type) {
    CHECK_EQ(OpArgCount(op_type), 0);
  }

  Intrinsics(IntrinsicsOp op_type, const Expr* v1)
      : BaseClass(IntrinsicsDtype(op_type, v1->dtype()), kIntrinsics, {v1}),
        op_type_(op_type) {
    CHECK_EQ(OpArgCount(op_type), 1);
  }

  Intrinsics(IntrinsicsOp op_type, const Expr* v1, const Expr* v2)
      : BaseClass(
            IntrinsicsDtype(op_type, v1->dtype(), v2->dtype()),
            kIntrinsics,
            {v1, v2}),
        op_type_(op_type) {
    CHECK_EQ(OpArgCount(op_type), 2);
  }

  Intrinsics(IntrinsicsOp op_type, const std::vector<const Expr*>& params)
      : BaseClass(IntrinsicsDtype(op_type, params), kIntrinsics, params),
        op_type_(op_type) {
    CHECK_EQ(OpArgCount(op_type), nparams());
  }

 private:

  TORCH_API static int OpArgCount(IntrinsicsOp op_type);

  const Expr* DefaultMutator(const std::vector<const Expr*>& new_params) const override {
    return new Intrinsics(this->op_type(), new_params);
  }

  TORCH_API static Dtype IntrinsicsDtype(IntrinsicsOp op_type, Dtype dt1);
  TORCH_API static Dtype IntrinsicsDtype(
      IntrinsicsOp op_type,
      Dtype dt1,
      Dtype dt2);
  TORCH_API static Dtype IntrinsicsDtype(
      IntrinsicsOp op_type,
      const std::vector<const Expr*>& params);

  IntrinsicsOp op_type_;
};

class FunctionCall;

TORCH_API std::vector<const Expr*> ExprHandleVectorToExprVector(const std::vector<ExprHandle>&);
TORCH_API std::vector<ExprHandle> ExprVectorToExprHandleVector(const std::vector<const Expr*>&);
TORCH_API std::vector<const Var*> VarHandleVectorToVarVector(const std::vector<VarHandle>&);
TORCH_API std::vector<VarHandle> VarVectorToVarHandleVector(const std::vector<const Var*>&);


} // namespace tensorexpr
} // namespace jit
} // namespace torch
