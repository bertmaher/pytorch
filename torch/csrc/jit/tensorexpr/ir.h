#pragma once

#include <string>
#include <vector>

#include "torch/csrc/jit/tensorexpr/expr.h"

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
  const BaseExprNode* src_value() const {
    return src_value_;
  }
  static Expr make(Dtype dtype, const Expr& src_value) {
    return Expr(new Cast(dtype, src_value.node()));
  }
  Cast(Dtype dtype, const BaseExprNode* src_value)
      : ExprNodeBase(dtype), src_value_(src_value) {}

 private:
  const BaseExprNode* src_value_;
};

template <typename T>
Expr cast(const Expr& src_value) {
  return Cast::make(Dtype(ToDtype<T>(), src_value.dtype().lanes()), src_value);
}

// Represent the expression node for binary operators.
// A CRTP pattern to share common code among the operators.
template <typename Op>
class BinaryOpNode : public ExprNode<Op> {
 public:
  const BaseExprNode* lhs() const {
    return this->lhs_;
  }
  const BaseExprNode* rhs() const {
    return this->rhs_;
  }
  IRNodeType expr_type() const {
    return expr_type_;
  }

  static Expr make(const Expr& lhs, const Expr& rhs) {
    return Expr(new Op(lhs.node(), rhs.node()));
  }

  BinaryOpNode(
      const BaseExprNode* lhs_v,
      const BaseExprNode* rhs_v,
      IRNodeType expr_type,
      ReturnType ret_type = ReturnType::knone)
      : ExprNode<Op>(BinaryOpDtype(lhs_v->dtype(), rhs_v->dtype(), ret_type)),
        lhs_(CastIfNeeded(lhs_v, ExprNode<Op>::dtype())),
        rhs_(CastIfNeeded(rhs_v, ExprNode<Op>::dtype())),
        expr_type_(expr_type) {}

 private:
  static const BaseExprNode* CastIfNeeded(const BaseExprNode* expr, Dtype dst_dtype) {
    if (expr->dtype() == dst_dtype) {
      return expr;
    }
    return Cast::make(dst_dtype, Expr(expr)).node();
  }

  const BaseExprNode* lhs_;
  const BaseExprNode* rhs_;
  IRNodeType expr_type_;
};

class Add : public BinaryOpNode<Add> {
 public:
  Add(const BaseExprNode* lhs, const BaseExprNode* rhs)
      : BinaryOpNode(lhs, rhs, IRNodeType::kAdd) {}
};

class Sub : public BinaryOpNode<Sub> {
 public:
  Sub(const BaseExprNode* lhs, const BaseExprNode* rhs)
      : BinaryOpNode(lhs, rhs, IRNodeType::kSub) {}
};

class Mul : public BinaryOpNode<Mul> {
 public:
  Mul(const BaseExprNode* lhs, const BaseExprNode* rhs)
      : BinaryOpNode(lhs, rhs, IRNodeType::kMul) {}
};

class Div : public BinaryOpNode<Div> {
 public:
  Div(const BaseExprNode* lhs, const BaseExprNode* rhs)
      : BinaryOpNode(lhs, rhs, IRNodeType::kDiv) {}
};

class Mod : public BinaryOpNode<Mod> {
 public:
  Mod(const BaseExprNode* lhs, const BaseExprNode* rhs)
      : BinaryOpNode(lhs, rhs, IRNodeType::kMod) {}
};

class Max : public BinaryOpNode<Max> {
 private:
  bool propagate_nans_;

 public:
  Max(const BaseExprNode* lhs, const BaseExprNode* rhs, bool propagate_nans)
      : BinaryOpNode(lhs, rhs, IRNodeType::kMax),
        propagate_nans_(propagate_nans) {}

  bool propagate_nans() const {
    return propagate_nans_;
  }

  static Expr make(const Expr& lhs, const Expr& rhs) = delete;
  static Expr make(const Expr& lhs, const Expr& rhs, bool propagate_nans) {
    return Expr(new Max(lhs.node(), rhs.node(), propagate_nans));
  }
};

class Min : public BinaryOpNode<Min> {
 private:
  bool propagate_nans_;

 public:
  Min(const BaseExprNode* lhs, const BaseExprNode* rhs, bool propagate_nans)
      : BinaryOpNode(lhs, rhs, IRNodeType::kMin),
        propagate_nans_(propagate_nans) {}

  bool propagate_nans() const {
    return propagate_nans_;
  }

  static Expr make(const Expr& lhs, const Expr& rhs) = delete;
  static Expr make(const Expr& lhs, const Expr& rhs, bool propagate_nans) {
    return Expr(new Min(lhs.node(), rhs.node(), propagate_nans));
  }
};

// Encode an integer immediate value.
class IntImm : public ExprNode<IntImm> {
 public:
  int value() const {
    return value_;
  }
  static Expr make(int value) {
    return Expr(new IntImm(value));
  }

 private:
  IntImm(int value) : ExprNodeBase(kInt32), value_(value) {}
  int value_;
};

// Encode an fp32 immediate value.
class FloatImm : public ExprNode<FloatImm> {
 public:
  float value() const {
    return value_;
  }
  static Expr make(float value) {
    return Expr(new FloatImm(value));
  }

 private:
  FloatImm(float value) : ExprNodeBase(kFloat32), value_(value) {}
  float value_;
};

// The underlying representation node to a Variable.
// Currently, each Variable object represents a unique variable, even though the
// names might be the same. We should consider add a unique_name as well.
class Variable : public ExprNode<Variable> {
 public:
  static Expr make(const std::string& name_hint, Dtype dtype) {
    return Expr(new Variable(name_hint, dtype));
  }
  static Expr make(Dtype dtype) {
    return Expr(new Variable("", dtype));
  }

  // TODO: unique_name
  const std::string& name_hint() const {
    return name_hint_;
  }

 private:
  Variable(const std::string& name_hint, Dtype dtype)
      : ExprNodeBase(dtype), name_hint_(name_hint) {}
  std::string name_hint_;
};

// An expression to construct the underlying variable node.
// Note: do not store any info here, since it is often possible to slice this
// object. For example: Var x('x'); Expr x2 = x;
class Var : public Expr {
 public:
  Var() : Expr(nullptr) {}
  explicit Var(Dtype dtype) : Expr(Variable::make(dtype)) {}
  Var(const std::string& name_hint, Dtype dtype)
      : Expr(Variable::make(name_hint, dtype)) {}
  explicit Var(const Variable* node) : Expr(node) {}
  const Variable* node() const {
    return static_cast<const Variable*>(Expr::node());
  }
  bool operator==(const Var& other) const {
    return this->node() == other.node();
  }
  bool operator!=(const Var& other) const {
    return !(*this == other);
  }

  const std::string& name_hint() const {
    return this->node()->name_hint();
  }
  bool empty() const {
    return (this->node() == nullptr);
  }
};

// Bind the value to the var and evaluate the body.
class Let : public ExprNode<Let> {
 public:
  const BaseExprNode* var() const {
    return var_;
  }
  const BaseExprNode* value() const {
    return value_;
  }
  const BaseExprNode* body() const {
    return body_;
  }

  static Expr make(const Expr& var, const Expr& value, const Expr& body) {
    return Expr(new Let(var.node(), value.node(), body.node()));
  }

  Let(const BaseExprNode* var, const BaseExprNode* value, const BaseExprNode* body)
      : ExprNodeBase(body->dtype()), var_(var), value_(value), body_(body) {}

 private:
  const BaseExprNode* var_;
  const BaseExprNode* value_;
  const BaseExprNode* body_;
};

class LetStmt : public StmtNode<LetStmt> {
 public:
  const Variable* var() const {
    return var_;
  }

  const BaseExprNode* value() const {
    return value_;
  }

  Stmt* body() const {
    return body_;
  }

  static Stmt* make(const Var& var, const Expr& value, Stmt* body) {
    return new LetStmt(var.node(), value.node(), body);
  }

  LetStmt(const Variable* var, const BaseExprNode* value, Stmt* body)
      : var_(var), value_(value), body_(body) {}

 private:
  const Variable* var_;
  const BaseExprNode* value_;
  Stmt* body_;
};

class Block : public StmtNode<Block> {
 public:
  static Stmt* make(const std::vector<Stmt*>& stmts) {
    std::vector<Stmt*> valid_stmts;
    for (size_t i = 0; i < stmts.size(); i++) {
      if (!stmts[i]) {
        continue;
      }
      valid_stmts.push_back(stmts[i]);
    }
    if (valid_stmts.empty()) {
      return nullptr;
    }
    return new Block(valid_stmts);
  }
  int nstmts() const {
    return stmts_.size();
  }
  Stmt* stmt(int index) const {
    return stmts_[index];
  }

 private:
  explicit Block(const std::vector<Stmt*>& stmts) : stmts_(stmts) {}
  std::vector<Stmt*> stmts_;
};

class LoopOptions {
 public:
  // GPU Block Index
  bool is_gpu_block_index() const {
    return gpu_block_index_ != -1;
  }

  bool gpu_block_index() const {
    return gpu_block_index_;
  }

  std::string gpu_block_index_str() const {
    DCHECK(is_gpu_block_index());
    static const char* kBlockIndexNames[] = {
        "blockIdx.x",
        "blockIdx.y",
        "blockIdx.z",
        "blockIdx.w",
    };
    DCHECK(gpu_block_index_ >= 0 && gpu_block_index_ < 4);
    return kBlockIndexNames[gpu_block_index_];
  }

  void set_gpu_block_index(int index) {
    if (is_gpu_thread_index()) {
      throw std::runtime_error("Cannot set both gpu block and thread index");
    }
    if (is_gpu_block_index() && gpu_block_index() != index) {
      throw std::runtime_error(
          "Cannot set a previously set block index: " +
          std::to_string(gpu_block_index()) + " vs " + std::to_string(index));
    }
    gpu_block_index_ = index;
  }

  // GPU Thread Index
  bool is_gpu_thread_index() const {
    return gpu_thread_index() != -1;
  }

  int gpu_thread_index() const {
    return gpu_thread_index_;
  }

  std::string gpu_thread_index_str() const {
    DCHECK(is_gpu_thread_index());
    static const char* kThreadIndexNames[] = {
        "threadIdx.x", "threadIdx.y", "threadIdx.z", "threadIdx.w"};
    DCHECK(gpu_thread_index_ >= 0 && gpu_thread_index_ < 4);
    return kThreadIndexNames[gpu_thread_index_];
  }

  void set_gpu_thread_index(int index) {
    if (is_gpu_block_index()) {
      throw std::runtime_error("Cannot set both gpu thread and block index");
    }
    if (is_gpu_thread_index() && gpu_thread_index() != index) {
      throw std::runtime_error(
          "Cannot set a previously set thread index: " +
          std::to_string(gpu_thread_index()) + " vs " + std::to_string(index));
    }
    gpu_thread_index_ = index;
  }

  std::string ToString() const {
    std::ostringstream oss;
    if (is_gpu_block_index()) {
      oss << gpu_block_index_str();
    } else if (is_gpu_thread_index()) {
      oss << gpu_thread_index_str();
    }
    return oss.str();
  }

 private:
  int gpu_block_index_ = -1;
  int gpu_thread_index_ = -1;
};

class For : public StmtNode<For> {
 public:
  const Variable* var() const {
    return var_;
  }
  const BaseExprNode* start() const {
    return start_;
  }
  const BaseExprNode* stop() const {
    return stop_;
  }
  Stmt* body() const {
    return body_;
  }
  static Stmt* make(
      const Var& var,
      const Expr& start,
      const Expr& stop,
      Stmt* body) {
    if (!body) {
      return nullptr;
    }
    return new For(var.node(), start.node(), stop.node(), body);
  }
  static Stmt* make(
      const Var& var,
      const Expr& start,
      const Expr& stop,
      Stmt* body,
      const LoopOptions& loop_options) {
    if (!body) {
      return nullptr;
    }
    return new For(var.node(), start.node(), stop.node(), body, loop_options);
  }
  const LoopOptions loop_options() const {
    return loop_options_;
  }

  For(const Variable* var, const BaseExprNode* start, const BaseExprNode* stop, Stmt* body)
      : var_(var), start_(start), stop_(stop), body_(body) {
          CHECK(var && start && stop && body);
      }

  For(const Variable* var,
      const BaseExprNode* start,
      const BaseExprNode* stop,
      Stmt* body,
      const LoopOptions& loop_options)
      : var_(var),
        start_(start),
        stop_(stop),
        body_(body),
        loop_options_(loop_options) {
          CHECK(var && start && stop && body);
        }

 private:
  const Variable* var_;
  const BaseExprNode* start_;
  const BaseExprNode* stop_;
  Stmt* body_;
  LoopOptions loop_options_;
};

// Represents a ramp vector node:
//     [base, base + 1 * stride, ... , base + (lanes - 1) * stride]
class Ramp : public ExprNode<Ramp> {
 public:
  const BaseExprNode* base() const {
    return base_;
  }
  const BaseExprNode* stride() const {
    return stride_;
  }
  static Expr make(const Expr& base, const Expr& stride, int lanes) {
    return Expr(new Ramp(base.node(), stride.node(), lanes));
  }
  int lanes() const {
    return lanes_;
  }

  Ramp(const BaseExprNode* base, const BaseExprNode* stride, int lanes)
      : ExprNodeBase(Dtype(base->dtype(), lanes)),
        base_(base),
        stride_(stride),
        lanes_(lanes) {
    CHECK_EQ(stride->dtype(), base->dtype());
  }

 private:
  const BaseExprNode* base_;
  const BaseExprNode* stride_;
  int lanes_;
};

class TORCH_API Load : public ExprNode<Load> {
 public:
  const Variable* base_handle() const {
    return base_handle_;
  }
  const BaseExprNode* index() const {
    return index_;
  }
  const BaseExprNode* mask() const {
    return mask_;
  }
  static Expr make(const Buffer& buffer, const Expr& index, const Expr& mask) {
    return Expr(new Load(buffer, index.node(), mask.node()));
  }
  static Expr make(
      Dtype dtype,
      const Var& base_handle,
      const Expr& index,
      const Expr& mask) {
    return Expr(new Load(dtype, base_handle.node(), index.node(), mask.node()));
  }

  Load(const Buffer& buffer, const BaseExprNode* index, const BaseExprNode* mask);
  Load(
      Dtype dtype,
      const Variable* base_handle,
      const BaseExprNode* index,
      const BaseExprNode* mask);

 private:
  const Variable* base_handle_;
  const BaseExprNode* index_;
  const BaseExprNode* mask_;
};

class TORCH_API Store : public StmtNode<Store> {
 public:
  const Variable* base_handle() const {
    return base_handle_;
  }
  const BaseExprNode* index() const {
    return index_;
  }
  const BaseExprNode* value() const {
    return value_;
  }
  const BaseExprNode* mask() const {
    return mask_;
  }

  static Stmt* make(
      const Buffer& buffer,
      const Expr& index,
      const Expr& value,
      const Expr& mask) {
    return new Store(buffer, index.node(), value.node(), mask.node());
  }

  static Stmt* make(
      const Var& base_handle,
      const Expr& index,
      const Expr& value,
      const Expr& mask) {
    return new Store(base_handle.node(), index.node(), value.node(), mask.node());
  }

  static Stmt* make(
      const Var& base_handle,
      const Expr& index,
      const Expr& value) {
    return new Store(base_handle.node(), index.node(), value.node(), Expr(1).node());
  }

  // TODO: merge this with Load.
  Store(
      const Buffer& buffer,
      const BaseExprNode* index,
      const BaseExprNode* value,
      const BaseExprNode* mask);

  Store(
      const Variable* base_handle,
      const BaseExprNode* index,
      const BaseExprNode* value,
      const BaseExprNode* mask)
      : base_handle_(base_handle), index_(index), value_(value), mask_(mask) {
    CHECK_EQ(base_handle_->dtype(), kHandle);
    CHECK_EQ(index->dtype().lanes(), mask->dtype().lanes());
    CHECK_EQ(index->dtype().lanes(), value->dtype().lanes());
    CHECK_EQ(index->dtype().scalar_type(), kInt32);
  }
 private:

  const Variable* base_handle_;
  const BaseExprNode* index_;
  const BaseExprNode* value_;
  const BaseExprNode* mask_;
};

class Broadcast : public ExprNode<Broadcast> {
 public:
  const BaseExprNode* value() const {
    return value_;
  }
  int lanes() const {
    return lanes_;
  }
  static Expr make(const Expr& value, int lanes) {
    return Expr(new Broadcast(value.node(), lanes));
  }
  Broadcast(const BaseExprNode* value, int lanes)
      : ExprNodeBase(Dtype(value->dtype(), lanes)),
        value_(value),
        lanes_(lanes) {}

 private:
  const BaseExprNode* value_;
  int lanes_;
};

class IfThenElse : public ExprNode<IfThenElse> {
 public:
  const BaseExprNode* condition() const {
    return condition_;
  }

  // Lazily evaluated only if condition is true
  const BaseExprNode* true_value() const {
    return true_;
  }

  // Lazily evaluated only if condition is false
  const BaseExprNode* false_value() const {
    return false_;
  }

  static Expr make(const Expr& c, const Expr& t, const Expr& f) {
    return Expr(new IfThenElse(c.node(), t.node(), f.node()));
  }

  IfThenElse(const BaseExprNode* c, const BaseExprNode* t, const BaseExprNode* f)
      : ExprNodeBase(t->dtype()), condition_(c), true_(t), false_(f) {
    CHECK_EQ(c->dtype().scalar_type(), kInt32);
    CHECK_EQ(c->dtype().lanes(), 1);
    CHECK_EQ(t->dtype(), f->dtype());
  }

 private:
  const BaseExprNode* condition_;
  const BaseExprNode* true_;
  const BaseExprNode* false_;
};

class BaseCallNode : public BaseExprNode {
 public:
  enum CallType {
    kIntrinsics,
    kFunctionCall,
  };

  int nparams() const {
    return params_.size();
  }

  const BaseExprNode* param(int index) const {
    return params_[index];
  }
  const std::vector<const BaseExprNode*>& params() const {
    return params_;
  }

  virtual std::string func_name() const = 0;

  CallType call_type() const {
    return call_type_;
  }

 protected:
  BaseCallNode(Dtype dtype, CallType call_type, const std::vector<const BaseExprNode*>& params)
      : BaseExprNode(dtype), call_type_(call_type), params_(params) {}

 private:
  // The handler for the default ir_mutator to make a copy of this node with new
  // params.
  virtual const BaseExprNode* DefaultMutator(const std::vector<const BaseExprNode*>& new_params) const = 0;

  template <class U, class B>
  friend class ExprNode;
  friend class IRMutator;

  CallType call_type_;
  std::vector<const BaseExprNode*> params_;
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
  const BaseExprNode* lhs() const {
    return this->lhs_;
  }
  const BaseExprNode* rhs() const {
    return this->rhs_;
  }
  const BaseExprNode* ret_val1() const {
    return this->ret_val1_;
  }
  const BaseExprNode* ret_val2() const {
    return this->ret_val2_;
  }

  static Expr make(
      const Expr& lhs,
      const Expr& rhs,
      CompareSelectOperation cmp_op) {
    CHECK_EQ(lhs.dtype(), rhs.dtype());
    return Expr(new CompareSelect(
        lhs.node(),
        rhs.node(),
        IntImm::make(1).node(),
        IntImm::make(0).node(),
        cmp_op));
  }

  static Expr make(
      const Expr& lhs,
      const Expr& rhs,
      const Expr& ret_val1,
      const Expr& ret_val2,
      CompareSelectOperation cmp_op) {
    CHECK_EQ(lhs.dtype(), rhs.dtype());
    CHECK_EQ(ret_val1.dtype(), ret_val2.dtype());
    return Expr(new CompareSelect(
        lhs.node(), rhs.node(), ret_val1.node(), ret_val2.node(), cmp_op));
  }

 private:
  const BaseExprNode* lhs_;
  const BaseExprNode* rhs_;
  const BaseExprNode* ret_val1_;
  const BaseExprNode* ret_val2_;
  CompareSelectOperation compare_op_;
  CompareSelect(
      const BaseExprNode* lhs,
      const BaseExprNode* rhs,
      const BaseExprNode* ret_val1,
      const BaseExprNode* ret_val2,
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
  static Expr make(IntrinsicsOp op_type, const Expr& v1) {
    return Expr(new Intrinsics(op_type, v1.node()));
  }

  static Expr make(IntrinsicsOp op_type, const Expr& v1, const Expr& v2) {
    return Expr(new Intrinsics(op_type, v1.node(), v2.node()));
  }

  static Expr make(IntrinsicsOp op_type, const std::vector<Expr>& params) {
    std::vector<const BaseExprNode*> params_nodes(params.size());
    for (size_t i = 0; i < params.size(); i++) {
      params_nodes[i] = params[i].node();
    }
    return Expr(new Intrinsics(op_type, params_nodes));
  }

  static Expr make(IntrinsicsOp op_type, Dtype dtype) {
    return Expr(new Intrinsics(op_type, dtype));
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

  Intrinsics(IntrinsicsOp op_type, const BaseExprNode* v1)
      : BaseClass(IntrinsicsDtype(op_type, v1->dtype()), kIntrinsics, {v1}),
        op_type_(op_type) {
    CHECK_EQ(OpArgCount(op_type), 1);
  }

  Intrinsics(IntrinsicsOp op_type, const BaseExprNode* v1, const BaseExprNode* v2)
      : BaseClass(
            IntrinsicsDtype(op_type, v1->dtype(), v2->dtype()),
            kIntrinsics,
            {v1, v2}),
        op_type_(op_type) {
    CHECK_EQ(OpArgCount(op_type), 2);
  }

  Intrinsics(IntrinsicsOp op_type, const std::vector<const BaseExprNode*>& params)
      : BaseClass(IntrinsicsDtype(op_type, params), kIntrinsics, params),
        op_type_(op_type) {
    CHECK_EQ(OpArgCount(op_type), nparams());
  }

 private:

  TORCH_API static int OpArgCount(IntrinsicsOp op_type);

  const BaseExprNode* DefaultMutator(const std::vector<const BaseExprNode*>& new_params) const override {
    return new Intrinsics(this->op_type(), new_params);
  }

  TORCH_API static Dtype IntrinsicsDtype(IntrinsicsOp op_type, Dtype dt1);
  TORCH_API static Dtype IntrinsicsDtype(
      IntrinsicsOp op_type,
      Dtype dt1,
      Dtype dt2);
  TORCH_API static Dtype IntrinsicsDtype(
      IntrinsicsOp op_type,
      const std::vector<const BaseExprNode*>& params);

  IntrinsicsOp op_type_;
};

class FunctionCall;

// Allocate a buffer of given shapes and dtypes and bind it with the given
// buffer var. The life span is at most through the current program, until it is
// explicitly freed. An unfreed memory is likely considered an error.
class Allocate : public StmtNode<Allocate> {
 public:
  static Stmt* make(
      const Var& buffer_var,
      Dtype dtype,
      const std::vector<Expr>& dims) {
    std::vector<const BaseExprNode*> dims_nodes(dims.size());
    for (size_t i = 0; i < dims.size(); i++) {
      dims_nodes[i] = dims[i].node();
    }
    return new Allocate(buffer_var.node(), dtype, dims_nodes);
  }

  const Variable* buffer_var() const {
    return buffer_var_;
  }

  Dtype dtype() const {
    return dtype_;
  }

  const std::vector<const BaseExprNode*>& dims() const {
    return dims_;
  }

  Allocate(const Variable* buffer_var, Dtype dtype, const std::vector<const BaseExprNode*>& dims)
      : buffer_var_(buffer_var), dtype_(dtype), dims_(dims) {}

 private:
  const Variable* buffer_var_;
  Dtype dtype_;
  std::vector<const BaseExprNode*> dims_;
  // TODO: add memory types.
};

// Free the specific buffer. It is an error.
class Free : public StmtNode<Free> {
 public:
  static Stmt* make(const Var& buffer_var) {
    return new Free(buffer_var.node());
  }

  const Variable* buffer_var() const {
    return buffer_var_;
  }

  Free(const Variable* buffer_var) : buffer_var_(buffer_var) {}

 private:
  const Variable* buffer_var_;
};

class Cond : public StmtNode<Cond> {
 public:
  static Stmt* make(
      const Expr& condition,
      Stmt* true_stmt,
      Stmt* false_stmt) {
    return new Cond(condition.node(), true_stmt, false_stmt);
  }

  const BaseExprNode* condition() const {
    return condition_;
  }

  Stmt* true_stmt() const {
    return true_stmt_;
  }

  Stmt* false_stmt() const {
    return false_stmt_;
  }

  Cond(const BaseExprNode* condition, Stmt* true_stmt, Stmt* false_stmt)
      : condition_(condition), true_stmt_(true_stmt), false_stmt_(false_stmt) {}

 private:
  const BaseExprNode* condition_;
  Stmt* true_stmt_;
  Stmt* false_stmt_;
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
