#pragma once

#include <vector>

#include "torch/csrc/jit/tensorexpr/expr.h"
#include "torch/csrc/jit/tensorexpr/function.h"
#include "torch/csrc/jit/tensorexpr/refcount.h"

namespace torch {
namespace jit {
namespace compiler {
namespace schedule {
class TensorExprNode;
class ScheduleNode;
} // namespace schedule

using schedule::TensorExprNode;

class TensorOperation;
class TensorOperationNode : public RefCounted {
 public:
  void SplitWithTail(
      const Var& loop_var,
      int factor,
      bool factor_on_inner,
      Var* outer_var,
      Var* inner_var,
      Var* tail_var,
      TensorOperation* tail_op);

  void ComputeInline();

  TensorExprNode* expr_node() {
    return expr_node_;
  }

 protected:
  TensorOperationNode() {}
  explicit TensorOperationNode(TensorExprNode* expr_node)
      : expr_node_(expr_node) {}

 private:
  void check_expr_node();

  friend class TensorOperation;
  friend class schedule::ScheduleNode;
  TensorExprNode* expr_node_ = nullptr;
};

class TensorNode : public TensorOperationNode {
 public:
  int ndim() const {
    return function_.ndim();
  }
  const Expr& dim(int index) const {
    return function_.dim(index);
  }
  const std::vector<Expr>& dims() const {
    return function_.dims();
  }
  const Function& function() const {
    return function_;
  }
  int output_index() const {
    return output_index_;
  }
  const Var& buffer_var() const {
    return function_.func_var();
  }
  Dtype dtype() const {
    return function_.body().dtype();
  }

 private:
  friend class Tensor;
  TensorNode(const Function& function, int output_index)
      : function_(function), output_index_(output_index) {}
  Function function_;
  int output_index_;
};

class TensorOperation : public RefHandle<TensorOperationNode> {
 public:
  using BaseClass = RefHandle<TensorOperationNode>;
  TensorOperation() : BaseClass(nullptr) {}
  static TensorOperation make() {
    return TensorOperation(new TensorOperationNode());
  }
  static TensorOperation make(TensorExprNode* expr_node) {
    return TensorOperation(new TensorOperationNode(expr_node));
  }
  TensorExprNode* expr_node() {
    return node()->expr_node();
  }

  void SplitWithTail(
      const Var& loop_var,
      int factor,
      bool factor_on_inner,
      Var* outer_var,
      Var* inner_var,
      Var* tail_var,
      TensorOperation* tail_op) {
    return node()->SplitWithTail(
        loop_var,
        factor,
        factor_on_inner,
        outer_var,
        inner_var,
        tail_var,
        tail_op);
  }

  void ComputeInline() {
    node()->ComputeInline();
  }

 protected:
  TensorOperation(TensorOperationNode* node) : BaseClass(node) {}
};

class Tensor : public TensorOperation {
 public:
  Tensor(const Function& function, int output_index)
      : TensorOperation(new TensorNode(function, output_index)) {}

  explicit Tensor(TensorNode* tensor_node) : TensorOperation(tensor_node) {}

  int ndim() const {
    return node()->ndim();
  }
  const Expr& dim(int index) const {
    return node()->dim(index);
  }
  const std::vector<Expr>& dims() const {
    return node()->dims();
  }
  const Function& function() const {
    return node()->function();
  }
  int output_index() const {
    return node()->output_index();
  }
  const Var& buffer_var() const {
    return node()->buffer_var();
  }
  Dtype dtype() const {
    return node()->dtype();
  }

  template <typename... Ts>
  Expr operator()(const Ts&... ts) const;

  TensorNode* node() {
    // TODO: switch to dynamic_cast when it becomes available.
    return static_cast<TensorNode*>(TensorOperation::node());
  }

  const TensorNode* node() const {
    return const_cast<Tensor*>(this)->node();
  }

 private:
  friend class schedule::ScheduleNode;
};

// A helper structure to store the arguments to specify dimensions. In the
// Compute arugments for dim_args, all of the following is supported. For
// example:
//    dim_args: {1, 2, 3, 4}
//    dim_args: {{1, "x"}, {2, "y"}, {3, "z"}}
//    dim_args: {1, 2, {3, "x"}}
class DimArg {
 public:
  // Intentionally leave out explicit to allow implicit conversions.
  DimArg(const Expr& dim) : dim_(dim) {}
  DimArg(const Expr& dim, const std::string& name_hint)
      : dim_(dim), name_hint_(name_hint) {}
  const Expr& dim() const {
    return dim_;
  }
  const std::string& name_hint() const {
    return name_hint_;
  }

 private:
  Expr dim_;
  std::string name_hint_;
};

Tensor Compute(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    std::function<Expr(const Var&)> body_func);
Tensor Compute(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    std::function<Expr(const Var&, const Var&)> body_func);
Tensor Compute(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    std::function<Expr(const Var&, const Var&, const Var&)> body_func);
Tensor Compute(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    std::function<Expr(const Var&, const Var&, const Var&, const Var&)>
        body_func);
Tensor Compute(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    std::function<Expr(const std::vector<Var>&)> body_func);

class FunctionCall : public CallNode<FunctionCall> {
 public:
  using BaseClass = CallNode<FunctionCall>;
  static Expr make(const Tensor& tensor, const std::vector<Expr>& params) {
    return Expr(new FunctionCall(tensor, params));
  }

  const Tensor& tensor() const {
    return tensor_;
  }
  Tensor& tensor() {
    return tensor_;
  }

 private:
  Expr DefaultMutator(const std::vector<Expr>& new_params) const override {
    return FunctionCall::make(tensor_, new_params);
  }

  std::string func_name() const {
    return tensor_.function().func_var().name_hint();
  }

  FunctionCall(const Tensor& tensor, const std::vector<Expr>& params)
      : BaseClass(tensor.function().body().dtype(), kFunctionCall, params),
        tensor_(tensor) {}
  Tensor tensor_;
};

template <typename... Ts>
inline Expr Tensor::operator()(const Ts&... ts) const {
  std::vector<Expr> params({Expr(ts)...});
  return FunctionCall::make(*this, std::move(params));
}

} // namespace compiler
} // namespace jit
} // namespace torch
