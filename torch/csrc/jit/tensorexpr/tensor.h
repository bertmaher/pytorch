#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <vector>

#include "torch/csrc/jit/tensorexpr/expr.h"
#include "torch/csrc/jit/tensorexpr/function.h"

namespace torch {
namespace jit {
namespace tensorexpr {
namespace schedule {
class TensorExprNode;
class ScheduleNode;
} // namespace schedule

using schedule::TensorExprNode;

class TORCH_API TensorOperation : public KernelScopedObject {
 public:
  void SplitWithTail(
      const VarHandler& loop_var,
      int factor,
      bool factor_on_inner,
      VarHandler* outer_var,
      VarHandler* inner_var,
      VarHandler* tail_var,
      TensorOperation** tail_op);

  void SplitWithMask(
      const VarHandler& loop_var,
      int factor,
      bool factor_on_inner,
      VarHandler* outer_var,
      VarHandler* inner_var);

  void ComputeInline();

  void GPUExecConfig(
      const std::vector<VarHandler>& blockIdx,
      const std::vector<VarHandler>& threadIdx);

  TensorExprNode* expr_node() {
    return expr_node_;
  }

 protected:
  TensorOperation() {}
  explicit TensorOperation(TensorExprNode* expr_node) : expr_node_(expr_node) {}

 private:
  void check_expr_node();

  friend class schedule::ScheduleNode;
  TensorExprNode* expr_node_ = nullptr;
};

class Tensor : public TensorOperation {
 public:
  Function* function() const {
    return function_;
  }
  int output_index() const {
    return output_index_;
  }
  const VarHandler& arg(int index) const {
    return function_->arg(index);
  }

  Tensor(Function* function, int output_index)
      : function_(function), output_index_(output_index) {}
  template <typename... Ts>
  inline ExprHandler operator()(const Ts&... ts);
  template <typename T>
  inline ExprHandler call(const std::vector<T>& args);
  template <typename... Ts>
  inline ExprHandler call(const Ts&... ts);

 private:
  Function* function_;
  int output_index_;
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
  DimArg(const ExprHandler& dim) : dim_(dim) {}
  DimArg(const ExprHandler& dim, const std::string& name_hint)
      : dim_(dim), name_hint_(name_hint) {}
  const ExprHandler& dim() const {
    return dim_;
  }
  const std::string& name_hint() const {
    return name_hint_;
  }

 private:
  ExprHandler dim_;
  std::string name_hint_;
};

TORCH_API Tensor* Compute(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    std::function<ExprHandler(const VarHandler&)> body_func);
TORCH_API Tensor* Compute(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    std::function<ExprHandler(const VarHandler&, const VarHandler&)> body_func);
TORCH_API Tensor* Compute(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    std::function<ExprHandler(const VarHandler&, const VarHandler&, const VarHandler&)> body_func);
TORCH_API Tensor* Compute(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    std::function<ExprHandler(const VarHandler&, const VarHandler&, const VarHandler&, const VarHandler&)>
        body_func);
TORCH_API Tensor* Compute(
    const std::string& func_name,
    const std::vector<DimArg>& dim_args,
    std::function<ExprHandler(const std::vector<VarHandler>&)> body_func);

class FunctionCall : public CallNode<FunctionCall> {
 public:
  using BaseClass = CallNode<FunctionCall>;
  static ExprHandler make(Tensor* tensor, const std::vector<ExprHandler>& params) {
    std::vector<const Expr*> params_nodes(params.size());
    for (size_t i = 0; i < params.size(); i++) {
      params_nodes[i] = params[i].node();
    }
    return ExprHandler(new FunctionCall(tensor, params_nodes));
  }

  const Tensor* tensor() const {
    return tensor_;
  }
  Tensor* tensor() {
    return tensor_;
  }

  FunctionCall(Tensor* tensor, const std::vector<const Expr*>& params)
      : BaseClass(tensor->function()->body().dtype(), kFunctionCall, params),
        tensor_(tensor) {}
 private:
  const Expr* DefaultMutator(const std::vector<const Expr*>& new_params) const override {
    return new FunctionCall(tensor_, new_params);
  }

  std::string func_name() const {
    return tensor_->function()->func_var().name_hint();
  }

  Tensor* tensor_;
};
template <typename... Ts>
inline ExprHandler Tensor::operator()(const Ts&... ts) {
  std::vector<ExprHandler> params({ExprHandler(ts)...});
  return FunctionCall::make(this, std::move(params));
}

template <typename... Ts>
inline ExprHandler Tensor::call(const Ts&... ts) {
  std::vector<ExprHandler> params({ExprHandler(ts)...});
  return FunctionCall::make(this, std::move(params));
}

template <typename T>
inline ExprHandler Tensor::call(const std::vector<T>& args) {
  std::vector<ExprHandler> params(args.begin(), args.end());
  return FunctionCall::make(this, params);
}
} // namespace tensorexpr
} // namespace jit
} // namespace torch
