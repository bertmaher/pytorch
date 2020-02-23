#include "torch/csrc/jit/tensorexpr/tensor.h"
#include "torch/csrc/jit/tensorexpr/schedule.h"

namespace torch {
namespace jit {
namespace tensorexpr {

using schedule::TensorExprNode;
// using schedule::ScheduleNode;

void TensorOperation::SplitWithTail(
    const VarHandler& loop_var,
    int factor,
    bool factor_on_inner,
    VarHandler* outer_var,
    VarHandler* inner_var,
    VarHandler* tail_var,
    TensorOperation** tail_op) {
  check_expr_node();
  schedule::ScheduleNode* schedule = expr_node_->schedule();
  schedule::TensorExprNode* tail_expr_node = nullptr;
  schedule->SplitWithTail(
      expr_node_,
      loop_var,
      factor,
      factor_on_inner,
      outer_var,
      inner_var,
      tail_var,
      &tail_expr_node);
  if (!tail_expr_node) {
    *tail_op = new TensorOperation(tail_expr_node);
  }
}

void TensorOperation::SplitWithMask(
    const VarHandler& loop_var,
    int factor,
    bool factor_on_inner,
    VarHandler* outer_var,
    VarHandler* inner_var) {
  check_expr_node();
  schedule::ScheduleNode* schedule = expr_node_->schedule();
  schedule::TensorExprNode* tail_expr_node = nullptr;
  schedule->SplitWithMask(
      expr_node_, loop_var, factor, factor_on_inner, outer_var, inner_var);
}

void TensorOperation::GPUExecConfig(
    const std::vector<VarHandler>& blockIdx,
    const std::vector<VarHandler>& threadIdx) {
  check_expr_node();
  schedule::ScheduleNode* schedule = expr_node_->schedule();
  schedule->GPUExecConfig(expr_node_, blockIdx, threadIdx);
}

void TensorOperation::ComputeInline() {
  // TODO: find a better way to detect that no schedule might be created for this.
  // Even though this operation might be used at the Torch JIT level, it might be
  // still be pruned out at the expression level, such as "y = rand_like(x)".
  // For now, we tentatively treat as if this tensor is not part of the schedule.
  if (expr_node_ == nullptr) {
    return;
  }
  schedule::ScheduleNode* schedule = expr_node_->schedule();
  schedule->ComputeInline(expr_node_);
}

void TensorOperation::check_expr_node() {
  if (expr_node_ == nullptr) {
    throw std::runtime_error(
        "expr_node in this tensor is null. It is likely that no schedule is attached.");
  }
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
