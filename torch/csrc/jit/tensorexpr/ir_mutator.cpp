#include "torch/csrc/jit/tensorexpr/ir_mutator.h"

#include "torch/csrc/jit/tensorexpr/eval.h"
#include "torch/csrc/jit/tensorexpr/ir.h"

namespace torch {
namespace jit {
namespace tensorexpr {

template <typename Op>
static const BaseExprNode* mutate_binary_op(
    const BinaryOpNode<Op>* v,
    IRMutator* mutator,
    bool option = false) {
  const BaseExprNode* lhs = v->lhs();
  const BaseExprNode* rhs = v->rhs();
  const BaseExprNode* lhs_new = lhs->accept_mutator(mutator);
  const BaseExprNode* rhs_new = rhs->accept_mutator(mutator);
  if (lhs == lhs_new && rhs == rhs_new) {
    return v;
  }
  IRNodeType expr_type = v->expr_type();
  switch (expr_type) {
    case IRNodeType::kAdd:
      return new Add(lhs_new, rhs_new);
    case IRNodeType::kSub:
      return new Sub(lhs_new, rhs_new);
    case IRNodeType::kMul:
      return new Mul(lhs_new, rhs_new);
    case IRNodeType::kDiv:
      return new Div(lhs_new, rhs_new);
    case IRNodeType::kMod:
      return new Mod(lhs_new, rhs_new);
    case IRNodeType::kMax:
      return new Max(lhs_new, rhs_new, option);
    case IRNodeType::kMin:
      return new Min(lhs_new, rhs_new, option);
    default:
      LOG(FATAL) << "unsupported expr_type: " << static_cast<int>(expr_type);
      return nullptr;
  }
}

const BaseExprNode* IRMutator::mutate(const Add* v) {
  return mutate_binary_op(v, this);
}

const BaseExprNode* IRMutator::mutate(const Sub* v) {
  return mutate_binary_op(v, this);
}

const BaseExprNode* IRMutator::mutate(const Mul* v) {
  return mutate_binary_op(v, this);
}

const BaseExprNode* IRMutator::mutate(const Div* v) {
  return mutate_binary_op(v, this);
}

const BaseExprNode* IRMutator::mutate(const Mod* v) {
  return mutate_binary_op(v, this);
}

const BaseExprNode* IRMutator::mutate(const Max* v) {
  return mutate_binary_op(v, this, v->propagate_nans());
}

const BaseExprNode* IRMutator::mutate(const Min* v) {
  return mutate_binary_op(v, this, v->propagate_nans());
}

const BaseExprNode* IRMutator::mutate(const CompareSelect* v) {
  const BaseExprNode* lhs = v->lhs();
  const BaseExprNode* rhs = v->rhs();
  const BaseExprNode* retval1 = v->ret_val1();
  const BaseExprNode* retval2 = v->ret_val2();
  const BaseExprNode* lhs_new = lhs->accept_mutator(this);
  const BaseExprNode* rhs_new = rhs->accept_mutator(this);
  const BaseExprNode* retval1_new = retval1->accept_mutator(this);
  const BaseExprNode* retval2_new = retval2->accept_mutator(this);
  if (lhs == lhs_new && rhs == rhs_new && retval1 == retval1_new &&
      retval2 == retval2_new) {
    return v;
  }
  return CompareSelect::make(
             Expr(lhs_new),
             Expr(rhs_new),
             Expr(retval1_new),
             Expr(retval2_new),
             v->compare_select_op())
      .node();
}

const BaseExprNode* IRMutator::mutate(const IntImm* v) {
  return v;
}

const BaseExprNode* IRMutator::mutate(const FloatImm* v) {
  return v;
}

const BaseExprNode* IRMutator::mutate(const Cast* v) {
  const BaseExprNode* src_value = v->src_value();
  const BaseExprNode* src_value_new = src_value->accept_mutator(this);
  if (src_value_new == v->src_value()) {
    return v;
  }
  return new Cast(v->dtype(), src_value_new);
}

const BaseExprNode* IRMutator::mutate(const Variable* v) {
  return v;
}

const BaseExprNode* IRMutator::mutate(const Let* v) {
  const BaseExprNode* var = v->var();
  const BaseExprNode* value = v->value();
  const BaseExprNode* body = v->body();
  const BaseExprNode* var_new = var->accept_mutator(this);
  const BaseExprNode* value_new = value->accept_mutator(this);
  const BaseExprNode* body_new = body->accept_mutator(this);
  if ((var == var_new) && (value == value_new) &&
      (body == body_new)) {
    return v;
  }
  return new Let(var_new, value_new, body_new);
}

Stmt* IRMutator::mutate(const LetStmt* v) {
  const Variable* var = v->var();
  const BaseExprNode* value = v->value();
  Stmt* body = v->body();
  const Variable* var_new = dynamic_cast<const Variable*>(var->accept_mutator(this));
  if (var_new == nullptr) {
    throw std::runtime_error("LetStmt var must be variable");
  }
  const BaseExprNode* value_new = value->accept_mutator(this);
  Stmt* body_new = body->accept_mutator(this);
  if ((var == var_new) && (value == value_new) &&
      (body == body_new)) {
    return (Stmt*)v;
  }
  return new LetStmt(var_new, value_new, body_new);
}

const BaseExprNode* IRMutator::mutate(const Ramp* v) {
  const BaseExprNode* base = v->base();
  const BaseExprNode* stride = v->stride();
  const BaseExprNode* base_new = base->accept_mutator(this);
  const BaseExprNode* stride_new = stride->accept_mutator(this);
  if (base == base_new && stride == stride_new) {
    return v;
  }
  return new Ramp(base_new, stride_new, v->lanes());
}

const BaseExprNode* IRMutator::mutate(const Load* v) {
  Dtype dtype = v->dtype();
  const Variable* base_handle = v->base_handle();
  const BaseExprNode* index = v->index();
  const BaseExprNode* mask = v->mask();
  const BaseExprNode* base_handle_expr = base_handle->accept_mutator(this);
  const Variable* base_handle_new = dynamic_cast<const Variable*>(base_handle_expr);
  const BaseExprNode* index_new = index->accept_mutator(this);
  const BaseExprNode* mask_new = mask->accept_mutator(this);
  if (base_handle == base_handle_new && index == index_new &&
      mask == mask_new) {
    return v;
  }
  return new Load(dtype, base_handle_new, index_new, mask_new);
}

const BaseExprNode* IRMutator::mutate(const Broadcast* v) {
  const BaseExprNode* value = v->value();
  int lanes = v->lanes();
  const BaseExprNode* value_new = value->accept_mutator(this);
  if (value == value_new) {
    return v;
  }
  return new Broadcast(value_new, lanes);
}

const BaseExprNode* IRMutator::mutate(const IfThenElse* v) {
  const BaseExprNode* condition = v->condition();
  const BaseExprNode* true_value = v->true_value();
  const BaseExprNode* false_value = v->false_value();
  const BaseExprNode* condition_new = condition->accept_mutator(this);
  const BaseExprNode* true_value_new = true_value->accept_mutator(this);
  const BaseExprNode* false_value_new = false_value->accept_mutator(this);
  if (condition == condition_new &&
      true_value == true_value_new &&
      false_value == false_value_new) {
    return v;
  }

  return new IfThenElse(condition_new, true_value_new, false_value_new);
}

const BaseExprNode* IRMutator::mutate(const Intrinsics* v) {
  const BaseCallNode* base = v;
  return this->mutate(base);
}

const BaseExprNode* IRMutator::mutate(const FunctionCall* v) {
  const BaseCallNode* base = v;
  return this->mutate(base);
}

const BaseExprNode* IRMutator::mutate(const BaseCallNode* v) {
  std::vector<const BaseExprNode*> params(v->nparams());
  bool any_change = false;
  for (int i = 0; i < v->nparams(); i++) {
    const BaseExprNode* value = v->param(i);
    const BaseExprNode* value_new = value->accept_mutator(this);
    if (value != value_new) {
      any_change = true;
    }
    params[i] = std::move(value_new);
  }
  if (!any_change) {
    return v;
  }
  return v->DefaultMutator(params);
}

Stmt* IRMutator::mutate(const For* v) {
  const BaseExprNode* var = v->var();
  const BaseExprNode* start = v->start();
  const BaseExprNode* stop = v->stop();
  Stmt* body = v->body();
  LoopOptions loop_options = v->loop_options();
  const BaseExprNode* var_new_expr = var->accept_mutator(this);
  const Variable* var_new = dynamic_cast<const Variable*>(var_new_expr);
  const BaseExprNode* start_new = start->accept_mutator(this);
  const BaseExprNode* stop_new = stop->accept_mutator(this);
  Stmt* body_new = body->accept_mutator(this);
  if (!body_new) {
    return nullptr;
  }
  if (var == var_new && start == start_new &&
      stop == stop_new && body == body_new) {
    return (Stmt*)v;
  }
  return new For(var_new, start_new, stop_new, body_new, loop_options);
}

Stmt* IRMutator::mutate(const Block* v) {
  bool any_change = false;
  std::vector<Stmt*> stmts;
  for (int i = 0; i < v->nstmts(); i++) {
    Stmt* stmt = v->stmt(i);
    Stmt* stmt_new = stmt->accept_mutator(this);
    if (stmt != stmt_new) {
      any_change = true;
    }
    if (stmt_new) {
      stmts.push_back(stmt_new);
    }
  }
  if (!any_change) {
    return (Stmt*)v;
  }
  return Block::make(stmts);
}

Stmt* IRMutator::mutate(const Store* v) {
  const Variable* base_handle = v->base_handle();
  const BaseExprNode* index = v->index();
  const BaseExprNode* value = v->value();
  const BaseExprNode* mask = v->mask();
  const BaseExprNode* base_handle_expr = base_handle->accept_mutator(this);
  const Variable* base_handle_new = dynamic_cast<const Variable*>(base_handle_expr);
  const BaseExprNode* index_new = index->accept_mutator(this);
  const BaseExprNode* value_new = value->accept_mutator(this);
  const BaseExprNode* mask_new = mask->accept_mutator(this);
  if (base_handle == base_handle_new && index == index_new &&
      value == value_new && mask == mask_new) {
    return (Stmt*)v;
  }
  return new Store(base_handle_new, index_new, value_new, mask_new);
}

Stmt* IRMutator::mutate(const Allocate* v) {
  const Variable* buffer_var_old = v->buffer_var();
  const Variable* buffer_var_new =
      dynamic_cast<const Variable*>(buffer_var_old->accept_mutator(this));
  bool any_change = buffer_var_new == buffer_var_old;

  std::vector<const BaseExprNode*> dims_old = v->dims();
  std::vector<const BaseExprNode*> dims_new(dims_old.size());
  for (size_t i = 0; i < dims_old.size(); i++) {
    dims_new[i] = dims_old[i]->accept_mutator(this);
    any_change |= (dims_new[i] == dims_old[i]);
  }

  if (!any_change) {
    return (Stmt*)v;
  }

  return new Allocate(buffer_var_new, v->dtype(), dims_new);
}

Stmt* IRMutator::mutate(const Free* v) {
  const BaseExprNode* buffer_var_old = v->buffer_var();
  const Variable* buffer_var_new = dynamic_cast<const Variable*>(buffer_var_old->accept_mutator(this));
  if (buffer_var_new == buffer_var_old) {
    return (Stmt*)v;
  }

  return new Free(buffer_var_new);
}

Stmt* IRMutator::mutate(const Cond* v) {
  const BaseExprNode* cond_old = v->condition();
  Stmt* true_old = v->true_stmt();
  Stmt* false_old = v->false_stmt();

  const BaseExprNode* cond_new = cond_old->accept_mutator(this);
  Stmt* true_new = true_old ? true_old->accept_mutator(this) : true_old;
  Stmt* false_new = false_old ? false_old->accept_mutator(this) : false_old;

  if (cond_old == cond_new && true_old == true_new &&
      false_old == false_new) {
    return (Stmt*)v;
  }
  return new Cond(cond_new, true_new, false_new);
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
