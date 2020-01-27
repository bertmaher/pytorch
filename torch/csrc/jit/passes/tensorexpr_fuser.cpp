#include <torch/csrc/autograd/record_function.h>
#include <torch/csrc/jit/custom_operator.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/operator_options.h>
#include <torch/csrc/jit/pass_manager.h>
#include <torch/csrc/jit/passes/alias_analysis.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/tensorexpr/buffer.h>
#include <torch/csrc/jit/tensorexpr/eval.h>
#include <torch/csrc/jit/tensorexpr/llvm_codegen.h>
#include <torch/csrc/jit/tensorexpr/schedule.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

using namespace torch::jit;
using namespace torch::jit::compiler;

namespace {

const Symbol& getTensorExprSymbol() {
  static Symbol s = Symbol::fromQualString("tensorexpr::Group");
  return s;
}

value_list sortReverseTopological(
    ArrayRef<torch::jit::Value*> inputs,
    torch::jit::Block* block) {
  value_list result;
  for (auto i : inputs) {
    if (i->node()->owningBlock() == block) {
      result.push_back(i);
    }
  }
  // Sort in reverse topological order
  std::sort(
      result.begin(),
      result.end(),
      [&](torch::jit::Value* a, torch::jit::Value* b) {
        return a->node()->isAfter(b->node());
      });
  return result;
}

bool isSupported(Node* node) {
  // TODO:
  switch (node->kind()) {
  case aten::add:
  case aten::sub:
  case aten::mul:
  case aten::div:
  case aten::eq:
  case aten::ne:
  case aten::ge:
  case aten::gt:
  case aten::le:
  case aten::lt:
  case aten::log:
  case aten::log10:
  case aten::log2:
  case aten::exp:
  case aten::erf:
  case aten::cos:
  case aten::sin:
  case aten::tan:
    return true;
  default:
    return false;
  }
}

bool canHandle(Node* node, AliasDb& aliasDb) {
  if (node->kind() == prim::Constant) {
    return true;
  }
  if (node->kind() == prim::Loop) {
    return false; // TODO
  }
  return isSupported(node);
}

#define REQ(cond)                           \
  if (!(cond)) {                            \
    GRAPH_DEBUG("Failed cond " #cond "\n"); \
    return c10::nullopt;                    \
  }

c10::optional<Node*> tryMerge(
    Node* consumer,
    Node* producer,
    AliasDb& aliasDb) {
  GRAPH_DEBUG(
      "Trying producer ",
      producer->kind().toQualString(),
      " and consumer ",
      consumer->kind().toQualString(),
      ":\n");

  // Symbolic checks
  REQ(canHandle(producer, aliasDb));
  REQ(
      (canHandle(consumer, aliasDb) ||
       consumer->kind() == getTensorExprSymbol()));

  // Alias checks
  // Requirement:
  // - moveAfterTopologicallyValid(consumer, producer)
  // - One of:
  //   1) Both are in-place ops
  //   2) Consumer is in-place, producer !hasInputWriters
  //   3) Producer is in-place, consumer !hasOutputWriters
  REQ(aliasDb.moveAfterTopologicallyValid(consumer, producer));

  // 1)
  if (!(aliasDb.isMutable(consumer) && aliasDb.isMutable(producer))) {
    // 2)
    if (aliasDb.isMutable(consumer)) {
      REQ(!aliasDb.hasInputWriters(producer));
      // 3)
    } else if (aliasDb.isMutable(producer)) {
      REQ(!aliasDb.hasOutputWriters(consumer));
    }
  }

  if (!consumer->hasAttribute(attr::Subgraph) &&
      consumer->kind() != getTensorExprSymbol()) {
    consumer =
        SubgraphUtils::createSingletonSubgraph(consumer, getTensorExprSymbol());

    // createSingletonSubgraph pre-emptively folds constants into the subgraph,
    // so there's nothing more for us to do.
    if (producer->kind() == prim::Constant) {
      return consumer;
    }
  }

  if (producer->kind() == prim::Constant) {
    auto& subgraph = consumer->g(attr::Subgraph);
    Node* in_const = subgraph->createClone(
        producer, [](torch::jit::Value*) -> torch::jit::Value* {
          throw std::runtime_error("unexpected input");
        });

    subgraph->setInsertPoint(producer);
    subgraph->insertNode(in_const);
  } else {
    SubgraphUtils::mergeNodeIntoSubgraph(producer, consumer);
  }
  return consumer;
}
#undef REQ

std::pair<graph_node_list::iterator, bool> scanNode(
    Node* consumer,
    AliasDb& aliasDb,
    torch::jit::Block* block) {
  auto inputs = sortReverseTopological(consumer->inputs(), block);
  for (auto input : inputs) {
    if (auto group = tryMerge(consumer, input->node(), aliasDb)) {
      // we successfully merged, so the new group's `inputs` may have
      // changed. So rescan the new group for more merging opportunities.
      return {group.value()->reverseIterator(), true};
    }
  }
  return {++consumer->reverseIterator(), false};
}

void fuseTensorExprs(std::shared_ptr<Graph>& graph) {
#if TX_DEBUG
  std::cout << "Entering TExprFuser\n";
  std::cout << *graph;
#endif

  AliasDb aliasDb(graph);
  auto block = graph->block();

  bool any_changed = true;
  while (any_changed) {
    any_changed = false;
    for (auto it = block->nodes().rbegin(); it != block->nodes().rend();) {
      bool changed;
      std::tie(it, changed) = scanNode(*it, aliasDb, block);
      any_changed |= changed;
    }
  }

  EliminateCommonSubexpression(graph);
  EliminateDeadCode(graph);

#if TX_DEBUG
  std::cout << "Finishing TExprFuser\n";
  std::cout << *graph;
#endif
}

Dtype texprType(const c10::optional<at::ScalarType>& st) {
  switch (*st) {
    case at::ScalarType::Int:
      return kInt32;
    case at::ScalarType::Float:
      return kFloat32;
    default:
      LOG(FATAL) << "Unhandled datatype";
      return kUninitialized;
  }
}

std::vector<Expr> texprSizes(const c10::VaryingShape& shape) {
  std::vector<Expr> dims;
  for (size_t i = 0; i < *shape.size(); i++) {
    dims.push_back(IntImm::make(*shape[i]));
  }
  return dims;
}

std::vector<DimArg> texprDims(torch::jit::Value* v) {
  auto tt = v->type()->cast<TensorType>();
  auto exprDims = texprSizes(tt->sizes());
  return std::vector<DimArg>(exprDims.begin(), exprDims.end());
}

Buffer texprBuffer(const torch::jit::Value* v) {
  auto tt = v->type()->cast<TensorType>();
  return Buffer(
      v->debugName(), texprType(tt->scalarType()), texprSizes(tt->sizes()));
}

template <typename T>
int64_t bufferSize(T t) {
  int64_t size = 1;
  for (int i = 0; i < t.ndim(); i++) {
    size *= t.dim(i).template AsNode<IntImm>()->value();
  }
  return size;
}

template <typename T>
std::vector<int64_t> bufferSizes(const T& t) {
  std::vector<int64_t> sizes;
  for (int i = 0; i < t.ndim(); i++) {
    sizes.push_back(t.dim(i).template AsNode<IntImm>()->value());
  }
  return sizes;
}

template <typename T>
std::vector<Expr> computeIndicesToBroadcast(
    const std::vector<T>& output_axes,
    const std::vector<int64_t>& input_sizes) {
  TORCH_CHECK(
      output_axes.size() >= input_sizes.size(),
      "Cannot broadcast to a lower rank tensor");
  std::vector<Expr> bcast;
  auto axis_it = output_axes.rbegin();
  auto size_it = input_sizes.rbegin();
  while (size_it != input_sizes.rend()) {
    if (*size_it == 1) {
      bcast.push_back(0);
    } else {
      bcast.push_back(*axis_it);
    }
    ++axis_it;
    ++size_it;
  }
  std::reverse(bcast.begin(), bcast.end());
  return bcast;
}

struct TensorExprKernel {
  std::vector<Buffer> buffer_args;
  Tensor* tensor_output;
  std::unordered_map<int64_t, Tensor> tensors;
  std::unordered_map<int64_t, Expr> constants;
  Stmt stmt;

  Expr constant(torch::jit::Value* v) {
    if (v->node()->kind() == prim::Constant) {
      const auto val = toIValue(v).value();
      if (val.isDouble()) {
        return FloatImm::make(val.toDouble());
      } else if (val.isInt()) {
        return IntImm::make(val.toInt());
      } else {
        LOG(FATAL) << "Unhandled constant datatype";
      }
    }

    LOG(FATAL) << "Not a constant!";
    return Expr();
  }

  const Tensor& tensor(torch::jit::Value* v) {
    return tensors.at(v->unique());
  }

  template <typename T>
  Expr broadcast(const T& t, const std::vector<Var>& axes) {
    return t.call(computeIndicesToBroadcast(axes, bufferSizes(t)));
  }

  void promoteInputs(std::vector<Expr>& inputs) {
    bool any_float = std::any_of(inputs.begin(), inputs.end(),
      [](const Expr& e) { return e.dtype() == kFloat32; }
    );

    if (!any_float) return;

    for (Expr& e : inputs) {
      if (e.dtype() == kInt32) {
        e = cast<float>(e);
      }
    }
  }

  Expr demoteOutput(const Expr& e, torch::jit::Value* v) {
    auto tt = v->type()->cast<TensorType>()->scalarType();
    if (e.dtype() == kFloat32 && tt == at::ScalarType::Int) {
      return cast<int>(e);
    }

    return e;
  }

  Tensor ComputeOneOperand(const std::string& name, Node* n,
                           std::function<Expr(const Expr&)> inner_expr) {
    return Compute(
      name,
      texprDims(n->output()),
      [this, n, inner_expr](const std::vector<Var>& axes) {
        std::vector<Expr> inputs = {
          broadcast(tensor(n->inputs()[0]), axes)
        };

        promoteInputs(inputs);
        Expr compute = inner_expr(inputs[0]);
        return demoteOutput(compute, n->output());
      }
    );
  }

  Tensor ComputeTwoOperand(const std::string& name, Node* n,
                           std::function<Expr(const Expr&, const Expr&)> inner_expr) {
    return Compute(
      name,
      texprDims(n->output()),
      [this, n, inner_expr](const std::vector<Var>& axes) {
        std::vector<Expr> inputs = {
          broadcast(tensor(n->inputs()[0]), axes),
          broadcast(tensor(n->inputs()[1]), axes),
        };

        promoteInputs(inputs);
        Expr compute = inner_expr(inputs[0], inputs[1]);
        return demoteOutput(compute, n->output());
      }
    );
  }

  Tensor ComputeTwoOperandWithAlpha(const std::string& name, Node* n,
                                    std::function<Expr(const Expr&, const Expr&)> inner_expr) {
    return Compute(
      name,
      texprDims(n->output()),
      [this, n, inner_expr](const std::vector<Var>& axes) {
        std::vector<Expr> inputs = {
          broadcast(tensor(n->inputs()[0]), axes),
          broadcast(tensor(n->inputs()[1]), axes),
          constant(n->inputs()[2]),
        };

        promoteInputs(inputs);
        Expr compute = inner_expr(inputs[0], inputs[2] * inputs[1]);
        return demoteOutput(compute, n->output());
      }
    );
  }

  Tensor ComputeNode(Node* n) {
    switch (n->kind()) {
      case aten::add: {
        return ComputeTwoOperandWithAlpha("aten_add", n,
          [](const Expr& lhs, const Expr& rhs) { return lhs + rhs; }
        );
      } break;

      case aten::sub: {
        return ComputeTwoOperandWithAlpha("aten_sub", n,
          [](const Expr& lhs, const Expr& rhs) { return lhs - rhs; }
        );
      } break;

      case aten::mul: {
        return ComputeTwoOperand("aten_mul", n,
          [](const Expr& lhs, const Expr& rhs) { return lhs * rhs; }
        );
      } break;

      case aten::div: {
        return ComputeTwoOperand("aten_div", n,
          [](const Expr& lhs, const Expr& rhs) { return lhs / rhs; }
        );
      } break;

      case aten::eq: {
        return ComputeTwoOperand("aten_eq", n,
          [](const Expr& lhs, const Expr& rhs) { return lhs == rhs; }
        );
      } break;

      case aten::ne: {
        return ComputeTwoOperand("aten_ne", n,
          [](const Expr& lhs, const Expr& rhs) { return lhs != rhs; }
        );
      } break;
      case aten::ge: {
        return ComputeTwoOperand("aten_ge", n,
          [](const Expr& lhs, const Expr& rhs) { return lhs >= rhs; }
        );
      } break;

      case aten::gt: {
        return ComputeTwoOperand("aten_gt", n,
          [](const Expr& lhs, const Expr& rhs) { return lhs > rhs; }
        );
      } break;

      case aten::le: {
        return ComputeTwoOperand("aten_le", n,
          [](const Expr& lhs, const Expr& rhs) { return lhs <= rhs; }
        );
      } break;

      case aten::lt: {
        return ComputeTwoOperand("aten_lt", n,
          [](const Expr& lhs, const Expr& rhs) { return lhs < rhs; }
        );
      } break;

      case aten::log: {
        return ComputeOneOperand("aten_log", n,
          [](const Expr& a) { return log(a); }
        );
      } break;

      case aten::log10: {
        return ComputeOneOperand("aten_log10", n,
          [](const Expr& a) { return log10(a); }
        );
      } break;

      case aten::log2: {
        return ComputeOneOperand("aten_log2", n,
          [](const Expr& a) { return log2(a); }
        );
      } break;

      case aten::exp: {
        return ComputeOneOperand("aten_exp", n,
          [](const Expr& a) { return exp(a); }
        );
      } break;

      case aten::erf: {
        return ComputeOneOperand("aten_erf", n,
          [](const Expr& a) { return erf(a); }
        );
      } break;

      case aten::cos: {
        return ComputeOneOperand("aten_cos", n,
          [](const Expr& a) { return cos(a); }
        );
      } break;

      case aten::sin: {
        return ComputeOneOperand("aten_sin", n,
          [](const Expr& a) { return sin(a); }
        );
      } break;

      case aten::tan: {
        return ComputeOneOperand("aten_tan", n,
          [](const Expr& a) { return tan(a); }
        );
      } break;

      default: {
        LOG(FATAL) << "Unhandled node kind";
      }
    }
  }

  explicit TensorExprKernel(const Node* node) {
    auto subgraph = node->g(attr::Subgraph);

    // Bind inputs to buffers.
    for (auto const& input : subgraph->inputs()) {
      Buffer in_buffer = texprBuffer(input);
      tensors.emplace(
          input->unique(),
          Compute(
              "input",
              texprDims(input),
              [this, in_buffer](const std::vector<Var>& axes) {
                return broadcast(in_buffer, axes);
              }));
      buffer_args.push_back(std::move(in_buffer));
    }

    // Bind nodes to tensor compute expressions.
    for (auto const& n : subgraph->nodes()) {
      if (n->kind() == prim::Constant) {
        continue;
      }

      tensors.emplace(
        n->output()->unique(),
        ComputeNode(n)
      );
    }

    CHECK(subgraph->outputs().size() == 1ULL)
        << "Only handle single output subgraphs";
    auto const& output = subgraph->outputs()[0];
    CHECK(tensors.count(output->unique())) << "Output must be a tensor";
    tensor_output = &tensors.at(output->unique());
    torch::jit::compiler::schedule::Schedule sch({*tensor_output});
    for (auto& p : tensors) {
      auto& t = p.second;
      if (&t != tensor_output) {
        t.ComputeInline();
      }
    }
    stmt = sch.Lower();
  }

  void run(Stack& stack) {
#ifdef ENABLE_LLVM
    // Set up formal params (inputs, then outputs) for kernel.
    std::vector<Buffer*> params;
    for (auto& b : buffer_args) {
      params.push_back(&b);
    }
    Buffer outbuf(
        tensor_output->function().func_var(),
        tensor_output->dtype(),
        tensor_output->dims());
    params.push_back(&outbuf);

    // Generate code.
    LLVMCodeGen codegen(stmt, params);

    // Set up arguments (inputs, then outputs) for kernel call.
    auto inputs = last(stack, buffer_args.size());
    std::vector<void*> args;
    for (int i = 0; i < buffer_args.size(); i++) {
      args.push_back(inputs[i].toTensor().data_ptr());
    }
    at::Tensor output =
        at::empty(bufferSizes(*tensor_output), at::ScalarType::Float);
    args.push_back(output.data_ptr());

    // Call the kernel.
    codegen.value<int32_t>(args);

    // Update the stack.
    drop(stack, buffer_args.size());
    stack.insert(stack.end(), std::move(output));
#else
    SimpleIREvaluator eval(stmt);
    std::vector<std::vector<float>> backing;

    auto inputs = last(stack, buffer_args.size());
    for (size_t i = 0; i < buffer_args.size(); i++) {
      eval.bindBuffer(buffer_args[i], inputs[i].toTensor().data_ptr());
    }

    at::Tensor output =
        at::empty(bufferSizes(*tensor_output), at::ScalarType::Float);
    eval.bindBuffer(*tensor_output, output.data_ptr());

    eval.eval();
    drop(stack, buffer_args.size());
    stack.insert(stack.end(), std::move(output));
#endif
  }
};

Operation createTensorExprOp(const Node* node) {
  auto kernel = std::make_shared<TensorExprKernel>(node);
  return [kernel](Stack& stack) {
    RECORD_FUNCTION("TensorExpr", std::vector<c10::IValue>());
    kernel->run(stack);
    return 0;
  };
}

c10::OperatorOptions getAliasAnalysisOption(AliasAnalysisKind k) {
  auto options = c10::OperatorOptions();
  options.setAliasAnalysis(k);
  return options;
}

RegisterOperators TensorExprOps({
    torch::jit::Operator(
        getTensorExprSymbol(),
        createTensorExprOp,
        getAliasAnalysisOption(AliasAnalysisKind::PURE_FUNCTION)),
});

RegisterPass pass(fuseTensorExprs);

} // namespace
