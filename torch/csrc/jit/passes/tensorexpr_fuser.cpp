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
#include <torch/csrc/jit/tensorexpr/schedule.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

using namespace torch::jit;
using namespace torch::jit::tensorexpr;

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
    case aten::min:
    case aten::max:
    case aten::clamp:
    case aten::log10:
    case aten::log:
    case aten::log2:
    case aten::exp:
    case aten::erf:
    case aten::cos:
    case aten::sin:
    case aten::tan:
    case aten::acos:
    case aten::asin:
    case aten::atan:
    case aten::cosh:
    case aten::sinh:
    case aten::tanh:
    case aten::sqrt:
    case aten::rsqrt:
    case aten::abs:
    case aten::floor:
    case aten::ceil:
    case aten::round:
    case aten::trunc:
    case aten::remainder:
    case prim::ConstantChunk:
    case aten::cat:
    case prim::ListConstruct:
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

  // Only handle complete tensor types
  for (torch::jit::Value* output : consumer->outputs()) {
    REQ(output->isCompleteTensor());
  }

  // Only fuse within a block
  REQ(consumer->owningBlock() == producer->owningBlock());

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
    // Don't initiate a fusion group from prim::ListConstruct
    REQ(consumer->kind() != prim::ListConstruct);

    // Don't initiate a fusion group just for a constant operand
    REQ(producer->kind() != prim::Constant);

    consumer =
        SubgraphUtils::createSingletonSubgraph(consumer, getTensorExprSymbol());
  }

  if (producer->kind() == aten::cat) {
    REQ(producer->inputs()[0]->node()->kind() == prim::ListConstruct);
    REQ(producer->inputs()[0]->uses().size() == 1);
    REQ(producer->inputs()[1]->node()->kind() == prim::Constant);
    Node* listconstruct = producer->inputs()[0]->node();
    Node* constant = producer->inputs()[1]->node();
    SubgraphUtils::mergeNodeIntoSubgraph(producer, consumer);
    SubgraphUtils::mergeNodeIntoSubgraph(constant, consumer);
    SubgraphUtils::mergeNodeIntoSubgraph(listconstruct, consumer);
  } else {
    if (consumer->kind() == aten::cat) {
      REQ(consumer->inputs()[0]->node()->kind() == prim::ListConstruct);
      REQ(consumer->inputs()[0]->uses().size() == 1);
      REQ(consumer->inputs()[1]->node()->kind() == prim::Constant);
    }
    SubgraphUtils::mergeNodeIntoSubgraph(producer, consumer);
  }

  return consumer;
}
#undef REQ

std::pair<graph_node_list::iterator, bool> scanNode(
    Node* consumer,
    AliasDb& aliasDb) {
  auto inputs = sortReverseTopological(consumer->inputs(), consumer->owningBlock());
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

  std::vector<std::pair<graph_node_list_iterator,
                        graph_node_list_iterator>> worklist;
  std::unordered_set<torch::jit::Block*> visited_blocks;

  bool any_changed = true;
  while (any_changed) {
    any_changed = false;
    worklist.push_back({block->nodes().rbegin(), block->nodes().rend()});

    while (worklist.size()) {
      auto& it = worklist.back().first;
      auto end = worklist.back().second;

      if (it->blocks().size()) {
        Node* n = *it;
        ++it;
        for (auto b : n->blocks()) {
          if (!visited_blocks.count(b)) {
            worklist.push_back({b->nodes().rbegin(), b->nodes().rend()});
            visited_blocks.insert(b);
          }
        }
      } else {
        bool changed;
        std::tie(it, changed) = scanNode(*it, aliasDb);
        any_changed |= changed;
      }

      if (it == end) {
        worklist.pop_back();
      }
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

at::ScalarType tensorType(const Tensor& t) {
  auto const& stype = t.dtype().scalar_type();
  if (stype == kInt32) {
    return at::ScalarType::Int;
  } else if (stype == kFloat32) {
    return at::ScalarType::Float;
  }
  LOG(FATAL) << "Unhandled datatype";
  return at::ScalarType::Float;
}

std::vector<Expr> texprSizes(const c10::VaryingShape& shape) {
  std::vector<Expr> dims;
  for (size_t i = 0; i < *shape.size(); i++) {
    dims.push_back(IntImm::make(*shape[i]));
  }
  return dims;
}

std::vector<DimArg> texprDims(torch::jit::Value* v) {
  CHECK(v->type()->kind() == TypeKind::TensorType);
  auto tt = v->type()->cast<TensorType>();
  std::vector<DimArg> dimArgs;
  int i = 0;
  for (auto const& s : texprSizes(tt->sizes())) {
    dimArgs.push_back({s, "i" + std::to_string(i++)});
  }
  return dimArgs;
}

Buffer texprBuffer(const torch::jit::Value* v) {
  CHECK(v->type()->kind() == TypeKind::TensorType);
  auto tt = v->type()->cast<TensorType>();
  return Buffer(
      "t" + v->debugName(),
      texprType(tt->scalarType()),
      texprSizes(tt->sizes()));
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

class TensorExprKernel {
 private:
  enum BackendType {
    kUninitialized,
    kSimpleIREval,
    kLLVMCodeGen,
    kCudaCodeGen,
  };
  std::vector<CodeGen::BufferArg> buffer_args_;
  std::vector<Tensor> tensor_outputs_;
  std::unordered_map<int64_t, Tensor> tensors_;
  std::unordered_map<int64_t, Var> scalars_;
  std::unique_ptr<CodeGen> codegen_;
  KernelArena kernel_arena_;
  BackendType backend_type_ = BackendType::kUninitialized;
  at::Device device_ = at::kCPU;

 private:
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
    CHECK(scalars_.count(v->unique())) << "Couldn't find scalar value";
    return scalars_.at(v->unique());
  }

  template <typename T, typename T1>
  Expr broadcast(const T& t, const std::vector<T1>& axes) {
    return t.call(computeIndicesToBroadcast(axes, bufferSizes(t)));
  }

  template <typename T, typename T1>
  Expr chunk(
      const T& t,
      size_t chunk_idx,
      size_t dim,
      size_t chunks,
      const std::vector<T1>& axes) {
    auto sizes = bufferSizes(t);
    size_t step = sizes[dim] / chunks;

    std::vector<Expr> indices;
    for (size_t i = 0; i < axes.size(); ++i) {
      if (i == dim) {
        indices.push_back(axes[i] + IntImm::make(chunk_idx * step));
      } else {
        indices.push_back(axes[i]);
      }
    }

    return t.call(indices);
  }

  void promoteInputs(std::vector<Expr>& inputs) {
    bool any_float =
        std::any_of(inputs.begin(), inputs.end(), [](const Expr& e) {
          return e.dtype() == kFloat32;
        });

    if (!any_float)
      return;

    for (Expr& e : inputs) {
      if (e.dtype() == kInt32) {
        e = cast<float>(e);
      }
    }
  }

  Expr demoteOutput(const Expr& e, torch::jit::Value* v) {
    CHECK(v->type()->kind() == TypeKind::TensorType);
    auto tt = v->type()->cast<TensorType>()->scalarType();
    if (e.dtype() == kFloat32 && tt == at::ScalarType::Int) {
      return cast<int>(e);
    }

    return e;
  }

  template <typename T>
  Expr tensorOrConstant(torch::jit::Value* v, const std::vector<T>& axes) {
    auto ti = tensors_.find(v->unique());
    if (ti != tensors_.end()) {
      return broadcast(ti->second, axes);
    }
    return constant(v);
  }

  Tensor ComputeOneOperand(
      const std::string& name,
      torch::jit::Value* v,
      std::function<Expr(const Expr&)> inner_expr) {
    return Compute(
        name,
        texprDims(v),
        [this, v, inner_expr](const std::vector<Var>& axes) {
          Node* n = v->node();
          std::vector<Expr> inputs = {tensorOrConstant(n->inputs()[0], axes)};

          promoteInputs(inputs);
          Expr compute = inner_expr(inputs[0]);
          return demoteOutput(compute, n->output());
        });
  }

  Tensor ComputeTwoOperand(
      const std::string& name,
      torch::jit::Value* v,
      std::function<Expr(const Expr&, const Expr&)> inner_expr) {
    return Compute(
        name,
        texprDims(v),
        [this, v, inner_expr](const std::vector<Var>& axes) {
          Node* n = v->node();
          std::vector<Expr> inputs = {
              tensorOrConstant(n->inputs()[0], axes),
              tensorOrConstant(n->inputs()[1], axes),
          };

          promoteInputs(inputs);
          Expr compute = inner_expr(inputs[0], inputs[1]);
          return demoteOutput(compute, n->output());
        });
  }

  Tensor ComputeTwoOperandWithAlpha(
      const std::string& name,
      torch::jit::Value* v,
      std::function<Expr(const Expr&, const Expr&)> inner_expr) {
    return Compute(
        name,
        texprDims(v),
        [this, v, inner_expr](const std::vector<Var>& axes) {
          Node* n = v->node();
          std::vector<Expr> inputs = {
              tensorOrConstant(n->inputs()[0], axes),
              tensorOrConstant(n->inputs()[1], axes),
              tensorOrConstant(n->inputs()[2], axes),
          };

          promoteInputs(inputs);
          Expr compute = inner_expr(inputs[0], inputs[2] * inputs[1]);
          return demoteOutput(compute, n->output());
        });
  }

  Tensor ComputeThreeOperand(
      const std::string& name,
      torch::jit::Value* v,
      std::function<Expr(const Expr&, const Expr&, const Expr&)> inner_expr) {
    return Compute(
        name,
        texprDims(v),
        [this, v, inner_expr](const std::vector<Var>& axes) {
          Node* n = v->node();
          std::vector<Expr> inputs = {
              tensorOrConstant(n->inputs()[0], axes),
              tensorOrConstant(n->inputs()[1], axes),
              tensorOrConstant(n->inputs()[2], axes),
          };

          promoteInputs(inputs);
          Expr compute = inner_expr(inputs[0], inputs[1], inputs[2]);
          return demoteOutput(compute, n->output());
        });
  }

  Tensor ComputeValue(torch::jit::Value* v) {
    switch (v->node()->kind()) {
      case aten::add: {
        return ComputeTwoOperandWithAlpha(
            "aten_add", v, [](const Expr& lhs, const Expr& rhs) {
              return lhs + rhs;
            });
      } break;

      case aten::sub: {
        return ComputeTwoOperandWithAlpha(
            "aten_sub", v, [](const Expr& lhs, const Expr& rhs) {
              return lhs - rhs;
            });
      } break;

      case aten::mul: {
        return ComputeTwoOperand(
            "aten_mul", v, [](const Expr& lhs, const Expr& rhs) {
              return lhs * rhs;
            });
      } break;

      case aten::div: {
        return ComputeTwoOperand(
            "aten_div", v, [](const Expr& lhs, const Expr& rhs) {
              return lhs / rhs;
            });
      } break;

      case aten::eq: {
        return ComputeTwoOperand(
            "aten_eq", v, [](const Expr& lhs, const Expr& rhs) {
              return lhs == rhs;
            });
      } break;

      case aten::ne: {
        return ComputeTwoOperand(
            "aten_ne", v, [](const Expr& lhs, const Expr& rhs) {
              return lhs != rhs;
            });
      } break;
      case aten::ge: {
        return ComputeTwoOperand(
            "aten_ge", v, [](const Expr& lhs, const Expr& rhs) {
              return lhs >= rhs;
            });
      } break;

      case aten::gt: {
        return ComputeTwoOperand(
            "aten_gt", v, [](const Expr& lhs, const Expr& rhs) {
              return lhs > rhs;
            });
      } break;

      case aten::le: {
        return ComputeTwoOperand(
            "aten_le", v, [](const Expr& lhs, const Expr& rhs) {
              return lhs <= rhs;
            });
      } break;

      case aten::lt: {
        return ComputeTwoOperand(
            "aten_lt", v, [](const Expr& lhs, const Expr& rhs) {
              return lhs < rhs;
            });
      } break;

      case aten::min: {
        return ComputeTwoOperand(
            "aten_min", v, [](const Expr& lhs, const Expr& rhs) {
              return Min::make(lhs, rhs, false);
            });
      } break;

      case aten::max: {
        return ComputeTwoOperand(
            "aten_max", v, [](const Expr& lhs, const Expr& rhs) {
              return Max::make(lhs, rhs, false);
            });
      } break;

      case aten::clamp: {
        return ComputeThreeOperand(
            "aten_max",
            v,
            [](const Expr& in, const Expr& min, const Expr& max) {
              return Max::make(Min::make(in, max, false), min, false);
            });
      } break;

      case aten::log: {
        return ComputeOneOperand(
            "aten_log", v, [](const Expr& a) { return log(a); });
      } break;

      case aten::log10: {
        return ComputeOneOperand(
            "aten_log10", v, [](const Expr& a) { return log10(a); });
      } break;

      case aten::log2: {
        return ComputeOneOperand(
            "aten_log2", v, [](const Expr& a) { return log2(a); });
      } break;

      case aten::exp: {
        return ComputeOneOperand(
            "aten_exp", v, [](const Expr& a) { return exp(a); });
      } break;

      case aten::erf: {
        return ComputeOneOperand(
            "aten_erf", v, [](const Expr& a) { return erf(a); });
      } break;

      case aten::cos: {
        return ComputeOneOperand(
            "aten_cos", v, [](const Expr& a) { return cos(a); });
      } break;

      case aten::sin: {
        return ComputeOneOperand(
            "aten_sin", v, [](const Expr& a) { return sin(a); });
      } break;

      case aten::tan: {
        return ComputeOneOperand(
            "aten_tan", v, [](const Expr& a) { return tan(a); });
      } break;

      case aten::pow: {
        return ComputeTwoOperand(
            "aten_pow", v, [](const Expr& lhs, const Expr& rhs) {
              return pow(lhs, rhs);
            });
      } break;

      case aten::fmod: {
        return ComputeTwoOperand(
            "aten_fmod", v, [](const Expr& lhs, const Expr& rhs) {
              return fmod(lhs, rhs);
            });
      } break;

      case aten::remainder: {
        return ComputeTwoOperand(
            "aten_remainder", v, [](const Expr& lhs, const Expr& rhs) {
              return remainder(lhs, rhs);
            });

      } break;

      case aten::acos: {
        return ComputeOneOperand(
            "aten_acos", v, [](const Expr& a) { return acos(a); });
      } break;

      case aten::asin: {
        return ComputeOneOperand(
            "aten_asin", v, [](const Expr& a) { return asin(a); });
      } break;

      case aten::cosh: {
        return ComputeOneOperand(
            "aten_cosh", v, [](const Expr& a) { return cosh(a); });
      } break;

      case aten::sinh: {
        return ComputeOneOperand(
            "aten_sinh", v, [](const Expr& a) { return sinh(a); });
      } break;

      case aten::atan: {
        return ComputeOneOperand(
            "aten_atan", v, [](const Expr& a) { return atan(a); });
      } break;

      case aten::tanh: {
        return ComputeOneOperand(
            "aten_tanh", v, [](const Expr& a) {
              //return (Expr(-.67436811832e-5f)+(Expr(.2468149110712040f)+(Expr(.583691066395175e-1f)+Expr(.3357335044280075e-1f)*a)*a)*a)/(Expr(.2464845986383725f)+(Expr(.609347197060491e-1f)+(Expr(.1086202599228572f)+Expr(.2874707922475963e-1f)*a)*a)*a);
              return tanh(a);
              });
      } break;

      case aten::sqrt: {
        return ComputeOneOperand(
            "aten_sqrt", v, [](const Expr& a) { return sqrt(a); });
      } break;

      case aten::rsqrt: {
        return ComputeOneOperand(
            "aten_rsqrt", v, [](const Expr& a) { return rsqrt(a); });
      } break;

      case aten::abs: {
        return ComputeOneOperand(
            "aten_abs", v, [](const Expr& a) { return fabs(a); });
      } break;

      case aten::ceil: {
        return ComputeOneOperand(
            "aten_ceil", v, [](const Expr& a) { return ceil(a); });
      } break;

      case aten::floor: {
        return ComputeOneOperand(
            "aten_floor", v, [](const Expr& a) { return floor(a); });
      } break;

      case aten::round: {
        return ComputeOneOperand(
            "aten_round", v, [](const Expr& a) { return round(a); });
      } break;

      case aten::trunc: {
        return ComputeOneOperand(
            "aten_trunc", v, [](const Expr& a) { return trunc(a); });
      } break;

      case prim::ConstantChunk: {
        return Compute(
            "prim_constantchunk",
            texprDims(v),
            [this, v](const std::vector<Var>& axes) {
              Node* n = v->node();
              int64_t dim = n->i(attr::dim);
              int64_t chunks = n->i(attr::chunks);
              return chunk(
                  tensors_.at(n->inputs()[0]->unique()),
                  v->offset(),
                  dim,
                  chunks,
                  axes);
            });
      }

      case aten::cat: {
        return Compute(
            "aten_cat", texprDims(v), [this, v](const std::vector<Var>& axes) {
              Node* n = v->node();
              auto inputs = n->inputs()[0]->node()->inputs();
              size_t dim = n->inputs()[1]->node()->i(attr::value);

              std::vector<Expr> new_axes(axes.begin(), axes.end());
              Expr load = tensorOrConstant(inputs[0], new_axes);
              size_t offset =
                  bufferSizes(tensors_.at(inputs[0]->unique()))[dim];
              new_axes[dim] = new_axes[dim] - IntImm::make(offset);

              for (int ii = 1; ii < inputs.size(); ++ii) {
                load = ifThenElse(
                    CompareSelect::make(axes[dim], IntImm::make(offset), kLT),
                    load,
                    tensorOrConstant(inputs[ii], new_axes));
                offset += bufferSizes(tensors_.at(inputs[ii]->unique()))[dim];
                new_axes[dim] = new_axes[dim] - IntImm::make(offset);
              }

              return load;
            });
      }

      default: {
        LOG(FATAL) << "Unhandled node kind";
      }
    }
  }

  void LowerToBackend(BackendType backend_type) {
    std::vector<Tensor> tensor_outputs(tensor_outputs_);

    if (backend_type == BackendType::kCudaCodeGen) {
      for (int i = 0; i < tensor_outputs_.size(); i++) {
        const Tensor& tensor = tensor_outputs_[i];
        Expr total_count = tensor.dim(0);
        for (int i = 1; i < tensor.ndim(); i++) {
          total_count = total_count * tensor.dim(i);
        }
        // Flatten the index for GPU kernels.
        // TODO: move this to fusing axis when it is ready.
        Tensor new_out = Compute(
            tensor.function().func_var().name_hint() + "_flat",
            {total_count},
            [tensor](const Var& index) -> Expr {
              std::vector<Expr> dims;
              Expr value = index;
              for (int i = tensor.ndim() - 1; i >= 0; i--) {
                Expr idx = value;
                if (i > 0) {
                  idx = Mod::make(value, tensor.dim(i));
                }
                dims.push_back(idx);
                value = value / tensor.dim(i);
              }
              std::reverse(dims.begin(), dims.end());
              return tensor.call(dims);
            });
        tensor_outputs[i] = new_out;
      }
    }

    torch::jit::tensorexpr::schedule::Schedule sch(tensor_outputs);

    // Compute non-output tensors_ inline
    for (auto& p : tensors_) {
      p.second.ComputeInline();
    }
    if (backend_type == kCudaCodeGen) {
      for (int i = 0; i < tensor_outputs_.size(); i++) {
        tensor_outputs_[i].ComputeInline();
        Tensor tensor = tensor_outputs[i];
        Var index = tensor.arg(0);
        Var outer;
        Var inner;
        tensor.SplitWithMask(index, 1024, true, &outer, &inner);
        tensor.GPUExecConfig({outer}, {inner});
      }
    }

    Stmt stmt = sch.Lower();

    // Set up formal params (inputs, then outputs) for kernel.
    std::vector<CodeGen::BufferArg> params(
        buffer_args_.begin(), buffer_args_.end());
    for (auto& o : tensor_outputs) {
      params.push_back(o);
    }

    // Generate code.
    std::string codegen_name;
    switch (backend_type_) {
      case kCudaCodeGen:
        codegen_name = "cuda_codegen";
        break;
      case kLLVMCodeGen:
        codegen_name = "llvm_codegen";
        break;
      case kSimpleIREval:
        codegen_name = "simple_ir_eval";
        break;
      default:
        throw std::runtime_error(
            "invalid backend type: " +
            std::to_string(static_cast<int>(backend_type_)));
    }
    codegen_ = CreateCodeGen(codegen_name, stmt, params);
  }

  void PickAndCheckBackendType(const at::ArrayRef<IValue>& inputs) {
    at::Device device = [&inputs]() {
      for (auto const& input : inputs) {
        if (input.isTensor()) {
          return input.toTensor().device();
        }
      }
      throw std::runtime_error("No tensor inputs");
    }();
    BackendType backend_type = BackendType::kUninitialized;
    if (device.type() == at::kCUDA) {
      backend_type = kCudaCodeGen;
    } else if (device.type() == at::kCPU) {
#ifdef ENABLE_LLVM
      backend_type = kLLVMCodeGen;
#else
      backend_type = kSimpleIREval;
      ;
#endif
    } else {
      throw std::runtime_error("Invalid device type");
    }

    if (backend_type_ == kUninitialized) {
      backend_type_ = backend_type;
      device_ = device;
      LowerToBackend(backend_type);
    } else if (backend_type_ != backend_type) {
      // TODO: if we have to support muliptole backends with the same subgraph,
      // we need to add kernel caching.
      throw std::runtime_error(
          "Inconsistent backend_type: " + std::to_string(backend_type_) +
          " vs " + std::to_string(backend_type));
    }
  }

  void CodeGenRun(const std::vector<CodeGen::CallArg>& run_args) {
    switch (backend_type_) {
      case kSimpleIREval:
      case kLLVMCodeGen:
      case kCudaCodeGen:
        codegen_->call(run_args);
        break;
      default:
        throw std::runtime_error(
            "Invalid backend type: " + std::to_string(backend_type_));
    }
  }

  void bindInput(torch::jit::Value* input) {
    auto const& t = input->type();
    switch (t->kind()) {
      case TypeKind::TensorType: {
        Buffer in_buffer = texprBuffer(input);
        tensors_.emplace(
            input->unique(),
            Compute(
                "input",
                texprDims(input),
                [this, in_buffer](const std::vector<Var>& axes) {
                  return broadcast(in_buffer, axes);
                }));
        buffer_args_.push_back(std::move(in_buffer));
        break;
      }
      case TypeKind::FloatType: {
        Var v("v" + input->debugName(), kFloat32);
        buffer_args_.push_back(v);
        scalars_.emplace(input->unique(), v);
        break;
      }
      case TypeKind::IntType: {
        Var v("v" + input->debugName(), kInt32);
        buffer_args_.push_back(v);
        scalars_.emplace(input->unique(), v);
        break;
      }
      default: {
        LOG(FATAL) << "Unhandled input type: " << *t;
        break;
      }
    }
  }

 public:
  explicit TensorExprKernel(const Node* node) {
    KernelScope kernel_scope(kernel_arena_);
    auto subgraph = node->g(attr::Subgraph);

    // Bind inputs to buffers.
    for (auto const& input : subgraph->inputs()) {
      bindInput(input);
    }

    // Bind nodes to tensor compute expressions.
    for (auto const& n : subgraph->nodes()) {
      if (n->kind() == prim::Constant || n->kind() == prim::ListConstruct) {
        continue;
      } else {
        for (torch::jit::Value* output : n->outputs()) {
          if (output->hasUses()) {
            tensors_.emplace(output->unique(), ComputeValue(output));
          }
        }
      }
    }

    // Move output operands from `tensors_` to `tensor_outputs_`
    for (const auto& output : subgraph->outputs()) {
      CHECK(tensors_.count(output->unique())) << "Output must be a tensor";
      tensor_outputs_.emplace_back(tensors_.at(output->unique()));
      tensors_.erase(output->unique());
    }
  }

  void run(Stack& stack) {
    KernelScope kernel_scope(kernel_arena_);
    // Set up arguments (inputs, then outputs) for kernel call.
    auto inputs = last(stack, buffer_args_.size());
    PickAndCheckBackendType(inputs);

    std::vector<CodeGen::CallArg> run_args;
    for (int i = 0; i < buffer_args_.size(); i++) {
      if (buffer_args_[i].isVar()) {
        auto const& dtype = buffer_args_[i].dtype();
        if (dtype == kInt32) {
          run_args.push_back((int32_t)inputs[i].toInt());
        } else if (dtype == kFloat32) {
          run_args.push_back((float)inputs[i].toDouble());
        } else {
          LOG(FATAL) << "Unhandled dtype";
        }
      } else {
        run_args.push_back(inputs[i].toTensor().data_ptr());
      }
    }
    std::vector<at::Tensor> outputs;
    for (auto& o : tensor_outputs_) {
      outputs.push_back(at::empty(
          bufferSizes(o), c10::TensorOptions(tensorType(o)).device(device_)));
      run_args.push_back(outputs.back().data_ptr());
    }

    // Call the kernel.
    CodeGenRun(run_args);

    // Update the stack.
    drop(stack, buffer_args_.size());
    for (auto& o : outputs) {
      push_one(stack, std::move(o));
    }
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
