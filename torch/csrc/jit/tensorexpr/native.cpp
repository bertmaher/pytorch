#ifdef ENABLE_LLVM

#include "torch/csrc/jit/tensorexpr/native.h"
#include <torch/csrc/jit/tensorexpr/kernel.h>
#include "ATen/NativeFunctions.h"

std::unordered_map<std::string, std::pair<void*, TensorCreator>>&
getNativeFunctionRegistry() {
  static std::unordered_map<std::string, std::pair<void*, TensorCreator>> nfr_;
  return nfr_;
}

void matmul(float* a, float* b, size_t N, size_t M, size_t K, float* c) {
  for (auto i = 0; i < N * M; ++i) {
    c[i] = 0;
  }

  for (auto j = 0; j < N; ++j) {
    for (auto i = 0; i < M; ++i) {
      for (auto k = 0; k < K; ++k) {
        c[j * M + i] += a[j * K + k] * b[k * M + i];
      }
    }
  }
}

using namespace torch::jit::tensorexpr;

static RegisterNativeFunction f(
    "aten::matmul",
    &matmul,
    [](TensorExprKernel* tek, const torch::jit::Value* v) {
      return Compute(
          "aten_matmul",
          texprDims(v),
          [tek, v](const std::vector<VarHandle>& axes) -> ExprHandle {
            const torch::jit::Node* n = v->node();
            TORCH_CHECK(n->inputs().size() == 2);

            tek->addNoInline(n->inputs()[0]->unique());
            tek->addNoInline(n->inputs()[1]->unique());
            // TODO This is totally broken
            const Expr* e0 = tek->tensorOrConstant(n->inputs()[0], axes).node();
            auto t0 = tek->getTensor(n->inputs()[0]->unique())->function();
            const Expr* e1 = tek->tensorOrConstant(n->inputs()[1], axes).node();
            auto t1 = tek->getTensor(n->inputs()[1]->unique())->function();
            // N, M, K
            std::vector<const Expr*> inputs = {
                e0, e1, t0->dim(0), t1->dim(1), t0->dim(1)};
            return ExprHandle(CallExternal::make("aten::matmul", inputs));
          });
    });

#endif // ENABLE_LLVM
