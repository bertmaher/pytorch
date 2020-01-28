#include "test/cpp/tensorexpr/test_base.h"
#include <sstream>
#include <stdexcept>

#include <cmath>

#include "torch/csrc/jit/tensorexpr/buffer.h"
#include "torch/csrc/jit/tensorexpr/cuda_codegen.h"
#include "torch/csrc/jit/tensorexpr/schedule.h"
#include "torch/csrc/jit/tensorexpr/tensor.h"
#include "test/cpp/tensorexpr/padded_buffer.h"

namespace torch {
namespace jit {
using namespace torch::jit::compiler;
using namespace torch::jit::compiler::schedule;

void testCudaTestVectorAdd01() {
  const int block_count = 1024;
  const int block_size = 256;
  const int num_iter = 12;
  Buffer a_buf("a", kFloat32, {num_iter, block_count, block_size});
  Buffer b_buf("b", kFloat32, {num_iter, block_count, block_size});
  Tensor c = Compute(
      "c",
      {
          {num_iter, "n"},
          {block_size, "b_id"},
          {num_iter, "t_id"},
      },
      [&](const Var& n, const Var& b_id, const Var& t_id) {
        return a_buf(n, b_id, t_id) + b_buf(n, b_id, t_id);
      });
  Schedule sch({c});
  const Var& b_id = c.arg(1);
  const Var& t_id = c.arg(2);
  c.GPUExecConfig({b_id}, {t_id});
  // XXXQQQ: lower into: For(..., attrs={'threadIdx.x'})
  Stmt stmt = sch.Lower();
  CudaCodeGen cuda_cg(stmt, c, a_buf, b_buf);
  const int N = block_count * block_size * num_iter;
  PaddedBuffer<float> a_v(N);
  PaddedBuffer<float> b_v(N);
  PaddedBuffer<float> c_v(N);
  PaddedBuffer<float> c_ref(N);
  for (int i = 0; i < N; i++) {
    a_v(i) = i;
    b_v(i) = i * i;
    c_ref(i) = a_v(i) + b_v(i);
  }

  cuda_cg(c_v, a_v, b_v);

#if 0
  ExpectAllNear(c_v, c_ref, 1e-5);
#endif
}
} // namespace jit
} // namespace torch
