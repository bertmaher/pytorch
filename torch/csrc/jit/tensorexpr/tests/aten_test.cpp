#include <sstream>
#include <stdexcept>

#include <gtest/gtest.h>

#include "torch/csrc/jit/tensorexpr/ir_printer.h"
#include "torch/csrc/jit/tensorexpr/tests/test_utils.h"

using namespace torch::jit::compiler;

TEST(ATenTest, _cast_Float) {
  const int kTotalSize = 128;
  Buffer a_buf(Var("A", kHandle), kInt32, {Expr(kTotalSize)});
  Buffer b_buf(Var("B", kHandle), kFloat32, {Expr(kTotalSize)});

  Var index = Var("index", kInt32);
  Expr load_a = Load::make(
      a_buf,
      index,
      1);
  Expr to_float = Cast::make(kFloat32, load_a);
  Stmt store_b = Store::make(
      b_buf,
      index,
      to_float,
      1);
  Stmt stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
      a_v(i) = i;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf);
  ir_eval(a_v, b_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i) << "index: " << i;
    EXPECT_EQ(b_v(i), static_cast<float>(i)) << "index: " << i;
  }
}
