#include <sstream>
#include <stdexcept>

#include <gtest/gtest.h>

#include "torch/csrc/jit/tensorexpr/ir_printer.h"
#include "torch/csrc/jit/tensorexpr/tests/test_utils.h"

using namespace torch::jit::compiler;

TEST(ExprTest, BasicValueTest) {
  Expr a = IntImm::make(2), b = IntImm::make(3);
  Expr c = Add::make(a, b);
  SimpleIREvaluator eval(c);
  eval();
  EXPECT_EQ(eval.value().as<int>(), 5);
}

TEST(ExprTest, BasicValueTest02) {
  Expr a(2.0f);
  Expr b(3.0f);
  Expr c(4.0f);
  Expr d(5.0f);
  Expr f = (a + b) - (c + d);
  SimpleIREvaluator eval(f);
  eval();
  EXPECT_EQ(eval.value().as<float>(), -4.0f);
}

TEST(ExprTest, LetTest01) {
  Var x("x", kFloat32);
  Expr value = Expr(3.f);
  Expr body = Expr(2.f) + (x * Expr(3.f) + Expr(4.f));
  Expr result = Let::make(x, Expr(3.f), body);
  SimpleIREvaluator eval(result);
  eval();
  EXPECT_EQ(eval.value().as<float>(), 2 + (3 * 3 + 4));
}

TEST(ExprTest, LetTest02) {
  Var x("x", kFloat32);
  Var y("y", kFloat32);
  Expr value = Expr(3.f);
  Expr body = Expr(2.f) + (x * Expr(3.f) + Expr(4.f) * y);
  Expr e1 = Let::make(x, Expr(3.f), body);
  Expr e2 = Let::make(y, Expr(6.f), e1);
  SimpleIREvaluator eval(2);
  eval();
  EXPECT_EQ(eval.value().as<float>(), 2 + (3 * 3 + 4 * 6));
}

TEST(ExprTest, Tensor01) {
  Tensor tensor =
      Compute("f", {{3, "x"}, {4, "y"}}, [](const Var& x, const Var& y) {
        return Expr(1.0f) + cast<float>(x) * x + cast<float>(y) * y;
      });
  std::vector<float> result;
  SimpleTensorEvaluator<float> tensor_eval;
  tensor_eval.evaluate(tensor, &result);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      float reference_v = 1 + i * i + j * j;
      int index = i * 4 + j;
      EXPECT_EQ(result[index], reference_v);
    }
  }
}

TEST(ExprTest, VectorAdd01) {
  const int kVectorSize = 8;
  const int kVectorCount = 128;
  const int kTotalSize = kVectorSize * kVectorCount;

  Buffer a_buf(Var("A", kHandle), kFloat32, {Expr(kTotalSize)});
  Buffer b_buf(Var("B", kHandle), kFloat32, {Expr(kTotalSize)});
  Buffer c_buf(Var("C", kHandle), kFloat32, {Expr(kTotalSize)});

  /*
  Build the following:
    for (int index = 0; index < kVectorCount; index++) {
      store(c_buf, ramp(index * 8, 1, 8),
            load(a_buf, ramp(index * 8, 1, 8) +
            load(b_buf, ramp(index * 8, 1, 8))))
    }
  */
  Var index = Var("index", kInt32);
  Expr load_a = Load::make(
      a_buf,
      Ramp::make(index * kVectorSize, 1, kVectorSize),
      Broadcast::make(1, kVectorSize));
  Expr load_b = Load::make(
      b_buf,
      Ramp::make(index * kVectorSize, 1, kVectorSize),
      Broadcast::make(1, kVectorSize));
  Expr value = load_a + load_b;
  Stmt store_c = Store::make(
      c_buf,
      Ramp::make(index * kVectorSize, 1, kVectorSize),
      value,
      Broadcast::make(1, kVectorSize));
  Stmt stmt = For::make(index, 0, kVectorCount, store_c);

  EXPECT_EQ(load_a.dtype(), Dtype(kFloat32, kVectorSize));
  EXPECT_EQ(load_b.dtype(), Dtype(kFloat32, kVectorSize));
  EXPECT_EQ(value.dtype(), Dtype(kFloat32, kVectorSize));

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);
  PaddedBuffer<float> c_v(kTotalSize);
  PaddedBuffer<float> c_ref(kTotalSize);
  for (int i = 0; i < kTotalSize; i++) {
    a_v(i) = i * i;
    b_v(i) = i * i * 4;
    c_ref(i) = a_v(i) + b_v(i);
  }
  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf);
  ir_eval(a_v, b_v, c_v);
  ExpectAllNear(c_v, c_ref, 1e-5);
}

TEST(ExprTest, Substitute01) {
  {
    Expr x = Variable::make("x", kFloat32);
    Expr y = Variable::make("y", kFloat32);
    Expr e = (x - 1.0f) * (x + y + 2.0f);

    Expr z = Variable::make("z", kFloat32);
    Expr e2 = Substitute(&e, {{x, z + 1.0f}});
    Expr e2_ref = ((z + 1.0f) - 1.0f) * ((z + 1.0f) + y + 2.0f);
    std::ostringstream oss;
    oss << e2;
    std::string e2_str = oss.str();

    oss.str("");
    oss << e2_ref;
    std::string e2_ref_str = oss.str();
    ASSERT_EQ(e2_str, e2_ref_str);
  }
  // TODO: move this to a test fixture and enable for all tests.
  ASSERT_EQ(RefCounted::CheckNoLiveRefCount(), true);
}
