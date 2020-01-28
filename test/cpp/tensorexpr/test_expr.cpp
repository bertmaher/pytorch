#include "test/cpp/tensorexpr/test_base.h"

#include "torch/csrc/jit/tensorexpr/ir_printer.h"
#include "torch/csrc/jit/tensorexpr/schedule.h"
#include "torch/csrc/jit/tensorexpr/buffer.h"
#include "torch/csrc/jit/tensorexpr/eval.h"
#include "torch/csrc/jit/tensorexpr/function.h"
#include "torch/csrc/jit/tensorexpr/ir.h"
#include "torch/csrc/jit/tensorexpr/tensor.h"
#include "test/cpp/tensorexpr/padded_buffer.h"

#include <cmath>
#include <sstream>
#include <string>
#include <stdexcept>
#include <vector>

namespace torch {
namespace jit {
using namespace torch::jit::compiler;

void testExprBasicValueTest() {
  Expr a = IntImm::make(2), b = IntImm::make(3);
  Expr c = Add::make(a, b);
  SimpleIREvaluator eval(c);
  eval();
  EXPECT_EQ(eval.value().as<int>(), 5);
}

void testExprBasicValueTest02() {
  Expr a(2.0f);
  Expr b(3.0f);
  Expr c(4.0f);
  Expr d(5.0f);
  Expr f = (a + b) - (c + d);
  SimpleIREvaluator eval(f);
  eval();
  EXPECT_EQ(eval.value().as<float>(), -4.0f);
}

void testExprLetTest01() {
  Var x("x", kFloat32);
  Expr value = Expr(3.f);
  Expr body = Expr(2.f) + (x * Expr(3.f) + Expr(4.f));
  Expr result = Let::make(x, Expr(3.f), body);
  SimpleIREvaluator eval(result);
  eval();
  EXPECT_EQ(eval.value().as<float>(), 2 + (3 * 3 + 4));
}

void testExprLetTest02() {
  Var x("x", kFloat32);
  Var y("y", kFloat32);
  Expr value = Expr(3.f);
  Expr body = Expr(2.f) + (x * Expr(3.f) + Expr(4.f) * y);
  Expr e1 = Let::make(x, Expr(3.f), body);
  Expr e2 = Let::make(y, Expr(6.f), e1);
  SimpleIREvaluator eval(e2);
  eval();
  EXPECT_EQ(eval.value().as<float>(), 2 + (3 * 3 + 4 * 6));
}

static Expr test_01(const Expr& expr) {
  return expr;
}

void testExprNoLeakTest01() {
  ASSERT_EQ(RefCounted::CheckNoLiveRefCount(), true) << "leaked refcounted object before the test";
  {
    Expr r = 1;
    r = test_01(r);
  }
  ASSERT_EQ(RefCounted::CheckNoLiveRefCount(), true) << "leaked refcounted object after the test";
}

void testExprVectorAdd01() {
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

void testExprCompareSelectEQ() {
  constexpr int N = 1024;
  Buffer a(Var("A", kHandle), kInt32, {N});
  Buffer b(Var("B", kHandle), kInt32, {N});
  Buffer c(Var("C", kHandle), kInt32, {N});
  std::vector<int> a_buffer(N, 1);
  std::vector<int> b_buffer(N, 1);
  std::vector<int> c_buffer(N, 0);
  std::vector<int> c_ref(N, 0);

  auto mask = IntImm::make(1);
  Var i("i", kInt32);
  auto memcpy_expr = For::make(
      i,
      0,
      N,
      Store::make(
          c,
          i,
          CompareSelect::make(
              Load::make(a, i, mask),
              Load::make(b, i, mask),
              CompareSelectOperation::kEQ),
          mask));

  SimpleIREvaluator ir_eval(memcpy_expr, a, b, c);
  ir_eval(a_buffer, b_buffer, c_buffer);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);

  assertAllEqual(a_buffer, 1);
  assertAllEqual(b_buffer, 1);
  assertAllEqual(c_buffer, 1);
}

void testExprSubstitute01() {
  ASSERT_EQ(RefCounted::CheckNoLiveRefCount(), true) << "leaked refcounted object before the test";
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
  ASSERT_EQ(RefCounted::CheckNoLiveRefCount(), true) << "leaked refcounted object after the test";
}

void testExprMath01() {
  Expr v = sin(Expr(1.0f));

  std::ostringstream oss;
  oss << v;
  ASSERT_EQ(oss.str(), "sin(1)");

  SimpleIREvaluator eval(v);
  eval();
  float v_ref = std::sin(1.0f);
  float res = eval.value().as<float>();
  ASSERT_NEAR(res, v_ref, 1e-6);
}

void testExprUnaryMath01() {
  struct TestConfig {
    std::function<Expr(const Expr&)> func;
    std::function<float(float)> ref_func;
  };

  std::vector<TestConfig> test_configs = {
      {[](const Expr& v) { return sin(v); },
       [](float v) { return std::sin(v); }},
      {[](const Expr& v) { return sin(v); },
       [](float v) { return std::sin(v); }},
      {[](const Expr& v) { return tan(v); },
       [](float v) { return std::tan(v); }},
      {[](const Expr& v) { return asin(v); },
       [](float v) { return std::asin(v); }},
      {[](const Expr& v) { return acos(v); },
       [](float v) { return std::acos(v); }},
      {[](const Expr& v) { return atan(v); },
       [](float v) { return std::atan(v); }},
      {[](const Expr& v) { return sinh(v); },
       [](float v) { return std::sinh(v); }},
      {[](const Expr& v) { return cosh(v); },
       [](float v) { return std::cosh(v); }},
      {[](const Expr& v) { return tanh(v); },
       [](float v) { return std::tanh(v); }},
      {[](const Expr& v) { return exp(v); },
       [](float v) { return std::exp(v); }},
      {[](const Expr& v) { return fabs(v); },
       [](float v) { return std::fabs(v); }},
      {[](const Expr& v) { return log(v); },
       [](float v) { return std::log(v); }},
      {[](const Expr& v) { return log2(v); },
       [](float v) { return std::log2(v); }},
      {[](const Expr& v) { return log10(v); },
       [](float v) { return std::log10(v); }},
      {[](const Expr& v) { return erf(v); },
       [](float v) { return std::erf(v); }},
      {[](const Expr& v) { return sqrt(v); },
       [](float v) { return std::sqrt(v); }},
      {[](const Expr& v) { return rsqrt(v); },
       [](float v) { return 1.0f / std::sqrt(v); }},
      {[](const Expr& v) { return ceil(v); },
       [](float v) { return std::ceil(v); }},
      {[](const Expr& v) { return floor(v); },
       [](float v) { return std::floor(v); }},
      {[](const Expr& v) { return round(v); },
       [](float v) { return std::round(v); }},
      {[](const Expr& v) { return trunc(v); },
       [](float v) { return std::trunc(v); }},
  };

  for (const TestConfig& test_config : test_configs) {
    const float input_v = 0.8765f;
    Expr v = test_config.func(Expr(input_v));
    float v_ref = test_config.ref_func(input_v);
    SimpleIREvaluator eval(v);
    eval();
    EXPECT_NEAR(eval.value().as<float>(), v_ref, 1e-6) << "fail: " << v;
  }
}

void testExprBinaryMath01() {
  struct TestConfig {
    std::function<Expr(const Expr&, const Expr&)> func;
    std::function<float(float, float)> ref_func;
  };

  std::vector<TestConfig> test_configs = {
      {[](const Expr& v1, const Expr& v2) { return pow(v1, v2); },
       [](float v1, float v2) { return std::pow(v1, v2); }},
      {[](const Expr& v1, const Expr& v2) { return fmod(v1, v2); },
       [](float v1, float v2) { return std::fmod(v1, v2); }},
  };

  for (const TestConfig& test_config : test_configs) {
    const float v1 = 0.8765f;
    float v2 = 1.2345f;
    Expr v_expr = test_config.func(Expr(v1), Expr(v2));
    float v_ref = test_config.ref_func(v1, v2);
    SimpleIREvaluator eval(v_expr);
    eval();
    EXPECT_NEAR(eval.value().as<float>(), v_ref, 1e-6) << "fail: " << v_expr;
  }
}
} // namespace jit
} // namespace torch
