#ifdef ENABLE_LLVM
#include "test/cpp/tensorexpr/test_base.h"

#include "torch/csrc/jit/tensorexpr/ir.h"
#include "torch/csrc/jit/tensorexpr/ir_printer.h"
#include "torch/csrc/jit/tensorexpr/llvm_codegen.h"
#include "torch/csrc/jit/tensorexpr/schedule.h"
#include "torch/csrc/jit/tensorexpr/tensor.h"
#include "torch/csrc/jit/tensorexpr/tests/test_utils.h"

#include <numeric>

namespace torch {
namespace jit {

using namespace torch::jit::compiler;
using namespace torch::jit::compiler::schedule;

void testLLVMIntImm() {
  auto a = IntImm::make(2);
  LLVMCodeGen cg;
  a.accept(&cg);
  EXPECT_EQ(cg.value<int>(), 2);
}

void testLLVMFloatImm() {
  auto a = FloatImm::make(1.0);
  LLVMCodeGen cg({}, kFloat32);
  a.accept(&cg);
  EXPECT_EQ(cg.value<float>(), 1.0);
}

void testLLVMIntAdd() {
  auto a = IntImm::make(2);
  auto b = IntImm::make(3);
  auto c = Add::make(a, b);
  LLVMCodeGen cg;
  c.accept(&cg);
  EXPECT_EQ(cg.value<int>(), 5);
}

void testLLVMIntSub() {
  auto a = IntImm::make(2);
  auto b = IntImm::make(3);
  auto c = Sub::make(a, b);
  LLVMCodeGen cg;
  c.accept(&cg);
  EXPECT_EQ(cg.value<int>(), -1);
}

void testLLVMIntMul() {
  auto a = IntImm::make(2);
  auto b = IntImm::make(3);
  auto c = Mul::make(a, b);
  LLVMCodeGen cg;
  c.accept(&cg);
  EXPECT_EQ(cg.value<int>(), 6);
}

void testLLVMIntDiv() {
  auto a = IntImm::make(6);
  auto b = IntImm::make(3);
  auto c = Div::make(a, b);
  LLVMCodeGen cg;
  c.accept(&cg);
  EXPECT_EQ(cg.value<int>(), 2);
}

void testLLVMIntToFloatCast() {
  auto a = IntImm::make(2);
  auto b = Cast::make(kFloat32, a);
  LLVMCodeGen cg({}, kFloat32);
  b.accept(&cg);
  EXPECT_EQ(cg.value<float>(), 2.0);
}

void testLLVMFloatToIntCast() {
  auto a = FloatImm::make(2.0);
  auto b = Cast::make(kInt32, a);
  LLVMCodeGen cg;
  b.accept(&cg);
  EXPECT_EQ(cg.value<int>(), 2);
}

void testLLVMLetTest01() {
  Var x("x", kFloat32);
  Expr value = Expr(3.f);
  Expr body = Expr(2.f) + (x * Expr(3.f) + Expr(4.f));
  Expr result = Let::make(x, Expr(3.f), body);
  LLVMCodeGen cg({}, kFloat32);
  result.accept(&cg);
  EXPECT_EQ(cg.value<float>(), 2.f + (3.f * 3.f + 4.f));
}

void testLLVMLetTest02() {
  Var x("x", kFloat32);
  Var y("y", kFloat32);
  Expr value = Expr(3.f);
  Expr body = Expr(2.f) + (x * Expr(3.f) + Expr(4.f) * y);
  Expr e1 = Let::make(x, Expr(3.f), body);
  Expr e2 = Let::make(y, Expr(6.f), e1);
  LLVMCodeGen cg({}, kFloat32);
  e2.accept(&cg);
  EXPECT_EQ(cg.value<float>(), 2.f + (3.f * 3.f + 4.f * 6.f));
}

void testLLVMBuffer() {
  Buffer a(Var("A", kHandle), kFloat32, {32});
  LLVMCodeGen cg({&a});
  std::vector<int32_t> v(5);
  std::vector<void*> args({v.data()});
  auto rv = IntImm::make(0);
  rv.accept(&cg);
  EXPECT_EQ(cg.value<int>(args), 0);
}

void testLLVMBlock() {
  Buffer a(Var("A", kHandle), kInt32, {32});
  LLVMCodeGen cg({&a});
  std::vector<int32_t> v = {1, 2};
  std::vector<void*> args({v.data()});

  auto block = Block::make({
      Store::make(a, IntImm::make(0), IntImm::make(3), IntImm::make(1)),
      Store::make(a, IntImm::make(1), IntImm::make(4), IntImm::make(1)),
      Store::make(a, IntImm::make(0), IntImm::make(4), IntImm::make(1)),
  });

  block.accept(&cg);
  EXPECT_EQ(cg.value<int>(args), 0);
  EXPECT_EQ(v[0], 4);
  EXPECT_EQ(v[1], 4);
}

void testLLVMLoadStore() {
  Buffer a(Var("A", kHandle), kInt32, {1});
  Buffer b(Var("B", kHandle), kInt32, {1});
  std::vector<int32_t> a_buffer = {42};
  std::vector<int32_t> b_buffer = {-11};

  LLVMCodeGen cg({&a, &b});
  auto store = Store::make(
      b,
      IntImm::make(0),
      Load::make(a, IntImm::make(0), IntImm::make(1)),
      IntImm::make(1));
  store.accept(&cg);
  std::vector<void*> args({a_buffer.data(), b_buffer.data()});
  EXPECT_EQ(cg.value<int>(args), 0);
  EXPECT_EQ(a_buffer[0], 42);
  EXPECT_EQ(b_buffer[0], 42);
}

void testLLVMVecLoadStore() {
  Buffer a(Var("A", kHandle), kInt32, {1});
  Buffer b(Var("B", kHandle), kInt32, {1});
  std::vector<int32_t> a_buffer = {1, 1, 1, 1};
  std::vector<int32_t> b_buffer = {2, 2, 2, 2};

  LLVMCodeGen cg({&a, &b});
  auto store = Store::make(
      b,
      Ramp::make(0, 1, 4),
      Load::make(a, Ramp::make(0, 1, 4), Broadcast::make(IntImm::make(1), 4)),
      Broadcast::make(IntImm::make(1), 4));
  store.accept(&cg);
  std::vector<void*> args({a_buffer.data(), b_buffer.data()});
  EXPECT_EQ(cg.value<int>(args), 0);
  EXPECT_EQ(a_buffer[0], 1);
  EXPECT_EQ(a_buffer[1], 1);
  EXPECT_EQ(a_buffer[2], 1);
  EXPECT_EQ(a_buffer[3], 1);
  EXPECT_EQ(b_buffer[0], 1);
  EXPECT_EQ(b_buffer[1], 1);
  EXPECT_EQ(b_buffer[2], 1);
  EXPECT_EQ(b_buffer[3], 1);
}

void testLLVMMemcpy() {
  constexpr int N = 32;
  Buffer a(Var("A", kHandle), kInt32, {N});
  Buffer b(Var("B", kHandle), kInt32, {N});
  std::vector<int32_t> a_buffer(N, 42);
  std::vector<int32_t> b_buffer(N, 0);

  auto mask = IntImm::make(1);
  Var i("i", kInt32);
  auto memcpy_expr =
      For::make(i, 0, N, Store::make(b, i, Load::make(a, i, mask), mask));

  LLVMCodeGen cg({&a, &b});
  memcpy_expr.accept(&cg);

  std::vector<void*> args({a_buffer.data(), b_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  assertAllEqual(a_buffer, 42);
  assertAllEqual(b_buffer, 42);
}

void testLLVMBzero() {
  constexpr int N = 32;
  Buffer b(Var("B", kHandle), kInt32, {N});
  std::vector<int32_t> b_buffer(N, 11);

  auto mask = IntImm::make(1);
  Var i("i", kInt32);
  auto memcpy_expr =
      For::make(i, 0, N, Store::make(b, i, IntImm::make(0), mask));

  LLVMCodeGen cg({&b});
  memcpy_expr.accept(&cg);

  std::vector<void*> args({b_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(b_buffer.size(), N);
  assertAllEqual(b_buffer, 0);
}

void testLLVMElemwiseAdd() {
  constexpr int N = 1024;
  Buffer a(Var("A", kHandle), kInt32, {N});
  Buffer b(Var("B", kHandle), kInt32, {N});
  Buffer c(Var("C", kHandle), kInt32, {N});
  std::vector<int32_t> a_buffer(N, 41);
  std::vector<int32_t> b_buffer(N, 1);
  std::vector<int32_t> c_buffer(N, 1);

  auto mask = IntImm::make(1);
  Var i("i", kInt32);
  auto memcpy_expr = For::make(
      i,
      0,
      N,
      Store::make(
          c,
          i,
          Add::make(Load::make(a, i, mask), Load::make(b, i, mask)),
          mask));

  LLVMCodeGen cg({&a, &b, &c});
  memcpy_expr.accept(&cg);

  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);
  assertAllEqual(a_buffer, 41);
  assertAllEqual(b_buffer, 1);
  assertAllEqual(c_buffer, 42);
}

void testLLVMElemwiseAddFloat() {
  constexpr int N = 1024;
  Buffer a(Var("A", kHandle), kFloat32, {N});
  Buffer b(Var("B", kHandle), kFloat32, {N});
  Buffer c(Var("C", kHandle), kFloat32, {N});
  std::vector<float> a_buffer(N, 41);
  std::vector<float> b_buffer(N, 1);
  std::vector<float> c_buffer(N, 1);

  auto mask = IntImm::make(1);
  Var i("i", kInt32);
  auto memcpy_expr = For::make(
      i,
      0,
      N,
      Store::make(c, i, Load::make(a, i, mask) + Load::make(b, i, mask), mask));

  LLVMCodeGen cg({&a, &b, &c});
  memcpy_expr.accept(&cg);

  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);
  assertAllEqual(a_buffer, 41.0f);
  assertAllEqual(b_buffer, 1.0f);
  assertAllEqual(c_buffer, 42.0f);
}

void testLLVMElemwiseMaxInt() {
  constexpr int N = 1024;
  Buffer a(Var("A", kHandle), kInt32, {N});
  Buffer b(Var("B", kHandle), kInt32, {N});
  Buffer c(Var("C", kHandle), kInt32, {N});
  std::vector<int> a_buffer(N, 41);
  std::vector<int> b_buffer(N, 1);
  std::vector<int> c_buffer(N, 1);

  auto mask = IntImm::make(1);
  Var i("i", kInt32);
  auto memcpy_expr = For::make(
      i,
      0,
      N,
      Store::make(
          c,
          i,
          Max::make(Load::make(a, i, mask), Load::make(b, i, mask), false),
          mask));

  LLVMCodeGen cg({&a, &b, &c});
  memcpy_expr.accept(&cg);

  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);
  assertAllEqual(a_buffer, 41);
  assertAllEqual(b_buffer, 1);
  assertAllEqual(c_buffer, 41);
}

void testLLVMElemwiseMinInt() {
  constexpr int N = 1024;
  Buffer a(Var("A", kHandle), kInt32, {N});
  Buffer b(Var("B", kHandle), kInt32, {N});
  Buffer c(Var("C", kHandle), kInt32, {N});
  std::vector<int> a_buffer(N, 41);
  std::vector<int> b_buffer(N, 1);
  std::vector<int> c_buffer(N, 1);

  auto mask = IntImm::make(1);
  Var i("i", kInt32);
  auto memcpy_expr = For::make(
      i,
      0,
      N,
      Store::make(
          c,
          i,
          Min::make(Load::make(a, i, mask), Load::make(b, i, mask), false),
          mask));

  LLVMCodeGen cg({&a, &b, &c});
  memcpy_expr.accept(&cg);

  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);
  assertAllEqual(a_buffer, 41);
  assertAllEqual(b_buffer, 1);
  assertAllEqual(c_buffer, 1);
}

void testLLVMElemwiseMaxNumFloat() {
  constexpr int N = 1024;
  Buffer a(Var("A", kHandle), kFloat32, {N});
  Buffer b(Var("B", kHandle), kFloat32, {N});
  Buffer c(Var("C", kHandle), kFloat32, {N});
  std::vector<float> a_buffer(N, 41);
  std::vector<float> b_buffer(N, 1);
  std::vector<float> c_buffer(N, 1);

  auto mask = IntImm::make(1);
  Var i("i", kInt32);
  auto memcpy_expr = For::make(
      i,
      0,
      N,
      Store::make(
          c,
          i,
          Max::make(Load::make(a, i, mask), Load::make(b, i, mask), false),
          mask));

  LLVMCodeGen cg({&a, &b, &c});
  memcpy_expr.accept(&cg);

  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);
  assertAllEqual(a_buffer, 41.0f);
  assertAllEqual(b_buffer, 1.0f);
  assertAllEqual(c_buffer, 41.0f);
}

void testLLVMElemwiseMaxNumNaNFloat() {
  constexpr int N = 1024;
  Buffer a(Var("A", kHandle), kFloat32, {N});
  Buffer b(Var("B", kHandle), kFloat32, {N});
  Buffer c(Var("C", kHandle), kFloat32, {N});
  std::vector<float> a_buffer(N, NAN);
  std::vector<float> b_buffer(N, 1);
  std::vector<float> c_buffer(N, 1);

  auto mask = IntImm::make(1);
  Var i("i", kInt32);
  auto memcpy_expr = For::make(
      i,
      0,
      N,
      Store::make(
          c,
          i,
          Max::make(Load::make(a, i, mask), Load::make(b, i, mask), false),
          mask));

  LLVMCodeGen cg({&a, &b, &c});
  memcpy_expr.accept(&cg);

  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);
  assertAllEqual(b_buffer, 1.0f);
  assertAllEqual(c_buffer, 1.0f);
}

void testLLVMElemwiseMinNumFloat() {
  constexpr int N = 1024;
  Buffer a(Var("A", kHandle), kFloat32, {N});
  Buffer b(Var("B", kHandle), kFloat32, {N});
  Buffer c(Var("C", kHandle), kFloat32, {N});
  std::vector<float> a_buffer(N, 41);
  std::vector<float> b_buffer(N, 1);
  std::vector<float> c_buffer(N, 1);

  auto mask = IntImm::make(1);
  Var i("i", kInt32);
  auto memcpy_expr = For::make(
      i,
      0,
      N,
      Store::make(
          c,
          i,
          Min::make(Load::make(a, i, mask), Load::make(b, i, mask), false),
          mask));

  LLVMCodeGen cg({&a, &b, &c});
  memcpy_expr.accept(&cg);

  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);
  assertAllEqual(a_buffer, 41.0f);
  assertAllEqual(b_buffer, 1.0f);
  assertAllEqual(c_buffer, 1.0f);
}

void testLLVMElemwiseMinNumNaNFloat() {
  constexpr int N = 1024;
  Buffer a(Var("A", kHandle), kFloat32, {N});
  Buffer b(Var("B", kHandle), kFloat32, {N});
  Buffer c(Var("C", kHandle), kFloat32, {N});
  std::vector<float> a_buffer(N, NAN);
  std::vector<float> b_buffer(N, 1);
  std::vector<float> c_buffer(N, 1);

  auto mask = IntImm::make(1);
  Var i("i", kInt32);
  auto memcpy_expr = For::make(
      i,
      0,
      N,
      Store::make(
          c,
          i,
          Min::make(Load::make(a, i, mask), Load::make(b, i, mask), false),
          mask));

  LLVMCodeGen cg({&a, &b, &c});
  memcpy_expr.accept(&cg);

  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);
  assertAllEqual(b_buffer, 1.0f);
  assertAllEqual(c_buffer, 1.0f);
}

#if 1 // LLVM doesn't currently have implementations for maximum/minimum on x86
void testLLVMElemwiseMaximumFloat() {
  constexpr int N = 1024;
  Buffer a(Var("A", kHandle), kFloat32, {N});
  Buffer b(Var("B", kHandle), kFloat32, {N});
  Buffer c(Var("C", kHandle), kFloat32, {N});
  std::vector<float> a_buffer(N, 41);
  std::vector<float> b_buffer(N, 1);
  std::vector<float> c_buffer(N, 1);

  auto mask = IntImm::make(1);
  Var i("i", kInt32);
  auto memcpy_expr = For::make(
      i,
      0,
      N,
      Store::make(
          c,
          i,
          Max::make(Load::make(a, i, mask), Load::make(b, i, mask), true),
          mask));

  LLVMCodeGen cg({&a, &b, &c});
  memcpy_expr.accept(&cg);

  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);
  assertAllEqual(a_buffer, 41.0f);
  assertAllEqual(b_buffer, 1.0f);
  assertAllEqual(c_buffer, 41.0f);
}

void testLLVMElemwiseMaximumNaNFloat() {
  constexpr int N = 1024;
  Buffer a(Var("A", kHandle), kFloat32, {N});
  Buffer b(Var("B", kHandle), kFloat32, {N});
  Buffer c(Var("C", kHandle), kFloat32, {N});
  std::vector<float> a_buffer(N, NAN);
  std::vector<float> b_buffer(N, 1);
  std::vector<float> c_buffer(N, 1);

  auto mask = IntImm::make(1);
  Var i("i", kInt32);
  auto memcpy_expr = For::make(
      i,
      0,
      N,
      Store::make(
          c,
          i,
          Max::make(Load::make(a, i, mask), Load::make(b, i, mask), true),
          mask));

  LLVMCodeGen cg({&a, &b, &c});
  memcpy_expr.accept(&cg);

  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);
  for (int i = 0; i < N; ++i) {
    ASSERT_TRUE(std::isnan(a_buffer[i]));
    ASSERT_TRUE(std::isnan(c_buffer[i]));
  }
}

void testLLVMElemwiseMinimumFloat() {
  constexpr int N = 1024;
  Buffer a(Var("A", kHandle), kFloat32, {N});
  Buffer b(Var("B", kHandle), kFloat32, {N});
  Buffer c(Var("C", kHandle), kFloat32, {N});
  std::vector<float> a_buffer(N, 41);
  std::vector<float> b_buffer(N, 1);
  std::vector<float> c_buffer(N, 1);

  auto mask = IntImm::make(1);
  Var i("i", kInt32);
  auto memcpy_expr = For::make(
      i,
      0,
      N,
      Store::make(
          c,
          i,
          Min::make(Load::make(a, i, mask), Load::make(b, i, mask), true),
          mask));

  LLVMCodeGen cg({&a, &b, &c});
  memcpy_expr.accept(&cg);

  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);
  assertAllEqual(a_buffer, 41.0f);
  assertAllEqual(b_buffer, 1.0f);
  assertAllEqual(c_buffer, 1.0f);
}

void testLLVMElemwiseMinimumNaNFloat() {
  constexpr int N = 1024;
  Buffer a(Var("A", kHandle), kFloat32, {N});
  Buffer b(Var("B", kHandle), kFloat32, {N});
  Buffer c(Var("C", kHandle), kFloat32, {N});
  std::vector<float> a_buffer(N, NAN);
  std::vector<float> b_buffer(N, 1);
  std::vector<float> c_buffer(N, 1);

  auto mask = IntImm::make(1);
  Var i("i", kInt32);
  auto memcpy_expr = For::make(
      i,
      0,
      N,
      Store::make(
          c,
          i,
          Min::make(Load::make(a, i, mask), Load::make(b, i, mask), true),
          mask));

  LLVMCodeGen cg({&a, &b, &c});
  memcpy_expr.accept(&cg);

  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);
  for (int i = 0; i < N; ++i) {
    ASSERT_TRUE(std::isnan(a_buffer[i]));
    ASSERT_TRUE(std::isnan(c_buffer[i]));
  }
}
#endif

void testLLVMCompareSelectIntEQ() {
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

  LLVMCodeGen cg({&a, &b, &c});
  memcpy_expr.accept(&cg);

  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);

  assertAllEqual(a_buffer, 1);
  assertAllEqual(b_buffer, 1);
  assertAllEqual(c_buffer, 1);
}

void testLLVMCompareSelectFloatEQ() {
  constexpr int N = 1024;
  Buffer a(Var("A", kHandle), kFloat32, {N});
  Buffer b(Var("B", kHandle), kFloat32, {N});
  Buffer c(Var("C", kHandle), kInt32, {N});
  std::vector<float> a_buffer(N, 1.0f);
  std::vector<float> b_buffer(N, 1.0f);
  std::vector<int> c_buffer(N, 0);

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

  LLVMCodeGen cg({&a, &b, &c});
  memcpy_expr.accept(&cg);

  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);

  assertAllEqual(a_buffer, 1.0f);
  assertAllEqual(b_buffer, 1.0f);
  assertAllEqual(c_buffer, 1);
}

void testLLVMStoreFloat() {
  Buffer result(Var("result", kHandle), kFloat32, {1});
  std::vector<float> result_buffer = {0.0f};
  auto expr = Store::make(
      result, IntImm::make(0), FloatImm::make(3.14f), IntImm::make(1));
  LLVMCodeGen cg({&result});
  expr.accept(&cg);
  std::vector<void*> args({result_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);
  EXPECT_EQ(result_buffer[0], 3.14f);
}

void testLLVMSimpleMath01() {
  const int N = 1024;
  Tensor tensor = Compute(
      "f", {{N, "i"}}, [](const Var& i) { return cast<float>(i * i + 1); });
  Schedule sch = Schedule::make({tensor});
  Stmt stmt = sch.Lower();
  Buffer f_buf(tensor.function().func_var(), kFloat32, {N});
  LLVMCodeGen cg({&f_buf});
  stmt.accept(&cg);

  PaddedBuffer<float> f_v(N, "f_v");
  std::vector<void*> args({f_v.data()});
  int value = cg.value<int>(args);
  ASSERT_EQ(value, 0);
  PaddedBuffer<float> f_ref(N, "f_ref");
  for (int i = 0; i < N; i++) {
    f_ref(i) = i * i + 1;
  }
  ExpectAllNear(f_v, f_ref, 1e-5);
}

void testLLVMComputeMul() {
  const int N = 1024;
  Buffer a(Var("a", kHandle), kFloat32, {N});
  Buffer b(Var("b", kHandle), kFloat32, {N});
  Tensor c = Compute("c", {{N, "i"}}, [&](const Var& i) {
    return Load::make(a, i, 1) * Load::make(b, i, 1);
  });

  Buffer c_buf(c.function().func_var(), kFloat32, {N});
  Schedule sch = Schedule::make({c});
  Stmt s = sch.Lower();

  LLVMCodeGen cg({&a, &b, &c_buf});
  s.accept(&cg);

  std::vector<float> a_vec(N, 21.0f);
  std::vector<float> b_vec(N, 2.0f);
  std::vector<float> c_vec(N, 0.0f);
  std::vector<void*> args({a_vec.data(), b_vec.data(), c_vec.data()});
  ASSERT_EQ(cg.value<int>(args), 0);
  assertAllEqual(c_vec, 42.0f);
}

void testLLVMBroadcastAdd() {
  const int M = 32;
  const int N = 1024;
  Buffer a(Var("a", kHandle), kFloat32, {M, N});
  Buffer b(Var("b", kHandle), kFloat32, {N});
  Tensor c =
      Compute("c", {{M, "i"}, {N, "j"}}, [&](const Var& i, const Var& j) {
        Expr mask(1);
        return Load::make(a, i * N + j, mask) + Load::make(b, j, mask);
      });

  Buffer c_buf(c.function().func_var(), kFloat32, {M, N});
  Schedule sch = Schedule::make({c});
  Stmt s = sch.Lower();

  LLVMCodeGen cg({&a, &b, &c_buf});
  s.accept(&cg);

  std::vector<float> av(M * N);
  std::iota(av.begin(), av.end(), 0);
  std::vector<float> bv(N);
  std::iota(bv.begin(), bv.end(), 0);
  std::vector<float> cv(M * N, 0);
  std::vector<void*> args({av.data(), bv.data(), cv.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      ASSERT_EQ(cv[i * N + j], av[i * N + j] + bv[j]);
    }
  }
}
} // namespace jit
} // namespace torch

#endif // ENABLE_LLVM
