#include <test/cpp/tensorexpr/tests.h>

#include <gtest/gtest.h>

namespace torch {
namespace jit {

#define TENSOREXPR_GTEST(name) \
  TEST(TensorExprTest, name) { \
    test##name();       \
  }
TH_FORALL_TESTS(TENSOREXPR_GTEST)
#undef TENSOREXPR_TEST

#define TENSOREXPR_GTEST_CUDA(name)   \
  TEST(TensorExprTest, name##_CUDA) { \
    test##name();              \
  }
TH_FORALL_TESTS_CUDA(TENSOREXPR_GTEST_CUDA)
#undef TENSOREXPR_TEST_CUDA

} // namespace jit
} // namespace torch
