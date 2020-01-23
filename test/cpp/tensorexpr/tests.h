#pragma once

/**
 * See README.md for instructions on how to add a new test.
 */
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <c10/macros/Export.h>

namespace torch {
namespace jit {
#define TH_FORALL_TESTS(_)             \
  _(ExprBasicValue)                    \
  _(ExprBasicValue02)                  \
  _(ExprLet01)                         \
  _(DISABLED_ExprLet02)                \
  _(ExprTensor01)                      \
  _(ExprVectorAdd01)                   \
  _(ExprCompareSelectEQ)               \
  _(ExprSubstitute01)                  \
  _(ExprMath01)                        \
  _(ExprUnaryMath01)                   \
  _(ExprBinaryMath01)                  \
  _(IRPrinterBasicValueTest)           \
  _(IRPrinterBasicValueTest02)         \
  _(IRPrinterLetTest01)                \
  _(IRPrinterLetTest02)                \
  _(IRPrinterCastTest)                 \

#define TH_FORALL_TESTS_CUDA(_) \

#define DECLARE_TENSOREXPR_TEST(name) void test##name();
TH_FORALL_TESTS(DECLARE_TENSOREXPR_TEST)
TH_FORALL_TESTS_CUDA(DECLARE_TENSOREXPR_TEST)
#undef DECLARE_TENSOREXPR_TEST


} // namespace jit
} // namespace torch
