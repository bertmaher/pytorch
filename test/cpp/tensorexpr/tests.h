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
  _(ExprNoLeakTest01)                  \
  _(ExprFuserStyle)                    \
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
  _(ExprSimple01)                      \
  _(ExprLower01)                       \
  _(ExprSimple02)                      \
  _(ScheduleBroadcastAddBuffer)        \
  _(ScheduleFunctionCall01)            \
  _(TypeTest01)                        \
  _(AsmjitIntImmTest)                  \
  _(AsmjitIntAddTest)                  \
  _(AsmjitIntSubTest)                  \
  _(AsmjitIntMulTest)                  \
  _(AsmjitIntDivTest)                  \
  _(LLVMIntImm)                        \
  _(LLVMFloatImm)                      \
  _(LLVMIntAdd)                        \
  _(LLVMIntSub)                        \
  _(LLVMIntMul)                        \
  _(LLVMIntDiv)                        \
  _(LLVMIntToFloatCast)                \
  _(LLVMFloatToIntCast)                \
  _(LLVMLetTest01)                     \
  _(LLVMLetTest02)                     \
  _(LLVMBuffer)                        \
  _(LLVMBlock)                         \
  _(LLVMLoadStore)                     \
  _(LLVMVecLoadStore)                  \
  _(LLVMMemcpy)                        \
  _(LLVMBzero)                         \
  _(LLVMElemwiseAdd)                   \
  _(LLVMElemwiseAddFloat)              \
  _(LLVMElemwiseMaxInt)                \
  _(LLVMElemwiseMinInt)                \
  _(LLVMElemwiseMaxNumFloat)           \
  _(LLVMElemwiseMaxNumNaNFloat)        \
  _(LLVMElemwiseMinNumFloat)           \
  _(LLVMElemwiseMinNumNaNFloat)        \
  _(LLVMElemwiseMaximumFloat)          \
  _(LLVMElemwiseMaximumNaNFloat)       \
  _(LLVMElemwiseMinimumFloat)          \
  _(LLVMElemwiseMinimumNaNFloat)       \
  _(LLVMCompareSelectIntEQ)            \
  _(LLVMCompareSelectFloatEQ)          \
  _(LLVMStoreFloat)                    \
  _(LLVMSimpleMath01)                  \
  _(LLVMComputeMul)                    \
  _(LLVMBroadcastAdd)                  \



#define TH_FORALL_TESTS_CUDA(_) \

#define DECLARE_TENSOREXPR_TEST(name) void test##name();
TH_FORALL_TESTS(DECLARE_TENSOREXPR_TEST)
TH_FORALL_TESTS_CUDA(DECLARE_TENSOREXPR_TEST)
#undef DECLARE_TENSOREXPR_TEST


} // namespace jit
} // namespace torch
