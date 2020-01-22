#pragma once

/**
 * See README.md for instructions on how to add a new test.
 */
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <c10/macros/Export.h>

namespace torch {
namespace jit {
#define TH_FORALL_TESTS(_)             \
  _(BasicValue)                        \
  _(BasicValue02)                      \
  _(Let01)                             \
  _(DISABLED_Let02)                    \
  _(Tensor01)                          \
  _(VectorAdd01)                       \
  _(CompareSelectEQ)                   \
  _(Substitute01)                      \
  _(Math01)                            \
  _(UnaryMath01)                       \
  _(BinaryMath01)                      \

#define TH_FORALL_TESTS_CUDA(_) \

#define DECLARE_TENSOREXPR_TEST(name) void test##name();
TH_FORALL_TESTS(DECLARE_TENSOREXPR_TEST)
TH_FORALL_TESTS_CUDA(DECLARE_TENSOREXPR_TEST)
#undef DECLARE_TENSOREXPR_TEST


} // namespace jit
} // namespace torch
