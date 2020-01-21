#pragma once

/**
 * See README.md for instructions on how to add a new test.
 */
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <c10/macros/Export.h>

namespace torch {
namespace jit {
#define TH_FORALL_TESTS(_)             \
  _(True)

#define TH_FORALL_TESTS_CUDA(_) \

#define DECLARE_TENSOREXPR_TEST(name) void test##name();
TH_FORALL_TESTS(DECLARE_TENSOREXPR_TEST)
TH_FORALL_TESTS_CUDA(DECLARE_TENSOREXPR_TEST)
#undef DECLARE_TENSOREXPR_TEST


} // namespace jit
} // namespace torch
