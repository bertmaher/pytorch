#include "test/cpp/tensorexpr/test_base.h"
#include "torch/csrc/jit/tensorexpr/asmjit_codegen.h"
#include "torch/csrc/jit/tensorexpr/ir.h"

#include <gtest/gtest.h>
namespace torch {
namespace jit {

using namespace torch::jit::compiler;

// XXX: ASMJit codegen is broken due to some linker error, disabling the tests
// until it's fixed.

// #define ASMJIT_TESTS_ENABLED

void testAsmjitIntImmTest() {
#ifdef ASMJIT_TESTS_ENABLED
  auto a = IntImm::make(2);
  ASMJITCodeGen cg;
  a.accept(&cg);
  EXPECT_EQ(cg.value(), 2);
#endif
}

void testAsmjitIntAddTest() {
#ifdef ASMJIT_TESTS_ENABLED
  auto a = IntImm::make(2);
  auto b = IntImm::make(3);
  auto c = Add::make(a, b);
  ASMJITCodeGen cg;
  c.accept(&cg);
  EXPECT_EQ(cg.value(), 5);
#endif
}

void testAsmjitIntSubTest() {
#ifdef ASMJIT_TESTS_ENABLED
  auto a = IntImm::make(2);
  auto b = IntImm::make(3);
  auto c = Sub::make(a, b);
  ASMJITCodeGen cg;
  c.accept(&cg);
  EXPECT_EQ(cg.value(), -1);
#endif
}

void testAsmjitIntMulTest() {
#ifdef ASMJIT_TESTS_ENABLED
  auto a = IntImm::make(2);
  auto b = IntImm::make(3);
  auto c = Mul::make(a, b);
  ASMJITCodeGen cg;
  c.accept(&cg);
  EXPECT_EQ(cg.value(), 6);
#endif
}

void testAsmjitIntDivTest() {
#ifdef ASMJIT_TESTS_ENABLED
  auto a = IntImm::make(6);
  auto b = IntImm::make(3);
  auto c = Div::make(a, b);
  ASMJITCodeGen cg;
  c.accept(&cg);
  EXPECT_EQ(cg.value(), 2);
#endif
}

} // namespace jit
} // namespace torch
