#include "test/cpp/tensorexpr/test_base.h"
#include "torch/csrc/jit/tensorexpr/asmjit_codegen.h"
#include "torch/csrc/jit/tensorexpr/ir.h"

#include <gtest/gtest.h>
namespace torch {
namespace jit {

using namespace torch::jit::compiler;

void testAsmjitIntImmTest() {
  auto a = IntImm::make(2);
  ASMJITCodeGen cg;
  a.accept(&cg);
  EXPECT_EQ(cg.value(), 2);
}

void testAsmjitIntAddTest() {
  auto a = IntImm::make(2);
  auto b = IntImm::make(3);
  auto c = Add::make(a, b);
  ASMJITCodeGen cg;
  c.accept(&cg);
  EXPECT_EQ(cg.value(), 5);
}

void testAsmjitIntSubTest() {
  auto a = IntImm::make(2);
  auto b = IntImm::make(3);
  auto c = Sub::make(a, b);
  ASMJITCodeGen cg;
  c.accept(&cg);
  EXPECT_EQ(cg.value(), -1);
}

void testAsmjitIntMulTest() {
  auto a = IntImm::make(2);
  auto b = IntImm::make(3);
  auto c = Mul::make(a, b);
  ASMJITCodeGen cg;
  c.accept(&cg);
  EXPECT_EQ(cg.value(), 6);
}

void testAsmjitIntDivTest() {
  auto a = IntImm::make(6);
  auto b = IntImm::make(3);
  auto c = Div::make(a, b);
  ASMJITCodeGen cg;
  c.accept(&cg);
  EXPECT_EQ(cg.value(), 2);
}

} // namespace jit
} // namespace torch
