#include "torch/csrc/jit/compiler/include/ir.h"
#include "torch/csrc/jit/compiler/include/llvm_codegen.h"

#include <gtest/gtest.h>

using namespace torch::jit::compiler;

TEST(ExprTest, IntImmTest) {
  auto a = IntImm::make(2);
  LLVMCodeGen cg;
  a.accept(&cg);
  EXPECT_EQ(cg.value(), 2);
}

TEST(ExprTest, IntAddTest) {
  auto a = IntImm::make(2);
  auto b = IntImm::make(3);
  auto c = Add::make(a, b);
  LLVMCodeGen cg;
  c.accept(&cg);
  EXPECT_EQ(cg.value(), 5);
}

TEST(ExprTest, IntSubTest) {
  auto a = IntImm::make(2);
  auto b = IntImm::make(3);
  auto c = Sub::make(a, b);
  LLVMCodeGen cg;
  c.accept(&cg);
  EXPECT_EQ(cg.value(), -1);
}

TEST(ExprTest, IntMulTest) {
  auto a = IntImm::make(2);
  auto b = IntImm::make(3);
  auto c = Mul::make(a, b);
  LLVMCodeGen cg;
  c.accept(&cg);
  EXPECT_EQ(cg.value(), 6);
}

TEST(ExprTest, IntDivTest) {
  auto a = IntImm::make(6);
  auto b = IntImm::make(3);
  auto c = Div::make(a, b);
  LLVMCodeGen cg;
  c.accept(&cg);
  EXPECT_EQ(cg.value(), 2);
}
