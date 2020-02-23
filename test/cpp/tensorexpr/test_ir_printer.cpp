#include "test/cpp/tensorexpr/test_base.h"
#include <stdexcept>

#include "torch/csrc/jit/tensorexpr/expr.h"
#include "torch/csrc/jit/tensorexpr/ir.h"
#include "torch/csrc/jit/tensorexpr/ir_printer.h"

#include <sstream>
namespace torch {
namespace jit {

using namespace torch::jit::tensorexpr;

void testIRPrinterBasicValueTest() {
  KernelScope kernel_scope;
  ExprHandler a = IntImm::make(2), b = IntImm::make(3);
  ExprHandler c = Add::make(a, b);

  std::stringstream ss;
  ss << c;
  EXPECT_EQ(ss.str(), "(2 + 3)");
}

void testIRPrinterBasicValueTest02() {
  KernelScope kernel_scope;
  ExprHandler a(2.0f);
  ExprHandler b(3.0f);
  ExprHandler c(4.0f);
  ExprHandler d(5.0f);
  ExprHandler f = (a + b) - (c + d);

  std::stringstream ss;
  ss << f;
  EXPECT_EQ(ss.str(), "((2.f + 3.f) - (4.f + 5.f))");
}

void testIRPrinterLetTest01() {
  KernelScope kernel_scope;
  VarHandler x("x", kFloat32);
  ExprHandler value = ExprHandler(3.f);
  ExprHandler body = ExprHandler(2.f) + (x * ExprHandler(3.f) + ExprHandler(4.f));
  ExprHandler result = Let::make(x, ExprHandler(3.f), body);

  std::stringstream ss;
  ss << result;
  EXPECT_EQ(ss.str(), "(let x = 3.f in (2.f + ((x * 3.f) + 4.f)))");
}

void testIRPrinterLetTest02() {
  KernelScope kernel_scope;
  VarHandler x("x", kFloat32);
  VarHandler y("y", kFloat32);
  ExprHandler value = ExprHandler(3.f);
  ExprHandler body = ExprHandler(2.f) + (x * ExprHandler(3.f) + ExprHandler(4.f) * y);
  ExprHandler e1 = Let::make(x, ExprHandler(3.f), body);
  ExprHandler e2 = Let::make(y, ExprHandler(6.f), e1);

  std::stringstream ss;
  ss << e2;
  EXPECT_EQ(
      ss.str(), "(let y = 6.f in (let x = 3.f in (2.f + ((x * 3.f) + (4.f * y)))))");
}

void testIRPrinterCastTest() {
  KernelScope kernel_scope;
  VarHandler x("x", kFloat32);
  VarHandler y("y", kFloat32);
  ExprHandler value = ExprHandler(3.f);
  ExprHandler body = ExprHandler(2.f) + (x * ExprHandler(3.f) + ExprHandler(4.f) * y);
  ExprHandler e1 = Let::make(x, Cast::make(kInt32, ExprHandler(3.f)), body);
  ExprHandler e2 = Let::make(y, ExprHandler(6.f), e1);

  std::stringstream ss;
  ss << e2;
  EXPECT_EQ(
      ss.str(),
      "(let y = 6.f in (let x = int32(3.f) in (2.f + ((x * 3.f) + (4.f * y)))))");
}
} // namespace jit
} // namespace torch
