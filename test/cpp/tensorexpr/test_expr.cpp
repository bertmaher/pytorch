#include "test/cpp/tensorexpr/test_base.h"

#include "torch/csrc/jit/tensorexpr/ir_printer.h"
#include "torch/csrc/jit/tensorexpr/tests/test_utils.h"

#include <cmath>
#include <sstream>
#include <string>

namespace torch {
namespace jit {

using namespace compiler;

void testBasicValue() {
  Expr a = IntImm::make(2), b = IntImm::make(3);
  Expr c = Add::make(a, b);
  SimpleIREvaluator eval(c);
  eval();
  EXPECT_EQ(eval.value().as<int>(), 5);
}

} // namespace jit
} // namespace torch
