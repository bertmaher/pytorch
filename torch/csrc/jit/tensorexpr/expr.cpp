#include "torch/csrc/jit/tensorexpr/expr.h"

#include "torch/csrc/jit/tensorexpr/ir.h"

namespace torch {
namespace jit {
namespace tensorexpr {

Kernel::~Kernel() {
  for (KernelScopedObject* p : kernel_objects_) {
    delete p;
  }
}

KernelScopedObject::KernelScopedObject() {
  Kernel& kernel = Kernel::GetCurrentKernel();
  kernel.kernel_objects_.push_back(this);
}

KernelScopedObject::~KernelScopedObject() {}

Expr Expr::operator+(const Expr& other) const {
  return Add::make(*this, other);
}

static std::vector<Kernel*>& GetKernelStack() {
  thread_local std::vector<Kernel*> kernel_stacks;
  return kernel_stacks;
}

Kernel& Kernel::GetCurrentKernel() {
  std::vector<Kernel*>& kernel_stack = GetKernelStack();
  if (kernel_stack.empty()) {
    throw std::runtime_error(
        "A KernelScope must be bound before creating KernelScopedObject");
  }
  return *kernel_stack.back();
}

KernelScope::KernelScope() : owning_kernel_(true) {
  kernel_ = new Kernel;
  GetKernelStack().push_back(kernel_);
}

KernelScope::KernelScope(Kernel& kernel) : owning_kernel_(false) {
  kernel_ = &kernel;
  GetKernelStack().push_back(&kernel);
}

KernelScope::~KernelScope() {
  std::vector<Kernel*>& kernel_stack = GetKernelStack();
  if (kernel_ != kernel_stack.back()) {
    throw std::runtime_error("Mismatch KernelScope and kernel");
  }
  if (owning_kernel_) {
    delete kernel_;
  }
  kernel_stack.pop_back();
}

Expr Expr::operator-(const Expr& other) const {
  return Sub::make(*this, other);
}

Expr Expr::operator*(const Expr& other) const {
  return Mul::make(*this, other);
}

Expr Expr::operator/(const Expr& other) const {
  return Div::make(*this, other);
}

Expr Expr::operator==(const Expr& other) const {
  return CompareSelect::make(*this, other, CompareSelectOperation::kEQ);
}

Expr Expr::operator!=(const Expr& other) const {
  return CompareSelect::make(*this, other, CompareSelectOperation::kNE);
}

Expr Expr::operator>(const Expr& other) const {
  return CompareSelect::make(*this, other, CompareSelectOperation::kGT);
}

Expr Expr::operator>=(const Expr& other) const {
  return CompareSelect::make(*this, other, CompareSelectOperation::kGE);
}

Expr Expr::operator<(const Expr& other) const {
  return CompareSelect::make(*this, other, CompareSelectOperation::kLT);
}

Expr Expr::operator<=(const Expr& other) const {
  return CompareSelect::make(*this, other, CompareSelectOperation::kLE);
}

Expr::Expr(int v) : Expr(IntImm::make(v)) {}

Expr::Expr(float v) : Expr(FloatImm::make(v)) {}

Expr sin(const Expr& v) {
  return Intrinsics::make(kSin, v);
}

Expr cos(const Expr& v) {
  return Intrinsics::make(kCos, v);
}

Expr tan(const Expr& v) {
  return Intrinsics::make(kTan, v);
}

Expr asin(const Expr& v) {
  return Intrinsics::make(kAsin, v);
}

Expr acos(const Expr& v) {
  return Intrinsics::make(kAcos, v);
}

Expr atan(const Expr& v) {
  return Intrinsics::make(kAtan, v);
}

Expr sinh(const Expr& v) {
  return Intrinsics::make(kSinh, v);
}

Expr cosh(const Expr& v) {
  return Intrinsics::make(kCosh, v);
}

Expr tanh(const Expr& v) {
  return Intrinsics::make(kTanh, v);
}

Expr exp(const Expr& v) {
  return Intrinsics::make(kExp, v);
}

Expr fabs(const Expr& v) {
  return Intrinsics::make(kFabs, v);
}

Expr log(const Expr& v) {
  return Intrinsics::make(kLog, v);
}

Expr log2(const Expr& v) {
  return Intrinsics::make(kLog2, v);
}

Expr log10(const Expr& v) {
  return Intrinsics::make(kLog10, v);
}

Expr erf(const Expr& v) {
  return Intrinsics::make(kErf, v);
}

Expr sqrt(const Expr& v) {
  return Intrinsics::make(kSqrt, v);
}

Expr rsqrt(const Expr& v) {
  return Intrinsics::make(kRsqrt, v);
}

Expr ceil(const Expr& v) {
  return Intrinsics::make(kCeil, v);
}

Expr floor(const Expr& v) {
  return Intrinsics::make(kFloor, v);
}

Expr round(const Expr& v) {
  return Intrinsics::make(kRound, v);
}

Expr trunc(const Expr& v) {
  return Intrinsics::make(kTrunc, v);
}

Expr pow(const Expr& v1, const Expr& v2) {
  return Intrinsics::make(kPow, v1, v2);
}

Expr fmod(const Expr& v1, const Expr& v2) {
  return Intrinsics::make(kFmod, v1, v2);
}

Expr remainder(const Expr& v1, const Expr& v2) {
  return Intrinsics::make(kRemainder, v1, v2);
}
} // namespace tensorexpr
} // namespace jit
} // namespace torch
