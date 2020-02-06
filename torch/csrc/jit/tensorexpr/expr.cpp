#include "torch/csrc/jit/tensorexpr/expr.h"

#include "torch/csrc/jit/tensorexpr/ir.h"

namespace torch {
namespace jit {
namespace tensorexpr {

KernelArena::~KernelArena() {
  for (KernelScopedObject* p : kernel_objects_) {
    delete p;
  }
}

KernelScopedObject::KernelScopedObject() {
  KernelArena& kernel = KernelArena::GetCurrentKernelArena();
  kernel.kernel_objects_.push_back(this);
}

KernelScopedObject::~KernelScopedObject() {}

Expr Expr::operator+(const Expr& other) const {
  return Add::make(*this, other);
}

static std::vector<KernelArena*>& GetKernelArenaStack() {
  thread_local std::vector<KernelArena*> kernel_arena_stack;
  return kernel_arena_stack;
}

KernelArena& KernelArena::GetCurrentKernelArena() {
  std::vector<KernelArena*>& kernel_arena_stack = GetKernelArenaStack();
  if (kernel_arena_stack.empty()) {
    throw std::runtime_error(
        "A KernelScope must be bound before creating KernelScopedObject");
  }
  return *kernel_arena_stack.back();
}

KernelScope::KernelScope() : owning_kernel_arena_(true) {
  kernel_arena_ = new KernelArena;
  GetKernelArenaStack().push_back(kernel_arena_);
}

KernelScope::KernelScope(KernelArena& kernel_arena)
    : owning_kernel_arena_(false) {
  kernel_arena_ = &kernel_arena;
  GetKernelArenaStack().push_back(&kernel_arena);
}

KernelScope::~KernelScope() noexcept(false) {
  std::vector<KernelArena*>& kernel_arena_stack = GetKernelArenaStack();
  if (kernel_arena_ != kernel_arena_stack.back()) {
    throw std::runtime_error("Mismatch KernelScope and kernel");
  }
  if (owning_kernel_arena_) {
    delete kernel_arena_;
  }
  kernel_arena_stack.pop_back();
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

Expr expm1(const Expr& v) {
  return Intrinsics::make(kExpm1, v);
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

Expr log1p(const Expr& v) {
  return Intrinsics::make(kLog1p, v);
}

Expr erf(const Expr& v) {
  return Intrinsics::make(kErf, v);
}

Expr erfc(const Expr& v) {
  return Intrinsics::make(kErfc, v);
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

Expr frac(const Expr& v) {
  return Intrinsics::make(kFrac, v);
}

Expr lgamma(const Expr& v) {
  return Intrinsics::make(kLgamma, v);
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

Expr ifThenElse(const Expr& c, const Expr& t, const Expr& f) {
  return IfThenElse::make(c, t, f);
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
