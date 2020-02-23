#include "torch/csrc/jit/tensorexpr/expr.h"

#include "torch/csrc/jit/tensorexpr/ir.h"

namespace torch {
namespace jit {
namespace tensorexpr {

ExprHandler ExprHandler::operator+(const ExprHandler& other) const {
  return Add::make(*this, other);
}

ExprHandler ExprHandler::operator-(const ExprHandler& other) const {
  return Sub::make(*this, other);
}

ExprHandler ExprHandler::operator*(const ExprHandler& other) const {
  return Mul::make(*this, other);
}

ExprHandler ExprHandler::operator/(const ExprHandler& other) const {
  return Div::make(*this, other);
}

ExprHandler ExprHandler::operator==(const ExprHandler& other) const {
  return CompareSelect::make(*this, other, CompareSelectOperation::kEQ);
}

ExprHandler ExprHandler::operator!=(const ExprHandler& other) const {
  return CompareSelect::make(*this, other, CompareSelectOperation::kNE);
}

ExprHandler ExprHandler::operator>(const ExprHandler& other) const {
  return CompareSelect::make(*this, other, CompareSelectOperation::kGT);
}

ExprHandler ExprHandler::operator>=(const ExprHandler& other) const {
  return CompareSelect::make(*this, other, CompareSelectOperation::kGE);
}

ExprHandler ExprHandler::operator<(const ExprHandler& other) const {
  return CompareSelect::make(*this, other, CompareSelectOperation::kLT);
}

ExprHandler ExprHandler::operator<=(const ExprHandler& other) const {
  return CompareSelect::make(*this, other, CompareSelectOperation::kLE);
}

ExprHandler::ExprHandler(int v) : ExprHandler(IntImm::make(v)) {}

ExprHandler::ExprHandler(float v) : ExprHandler(FloatImm::make(v)) {}

ExprHandler sin(const ExprHandler& v) {
  return Intrinsics::make(kSin, v);
}

ExprHandler cos(const ExprHandler& v) {
  return Intrinsics::make(kCos, v);
}

ExprHandler tan(const ExprHandler& v) {
  return Intrinsics::make(kTan, v);
}

ExprHandler asin(const ExprHandler& v) {
  return Intrinsics::make(kAsin, v);
}

ExprHandler acos(const ExprHandler& v) {
  return Intrinsics::make(kAcos, v);
}

ExprHandler atan(const ExprHandler& v) {
  return Intrinsics::make(kAtan, v);
}

ExprHandler sinh(const ExprHandler& v) {
  return Intrinsics::make(kSinh, v);
}

ExprHandler cosh(const ExprHandler& v) {
  return Intrinsics::make(kCosh, v);
}

ExprHandler tanh(const ExprHandler& v) {
  return Intrinsics::make(kTanh, v);
}

ExprHandler exp(const ExprHandler& v) {
  return Intrinsics::make(kExp, v);
}

ExprHandler expm1(const ExprHandler& v) {
  return Intrinsics::make(kExpm1, v);
}

ExprHandler fabs(const ExprHandler& v) {
  return Intrinsics::make(kFabs, v);
}

ExprHandler log(const ExprHandler& v) {
  return Intrinsics::make(kLog, v);
}

ExprHandler log2(const ExprHandler& v) {
  return Intrinsics::make(kLog2, v);
}

ExprHandler log10(const ExprHandler& v) {
  return Intrinsics::make(kLog10, v);
}

ExprHandler log1p(const ExprHandler& v) {
  return Intrinsics::make(kLog1p, v);
}

ExprHandler erf(const ExprHandler& v) {
  return Intrinsics::make(kErf, v);
}

ExprHandler erfc(const ExprHandler& v) {
  return Intrinsics::make(kErfc, v);
}

ExprHandler sqrt(const ExprHandler& v) {
  return Intrinsics::make(kSqrt, v);
}

ExprHandler rsqrt(const ExprHandler& v) {
  return Intrinsics::make(kRsqrt, v);
}

ExprHandler ceil(const ExprHandler& v) {
  return Intrinsics::make(kCeil, v);
}

ExprHandler floor(const ExprHandler& v) {
  return Intrinsics::make(kFloor, v);
}

ExprHandler round(const ExprHandler& v) {
  return Intrinsics::make(kRound, v);
}

ExprHandler trunc(const ExprHandler& v) {
  return Intrinsics::make(kTrunc, v);
}

ExprHandler frac(const ExprHandler& v) {
  return Intrinsics::make(kFrac, v);
}

ExprHandler lgamma(const ExprHandler& v) {
  return Intrinsics::make(kLgamma, v);
}

ExprHandler atan2(const ExprHandler& v1, const ExprHandler& v2) {
  return Intrinsics::make(kAtan2, v1, v2);
}

ExprHandler pow(const ExprHandler& v1, const ExprHandler& v2) {
  return Intrinsics::make(kPow, v1, v2);
}

ExprHandler fmod(const ExprHandler& v1, const ExprHandler& v2) {
  return Intrinsics::make(kFmod, v1, v2);
}

ExprHandler remainder(const ExprHandler& v1, const ExprHandler& v2) {
  return Intrinsics::make(kRemainder, v1, v2);
}

ExprHandler ifThenElse(const ExprHandler& c, const ExprHandler& t, const ExprHandler& f) {
  return IfThenElse::make(c, t, f);
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
