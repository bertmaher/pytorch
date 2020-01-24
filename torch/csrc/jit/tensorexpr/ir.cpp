#include "torch/csrc/jit/tensorexpr/ir.h"

#include "torch/csrc/jit/tensorexpr/buffer.h"

namespace torch {
namespace jit {
namespace compiler {

static Dtype ChooseDtype(const Dtype& buffer_dtype, const Dtype& index_dtype) {
  return Dtype(buffer_dtype, index_dtype.lanes());
}

Load::Load(const Buffer& buffer, const Expr& index, const Expr& mask)
    : Load(
          ChooseDtype(buffer.dtype(), index.dtype()),
          buffer.data(),
          index,
          mask) {}

Load::Load(
    Dtype dtype,
    const Var& base_handle,
    const Expr& index,
    const Expr& mask)
    : ExprNodeBase(dtype),
      base_handle_(base_handle),
      index_(index),
      mask_(mask) {
  CHECK_EQ(base_handle_.dtype(), kHandle);
  CHECK_EQ(index.dtype().lanes(), mask.dtype().lanes());
  CHECK_EQ(index.dtype().scalar_type(), kInt32);
}

Store::Store(
    const Buffer& buffer,
    const Expr& index,
    const Expr& value,
    const Expr& mask)
    : Store(buffer.data(), index, value, mask) {
  CHECK_EQ(buffer.dtype().scalar_type(), value.dtype().scalar_type());
  CHECK_EQ(buffer.dtype().scalar_type(), value.dtype().scalar_type());
}

Dtype Intrinsics::IntrinsicsDtype(IntrinsicsOp op_type, Dtype dt1) {
  // TODO: check the op_type and make a real decision
  return dt1;
}

Dtype Intrinsics::IntrinsicsDtype(IntrinsicsOp op_type, Dtype dt1, Dtype dt2) {
  // TODO: check the op_type and make a real decision
  return dt1;
}

Dtype Intrinsics::IntrinsicsDtype(
    IntrinsicsOp op_type,
    const std::vector<Expr>& params) {
  // TODO: check the op_type an dmake a real decision
  CHECK_GE(params.size(), 1ULL);
  return params[0].dtype();
}

int Intrinsics::OpArgCount(IntrinsicsOp op_type) {
  switch (op_type) {
    case kSin:
    case kCos:
    case kTan:
    case kAsin:
    case kAcos:
    case kAtan:
    case kSinh:
    case kCosh:
    case kTanh:
    case kExp:
    case kFabs:
    case kLog:
    case kLog2:
    case kLog10:
    case kErf:
    case kSqrt:
    case kRsqrt:
    case kCeil:
    case kFloor:
    case kRound:
    case kTrunc:
      return 1;
    case kRand:
      return 0;
    case kFmod:
    case kPow:
      return 2;
    default:
      throw std::runtime_error("invalid op_type: " + std::to_string(op_type));
  }
}

std::string Intrinsics::func_name() const {
  switch (op_type()) {
    case kSin:
      return "sin";
    case kCos:
      return "cos";
    case kTan:
      return "tan";
    case kAsin:
      return "asin";
    case kAcos:
      return "acos";
    case kAtan:
      return "atan";
    case kSinh:
      return "sinh";
    case kCosh:
      return "cosh";
    case kTanh:
      return "tanh";
    case kExp:
      return "exp";
    case kFabs:
      return "fabs";
    case kLog:
      return "log";
    case kLog2:
      return "log2";
    case kLog10:
      return "log10";
    case kErf:
      return "erf";
    case kSqrt:
      return "sqrt";
    case kRsqrt:
      return "rsqrt";
    case kPow:
      return "pow";
    case kCeil:
      return "ceil";
    case kFloor:
      return "floor";
    case kRound:
      return "round";
    case kTrunc:
      return "trunc";
    case kRand:
      return "rand";
    case kFmod:
      return "fmod";
    default:
      throw std::runtime_error("invalid op_type: " + std::to_string(op_type()));
  }
}

} // namespace compiler
} // namespace jit
} // namespace torch
