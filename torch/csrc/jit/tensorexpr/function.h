#pragma once

#include <functional>
#include <vector>

#include "torch/csrc/jit/tensorexpr/expr.h"
#include "torch/csrc/jit/tensorexpr/ir.h"

namespace torch {
namespace jit {
namespace tensorexpr {

// represent a range [start, stop)
class Range {
 public:
  Range() {}
  Range(const ExprHandler& start, const ExprHandler& stop) : start_(start), stop_(stop) {}
  const ExprHandler& start() const {
    return start_;
  }
  const ExprHandler& stop() const {
    return stop_;
  }

 private:
  ExprHandler start_;
  ExprHandler stop_;
};

class Function : public KernelScopedObject {
 public:
  Function(
      const std::string& func_name,
      const std::vector<ExprHandler>& dims,
      const std::vector<VarHandler>& args,
      const ExprHandler& body)
      : func_var_(func_name, kHandle), dims_(dims), args_(args), body_(body) {}

  int ndim() const {
    return dims_.size();
  }
  const ExprHandler& dim(int index) const {
    CHECK_GE(index, 0) << "index out of lower bound";
    CHECK_LT(index, ndim()) << "index out of upper bound";
    return dims_[index];
  }
  const std::vector<ExprHandler>& dims() const {
    return dims_;
  }
  const VarHandler& arg(int index) const {
    CHECK_GE(index, 0) << "index out of lower bound";
    CHECK_LT(index, ndim()) << "index out of upper bound";
    return args_[index];
  }
  const std::vector<VarHandler>& args() const {
    return args_;
  }
  const ExprHandler& body() const {
    return body_;
  }
  const VarHandler& func_var() const {
    return func_var_;
  }
  Stmt* ElementStmt();

 private:
  VarHandler func_var_;
  std::vector<ExprHandler> dims_;
  std::vector<VarHandler> args_;
  ExprHandler body_;
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
