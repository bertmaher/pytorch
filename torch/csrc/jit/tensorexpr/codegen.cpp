#include "torch/csrc/jit/tensorexpr/codegen.h"

#include <sstream>

namespace torch {
namespace jit {
namespace tensorexpr {

RegisterCodeGenList::ExprFactoryMethod RegisterCodeGenList::
    FindExprFactoryMethod(const std::string& name) {
  auto iter = expr_factory_methods_.find(name);
  if (iter == expr_factory_methods_.end()) {
    std::ostringstream oss;
    oss << "Invalid codegen name: " << name << ". ";
    oss << "Existing codegen names: [";
    int index = 0;
    for (const auto& entry : expr_factory_methods_) {
      if (index != 0) {
        oss << ", ";
      }
      oss << entry.first;
      index++;
    }
    oss << "]";
    throw std::runtime_error(oss.str());
  }
  return iter->second;
}

void RegisterCodeGenList::AddExprFactoryMethod(
    const std::string& name,
    ExprFactoryMethod expr_factory_method) {
  auto insert_ret =
      expr_factory_methods_.insert(std::make_pair(name, expr_factory_method));
  if (!insert_ret.second) {
    throw std::runtime_error("Duplicated CodeGen names: " + name);
  }
}

std::unique_ptr<CodeGen> CreateCodeGen(
    const std::string& name,
    const Expr& expr,
    const std::vector<CodeGen::BufferArg>& params) {
  RegisterCodeGenList::ExprFactoryMethod method =
      RegisterCodeGenList::GetInstance().FindExprFactoryMethod(name);
  return method(expr, params);
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
