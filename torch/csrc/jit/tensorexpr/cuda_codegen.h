#pragma once

#include <unordered_map>
#include <unordered_set>

#include "torch/csrc/jit/tensorexpr/codegen.h"
#include "torch/csrc/jit/tensorexpr/ir.h"
#include "torch/csrc/jit/tensorexpr/ir_printer.h"
#include "torch/csrc/jit/tensorexpr/ir_visitor.h"

namespace torch {
namespace jit {
namespace compiler {

class UniqueNameManager {
 public:
  const std::string& get_unique_name(const Variable* v) {
    // Find if we have already encountered this variable.
    auto iter = unique_name_mapping_.find(v);
    if (iter != unique_name_mapping_.end()) {
      return iter->second;
    }

    // First use the name_hint as a prefix to check if there is another name
    // with the same prefix.
    const std::string& name_hint = v->name_hint();
    int& count = unique_name_count_[name_hint];
    while (1) {
      // Even if with a new count, this name might already be used. For example
      // ("x", 1) could collidewith ("x_1", 0)
      int count_v = count++;
      std::string unique_name = name_hint;
      if (count_v > -1) {
        unique_name += "_" + std::to_string(count_v);
      }
      if (all_unique_names_.count(unique_name) == 0) {
        all_unique_names_.insert(unique_name);
        auto result =
            unique_name_mapping_.insert(std::make_pair(v, unique_name));
        return result.first->second;
      }
    }
  }
  const std::string& get_unique_name(const Var& v) {
    return get_unique_name(v.node());
  }

 private:
  std::unordered_map<const Variable*, std::string> unique_name_mapping_;
  std::unordered_map<std::string, int> unique_name_count_;
  std::unordered_set<std::string> all_unique_names_;
};

class CudaPrinter : public IRPrinter {
 public:
  explicit CudaPrinter(std::ostream* os, UniqueNameManager* name_manager)
      : IRPrinter(*os), os_(os), name_manager_(name_manager) {}

  void visit(const Variable* v) override {
    (*os_) << name_manager_->get_unique_name(v);
  }

 private:
  std::ostream* os_ = nullptr;
  UniqueNameManager* name_manager_ = nullptr;
};

class CudaCodeGen : public CodeGen {
 public:
  template <typename... Ts>
  CudaCodeGen(const Stmt& stmt, Ts... ts)
      : CodeGen(stmt, std::forward<Ts>(ts)...) {
    printer_.reset(new CudaPrinter(&oss_, &name_manager_));
    // TODO: handle multiple kernels.
    // TODO: handle dynamic dimension.
    // TODO: call nvrtc.
    oss_ << "extern \"C\" __global__" << std::endl << "void f(";
    const std::vector<BufferArg> buffer_args = this->buffer_args();
    for (int i = 0; i < buffer_args.size(); i++) {
      if (i > 0) {
        oss_ << ", ";
      }
      const BufferArg& buffer_arg = buffer_args[i];
      const Var& var = buffer_arg.var();
      Dtype dtype = buffer_arg.dtype();
      oss_ << dtype.ToCppString() << "* " << name_manager_.get_unique_name(var);
    }
    oss_ << ") {";

    oss_ << std::endl;
    stmt.accept(printer_.get());
    oss_ << std::endl;
    oss_ << "}";
  }

  ~CudaCodeGen() override {}

  template <typename... Ts>
  void operator()(const Ts&... ts) {
    std::vector<CallArg> args({CallArg(ts)...});
    CHECK_EQ(args.size(), buffer_args().size());
  }

 private:
  UniqueNameManager name_manager_;
  std::ostringstream oss_;
  std::unique_ptr<CudaPrinter> printer_;
};

} // namespace compiler
} // namespace jit
} // namespace torch
