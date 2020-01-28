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

using VarNameMap = std::unordered_map<const Variable*, std::string>;

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
  friend class ScopedVarName;
  VarNameMap unique_name_mapping_;
  std::unordered_map<std::string, int> unique_name_count_;
  std::unordered_set<std::string> all_unique_names_;
};

// A RAII wrapper to manage a variable and name pair in the look-up table.
// TODO: move this to a more shared place.
class ScopedVarName {
 public:
  ScopedVarName(
      VarNameMap* mapping,
      const Variable* var,
      const std::string& name)
      : mapping_(mapping), var_(var) {
    auto iter = mapping->find(var);
    if (iter != mapping->end()) {
      throw std::runtime_error("Duplicate var entry: " + var->name_hint());
    }
    mapping->insert(std::make_pair(var, name));
  }

  ScopedVarName(
      UniqueNameManager* manager,
      const Variable* var,
      const std::string& name)
      : ScopedVarName(&manager->unique_name_mapping_, var, name) {}

  ~ScopedVarName() {
    auto iter = mapping_->find(var_);
    if (iter == mapping_->end()) {
      throw std::runtime_error("Invalid var entry: " + var_->name_hint());
    }
    mapping_->erase(var_);
  }

 private:
  ScopedVarName(const ScopedVarName&) = delete;
  ScopedVarName& operator=(const ScopedVarName&) = delete;

  VarNameMap* mapping_ = nullptr;
  const Variable* var_ = nullptr;
};

class CudaPrinter : public IRPrinter {
 public:
  explicit CudaPrinter(std::ostream* os, UniqueNameManager* name_manager)
      : IRPrinter(*os), os_(os), name_manager_(name_manager) {}

  void visit(const Variable* v) override {
    os() << name_manager_->get_unique_name(v);
  }

  void visit(const For* v) {
    const LoopOptions& loop_options = v->loop_options();
    if (loop_options.is_gpu_block_index()) {
      ScopedVarName var_name(
          name_manager_, v->var().node(), loop_options.gpu_block_index_str());
      v->body().accept(this);
      int gpu_block_index = loop_options.gpu_block_index();
      if (gpu_block_extents_.size() <= gpu_block_index) {
        gpu_block_extents_.resize(gpu_block_index + 1);
      }
      if (!is_zero(v->start())) {
        throw std::runtime_error(
            "start must be zero for gpu_block_index: " +
            std::to_string(v->start()));
      }
      gpu_block_extents_[gpu_block_index] = v->stop();
    } else if (loop_options.is_gpu_thread_index()) {
      ScopedVarName var_name(
          name_manager_, v->var().node(), loop_options.gpu_thread_index_str());
      v->body().accept(this);
      int gpu_thread_index = loop_options.gpu_thread_index();
      if (gpu_thread_extents_.size() <= gpu_thread_index) {
        gpu_thread_extents_.resize(gpu_thread_index + 1);
      }
      if (!is_zero(v->start())) {
        throw std::runtime_error(
            "start must be zero for gpu_block_index: " +
            std::to_string(v->start()));
      }
      gpu_thread_extents_[gpu_thread_index] = v->stop();
    } else {
      IRPrinter::visit(v);
    }
  }

  std::ostream& os() {
    return *os_;
  }

  const std::vector<Expr>& gpu_block_extents() const {
    return gpu_block_extents_;
  }

  const std::vector<Expr>& gpu_thread_extents() const {
    return gpu_thread_extents_;
  }

 private:
  static bool is_zero(const Expr& expr) {
    const IntImm* v = expr.AsNode<IntImm>();
    return (v->value() == 0);
  }
  std::ostream* os_ = nullptr;
  UniqueNameManager* name_manager_ = nullptr;
  std::vector<Expr> gpu_block_extents_;
  std::vector<Expr> gpu_thread_extents_;
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

    const std::vector<Expr>& gpu_block_extents = printer_->gpu_block_extents();
    const std::vector<Expr>& gpu_thread_extents =
        printer_->gpu_thread_extents();
    for (int i = 0; i < gpu_block_extents.size(); i++) {
      if (gpu_block_extents[i].empty()) {
        throw std::runtime_error(
            "Missing gpu_block_index: " + std::to_string(i));
      }
    }

#if 0
    std::cout << "XXXQQQ: stmt: " << std::endl;
    std::cout << oss_.str() << std::endl;
    std::cout << "block(";
    for (int i = 0; i < gpu_block_extents.size(); i++) {
      if (i > 0) {
	std::cout << ", ";
      }
      std::cout << gpu_block_extents[i];
    }
    std::cout << "), thread(";
    for (int i = 0; i < gpu_thread_extents.size(); i++) {
      if (i > 0) {
	std::cout << ", ";
      }
      std::cout << gpu_thread_extents[i];
    }
    std::cout << ")" << std::endl;;
#endif
  }

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
