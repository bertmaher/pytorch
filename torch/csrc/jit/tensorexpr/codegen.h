#pragma once

#include "torch/csrc/jit/tensorexpr/buffer.h"
#include "torch/csrc/jit/tensorexpr/ir.h"
#include "torch/csrc/jit/tensorexpr/tensor.h"

namespace torch {
namespace jit {
namespace tensorexpr {

template <typename T>
class PaddedBuffer;

class CodeGen {
 public:
  class BufferArg;
  class CallArg;

  template <typename... Ts>
  CodeGen(const Stmt& stmt, Ts... ts)
      : ir_node_(const_cast<BaseStmtNode*>(stmt.node())),
        buffer_args_({BufferArg(ts)...}) {}

  CodeGen(const Stmt& stmt, const std::vector<BufferArg>& buffer_args)
      : ir_node_(const_cast<BaseStmtNode*>(stmt.node())),
        buffer_args_(buffer_args) {}

  template <typename... Ts>
  CodeGen(const Expr& expr, Ts... ts)
      : ir_node_(const_cast<BaseExprNode*>(expr.node())),
        buffer_args_({BufferArg(ts)...}) {}

  CodeGen(const Expr& expr, const std::vector<BufferArg>& buffer_args)
      : ir_node_(const_cast<BaseExprNode*>(expr.node())),
        buffer_args_(buffer_args) {}

  CodeGen(const IRNode* node, const std::vector<BufferArg>& buffer_args)
      : ir_node_(const_cast<IRNode*>(node)), buffer_args_(buffer_args) {}

  virtual ~CodeGen() {}

  IRNode* ir_node() {
    return ir_node_;
  }

  const IRNode* ir_node() const {
    return ir_node_;
  }

  std::vector<BufferArg>& buffer_args() {
    return buffer_args_;
  }

  const std::vector<BufferArg>& buffer_args() const {
    return buffer_args_;
  }

  TORCH_API virtual void call(const std::vector<CallArg>& args) {
    LOG(FATAL) << "unimplemented call";
  }

 private:
  IRNode* ir_node_ = nullptr;
  std::vector<BufferArg> buffer_args_;
};

class CodeGen::BufferArg {
 public:
  BufferArg(const Buffer& buffer)
      : var_(buffer.data()), dtype_(buffer.dtype()) {}
  BufferArg(const Tensor& tensor)
      : var_(tensor.function().func_var()),
        dtype_(tensor.function().body().dtype()) {}
  BufferArg(const Function& func)
      : var_(func.func_var()), dtype_(func.body().dtype()) {}
  BufferArg(const Var& var) : var_(var), dtype_(var.dtype()), isVar_(true) {}

  const Var& var() const {
    return var_;
  }
  Var& var() {
    return var_;
  }
  Dtype dtype() const {
    return dtype_;
  }

  bool isVar() const {
    return isVar_;
  }

 private:
  Var var_;
  Dtype dtype_;
  bool isVar_{false};
};

class CodeGen::CallArg {
 public:
  template <typename T>
  CallArg(const PaddedBuffer<T>& buffer);

  template <typename T>
  CallArg(const std::vector<T>& buffer) : ptr_(const_cast<T*>(buffer.data())) {}

  CallArg(void* ptr) : ptr_(ptr) {}

  CallArg(int32_t i) : ival_(i) {}

  CallArg(float f) : fval_(f) {}

  void* data() const {
    return ptr_;
  }

  int32_t intData() const {
    return ival_;
  }

  float floatData() const {
    return fval_;
  }

  int* intPtr() const {
    return const_cast<int*>(&ival_);
  }

  float* floatPtr() const {
    return const_cast<float*>(&fval_);
  }

 private:
  union {
    void* ptr_;
    float fval_;
    int32_t ival_;
  };
};

class RegisterCodeGenList {
 public:
  static RegisterCodeGenList& GetInstance() {
    static RegisterCodeGenList codegen_list;
    return codegen_list;
  }

  using StmtFactoryMethod = std::function<std::unique_ptr<CodeGen>(
      const Stmt& stmt,
      const std::vector<CodeGen::BufferArg>&)>;
  using ExprFactoryMethod = std::function<std::unique_ptr<CodeGen>(
      const Expr& expr,
      const std::vector<CodeGen::BufferArg>&)>;

  TORCH_API StmtFactoryMethod FindStmtFactoryMethod(const std::string& name);
  TORCH_API ExprFactoryMethod FindExprFactoryMethod(const std::string& name);

 private:
  template <class CodeGenType>
  friend class RegisterCodeGen;
  RegisterCodeGenList() {}
  TORCH_API void AddStmtFactoryMethod(
      const std::string& name,
      StmtFactoryMethod stmt_factory_method);
  TORCH_API void AddExprFactoryMethod(
      const std::string& name,
      ExprFactoryMethod expr_factory_method);
  RegisterCodeGenList(const RegisterCodeGenList&) = delete;
  RegisterCodeGenList& operator=(const RegisterCodeGenList&) = delete;

  std::unordered_map<std::string, StmtFactoryMethod> stmt_factory_methods_;
  std::unordered_map<std::string, ExprFactoryMethod> expr_factory_methods_;
};

template <class CodeGenType>
class RegisterCodeGen {
 public:
  explicit RegisterCodeGen(const std::string& name) {
    RegisterCodeGenList& codegen_list = RegisterCodeGenList::GetInstance();
    codegen_list.AddStmtFactoryMethod(
        name,
        [](const Stmt& stmt, const std::vector<CodeGen::BufferArg>& params) {
          std::unique_ptr<CodeGen> method(new CodeGenType(stmt, params));
          return method;
        });
#if 0
    // TODO: decide whether we need this Expr version.
    codegen_list.AddExprFactoryMethod(name, [](const Expr& expr, const std::vector<CodeGen::BufferArg>& params) {
	std::unique_ptr<CodeGen> method(new CodeGenType(expr, params));
	return method;
      });
#endif
  }
};

TORCH_API std::unique_ptr<CodeGen> CreateCodeGen(
    const std::string& name,
    const Stmt& stmt,
    const std::vector<CodeGen::BufferArg>& params);

TORCH_API std::unique_ptr<CodeGen> CreateCodeGen(
    const std::string& name,
    const Expr& expr,
    const std::vector<CodeGen::BufferArg>& params);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
