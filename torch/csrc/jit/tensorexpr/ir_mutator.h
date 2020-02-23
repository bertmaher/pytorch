#pragma once
#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch {
namespace jit {
namespace tensorexpr {

class Add;
class Sub;
class Mul;
class Div;
class Mod;
class Max;
class Min;
class CompareSelect;
class IntImm;
class FloatImm;
class Cast;
class Variable;
class Let;
class LetStmt;
class Ramp;
class Load;
class For;
class Block;
class Store;
class Broadcast;
class IfThenElse;
class Expr;
class BaseExprNode;
class BaseCallNode;
class Intrinsics;
class FunctionCall;
class Allocate;
class Free;
class Cond;
class Stmt;

class TORCH_API IRMutator {
 public:
  virtual ~IRMutator() {}
  virtual const BaseExprNode* mutate(const Add* v);
  virtual const BaseExprNode* mutate(const Sub* v);
  virtual const BaseExprNode* mutate(const Mul* v);
  virtual const BaseExprNode* mutate(const Div* v);
  virtual const BaseExprNode* mutate(const Mod* v);
  virtual const BaseExprNode* mutate(const Max* v);
  virtual const BaseExprNode* mutate(const Min* v);
  virtual const BaseExprNode* mutate(const CompareSelect* v);
  virtual const BaseExprNode* mutate(const IntImm* v);
  virtual const BaseExprNode* mutate(const FloatImm* v);
  virtual const BaseExprNode* mutate(const Cast* v);
  virtual const BaseExprNode* mutate(const Variable* v);
  virtual const BaseExprNode* mutate(const Let* v);
  virtual Stmt* mutate(const LetStmt* v);
  virtual const BaseExprNode* mutate(const Ramp* v);
  virtual const BaseExprNode* mutate(const Load* v);
  virtual const BaseExprNode* mutate(const Broadcast* v);
  virtual const BaseExprNode* mutate(const IfThenElse* v);
  // BaseCallNode is the base class for all call nodes.
  // For any visitors that only needs the common behavior, only override this
  // function is enough. This is because all derived class handlers will call
  // this function by default.
  // Override the derived class handler only if the logic is more specific to
  // that.
  virtual const BaseExprNode* mutate(const BaseCallNode* v);
  virtual const BaseExprNode* mutate(const Intrinsics* v);
  virtual const BaseExprNode* mutate(const FunctionCall* v);

  virtual Stmt* mutate(const For* v);
  virtual Stmt* mutate(const Block* v);
  virtual Stmt* mutate(const Store* v);

  virtual Stmt* mutate(const Allocate* v);
  virtual Stmt* mutate(const Free* v);
  virtual Stmt* mutate(const Cond* v);
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
