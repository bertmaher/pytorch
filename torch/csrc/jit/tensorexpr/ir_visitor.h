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
class Ramp;
class Load;
class For;
class Block;
class Store;
class Broadcast;
class IfThenElse;
class BaseCallNode;
class Intrinsics;
class FunctionCall;
class Allocate;
class Free;
class Cond;

class TORCH_API IRVisitor {
 public:

  TORCH_API virtual ~IRVisitor() {}

  template<typename T>
  void visit(const T* v) {
    preorder_visit(v);
    traverse(v);
    postorder_visit(v);
  }

  TORCH_API virtual void traverse(const Add* v);
  TORCH_API virtual void traverse(const Sub* v);
  TORCH_API virtual void traverse(const Mul* v);
  TORCH_API virtual void traverse(const Div* v);
  TORCH_API virtual void traverse(const Mod* v);
  TORCH_API virtual void traverse(const Max* v);
  TORCH_API virtual void traverse(const Min* v);
  TORCH_API virtual void traverse(const CompareSelect* v);
  TORCH_API virtual void traverse(const IntImm* v);
  TORCH_API virtual void traverse(const FloatImm* v);
  TORCH_API virtual void traverse(const Cast* v);
  TORCH_API virtual void traverse(const Variable* v);
  TORCH_API virtual void traverse(const Let* v);
  TORCH_API virtual void traverse(const Ramp* v);
  TORCH_API virtual void traverse(const Load* v);
  TORCH_API virtual void traverse(const For* v);
  TORCH_API virtual void traverse(const Block* v);
  TORCH_API virtual void traverse(const Store* v);
  TORCH_API virtual void traverse(const Broadcast* v);
  TORCH_API virtual void traverse(const IfThenElse* v);

  // BaseCallNode is the base class for all call nodes.
  // For any traversers that only needs the common behavior, only override this
  // function is enough. This is because all derived class handlers will call
  // this function by default.
  // Override the derived class handler only if the logic is more specific to
  // that.

  TORCH_API virtual void traverse(const BaseCallNode* v);
  TORCH_API virtual void traverse(const Intrinsics* v);
  TORCH_API virtual void traverse(const FunctionCall* v);
  TORCH_API virtual void traverse(const Allocate* v);
  TORCH_API virtual void traverse(const Free* v);
  TORCH_API virtual void traverse(const Cond* v);

  TORCH_API virtual void preorder_visit(const Add* v) {}
  TORCH_API virtual void preorder_visit(const Sub* v) {}
  TORCH_API virtual void preorder_visit(const Mul* v) {}
  TORCH_API virtual void preorder_visit(const Div* v) {}
  TORCH_API virtual void preorder_visit(const Mod* v) {}
  TORCH_API virtual void preorder_visit(const Max* v) {}
  TORCH_API virtual void preorder_visit(const Min* v) {}
  TORCH_API virtual void preorder_visit(const CompareSelect* v) {}
  TORCH_API virtual void preorder_visit(const IntImm* v) {}
  TORCH_API virtual void preorder_visit(const FloatImm* v) {}
  TORCH_API virtual void preorder_visit(const Cast* v) {}
  TORCH_API virtual void preorder_visit(const Variable* v) {}
  TORCH_API virtual void preorder_visit(const Let* v) {}
  TORCH_API virtual void preorder_visit(const Ramp* v) {}
  TORCH_API virtual void preorder_visit(const Load* v) {}
  TORCH_API virtual void preorder_visit(const For* v) {}
  TORCH_API virtual void preorder_visit(const Block* v) {}
  TORCH_API virtual void preorder_visit(const Store* v) {}
  TORCH_API virtual void preorder_visit(const Broadcast* v) {}
  TORCH_API virtual void preorder_visit(const IfThenElse* v) {}

  TORCH_API virtual void preorder_visit(const BaseCallNode* v) {}
  TORCH_API virtual void preorder_visit(const Intrinsics* v) {}
  TORCH_API virtual void preorder_visit(const FunctionCall* v) {}
  TORCH_API virtual void preorder_visit(const Allocate* v) {}
  TORCH_API virtual void preorder_visit(const Free* v) {}
  TORCH_API virtual void preorder_visit(const Cond* v) {}

  TORCH_API virtual void postorder_visit(const Add* v) {}
  TORCH_API virtual void postorder_visit(const Sub* v) {}
  TORCH_API virtual void postorder_visit(const Mul* v) {}
  TORCH_API virtual void postorder_visit(const Div* v) {}
  TORCH_API virtual void postorder_visit(const Mod* v) {}
  TORCH_API virtual void postorder_visit(const Max* v) {}
  TORCH_API virtual void postorder_visit(const Min* v) {}
  TORCH_API virtual void postorder_visit(const CompareSelect* v) {}
  TORCH_API virtual void postorder_visit(const IntImm* v) {}
  TORCH_API virtual void postorder_visit(const FloatImm* v) {}
  TORCH_API virtual void postorder_visit(const Cast* v) {}
  TORCH_API virtual void postorder_visit(const Variable* v) {}
  TORCH_API virtual void postorder_visit(const Let* v) {}
  TORCH_API virtual void postorder_visit(const Ramp* v) {}
  TORCH_API virtual void postorder_visit(const Load* v) {}
  TORCH_API virtual void postorder_visit(const For* v) {}
  TORCH_API virtual void postorder_visit(const Block* v) {}
  TORCH_API virtual void postorder_visit(const Store* v) {}
  TORCH_API virtual void postorder_visit(const Broadcast* v) {}
  TORCH_API virtual void postorder_visit(const IfThenElse* v) {}

  TORCH_API virtual void postorder_visit(const BaseCallNode* v) {}
  TORCH_API virtual void postorder_visit(const Intrinsics* v) {}
  TORCH_API virtual void postorder_visit(const FunctionCall* v) {}
  TORCH_API virtual void postorder_visit(const Allocate* v) {}
  TORCH_API virtual void postorder_visit(const Free* v) {}
  TORCH_API virtual void postorder_visit(const Cond* v) {}
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
