#pragma once

namespace torch {
namespace jit {
namespace compiler {

class Add;
class Sub;
class Mul;
class Div;
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
class BaseCallNode;
class Intrinsics;
class FunctionCall;
class Allocate;
class Free;

class IRVisitor {
 public:
  virtual void visit(const Add* v);
  virtual void visit(const Sub* v);
  virtual void visit(const Mul* v);
  virtual void visit(const Div* v);
  virtual void visit(const Max* v);
  virtual void visit(const Min* v);
  virtual void visit(const CompareSelect* v);
  virtual void visit(const IntImm* v);
  virtual void visit(const FloatImm* v);
  virtual void visit(const Cast* v);
  virtual void visit(const Variable* v);
  virtual void visit(const Let* v);
  virtual void visit(const Ramp* v);
  virtual void visit(const Load* v);
  virtual void visit(const For* v);
  virtual void visit(const Block* v);
  virtual void visit(const Store* v);
  virtual void visit(const Broadcast* v);
  // BaseCallNode is the base class for all call nodes.
  // For any visitors that only needs the common behavior, only override this
  // function is enough. This is because all derived class handlers will call
  // this function by default.
  // Override the derived class handler only if the logic is more specific to
  // that.
  virtual void visit(const BaseCallNode* v);
  virtual void visit(const Intrinsics* v);
  virtual void visit(const FunctionCall* v);
  virtual void visit(const Allocate* v);
  virtual void visit(const Free* v);
};

} // namespace compiler
} // namespace jit
} // namespace torch
