#pragma once

#include <stdexcept>
#include <sstream>

// Forward declarations of types
namespace torch {
namespace jit {
namespace tensorexpr {
class Expr;
class Stmt;
}
}
}

// Forward declarations of functions
namespace std {
std::string to_string(const torch::jit::tensorexpr::Expr*);
std::string to_string(const torch::jit::tensorexpr::Stmt*);
}

namespace torch {
namespace jit {
namespace tensorexpr {

class unsupported_dtype : public std::runtime_error {
 public:
  unsupported_dtype()
    : std::runtime_error("UNSUPPORTED DTYPE") {}
  unsupported_dtype(const std::string& err)
    : std::runtime_error("UNSUPPORTED DTYPE: " + err) {}
};

class unimplemented_lowering : public std::runtime_error {
 public:
  unimplemented_lowering(const Expr* expr)
    : std::runtime_error("UNIMPLEMENTED LOWERING: " + std::to_string(expr)) { }
  unimplemented_lowering(const Stmt* stmt)
    : std::runtime_error("UNIMPLEMENTED LOWERING: " + std::to_string(stmt)) { }
};

class malformed_input : public std::runtime_error {
 public:
  malformed_input() : std::runtime_error("MALFORMED INPUT") { }
  malformed_input(const Expr* expr)
    : std::runtime_error("MALFORMED INPUT: " + std::to_string(expr)) { }
  malformed_input(const Stmt* stmt)
    : std::runtime_error("MALFORMED INPUT: " + std::to_string(stmt)) { }
};


} // namespace tensorexpr
} // namespace jit
} // namespace torch
