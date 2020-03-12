#pragma once

namespace torch {
namespace jit {
namespace tensorexpr {

class unsupported_dtype : public std::exception {
 public:
  unsupported_dtype() : std::exception() {}
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
