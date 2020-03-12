#pragma once

#include <stdexcept>

namespace torch {
namespace jit {
namespace tensorexpr {

class unsupported_dtype : public std::runtime_error {
 public:
  unsupported_dtype()
    : std::runtime_error("UNSUPPORTED_DTYPE") {}
  unsupported_dtype(const std::string& err)
    : std::runtime_error("UNSUPPORTED_DTYPE: " + err) {}
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
