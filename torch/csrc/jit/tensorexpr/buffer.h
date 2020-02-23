#pragma once

#include "torch/csrc/jit/tensorexpr/ir.h"

namespace torch {
namespace jit {
namespace tensorexpr {

class Buffer {
 public:
  Buffer(const VarHandler& data, const Dtype& dtype, const std::vector<ExprHandler>& dims)
      : data_(data), dtype_(dtype), dims_(dims), strides_(dims.size()) {
    CHECK_EQ(data.dtype(), kHandle);
    for (int i = ndim() - 1; i >= 0; i--) {
      if (i == ndim() - 1) {
        strides_[i] = 1;
      } else {
        strides_[i] = strides_[i + 1] * dim(i + 1);
      }
    }
  }
  Buffer(
      const std::string& name,
      const Dtype& dtype,
      const std::vector<ExprHandler>& dims)
      : Buffer(VarHandler(name, kHandle), dtype, dims) {}

  const VarHandler& data() const {
    return data_;
  }
  const Dtype& dtype() const {
    return dtype_;
  }
  int ndim() const {
    return dims_.size();
  }
  const ExprHandler& dim(int index) const {
    return dims_[index];
  }

  // TODO: consider defer the storage flatten to a later stage.
  template <typename... Args>
  ExprHandler operator()(Args... args) const {
    ExprHandler index = Index(std::forward<Args>(args)...);
    return LoadValue(index);
  }

  template <typename T>
  ExprHandler call(const std::vector<T>& args) const {
    std::vector<ExprHandler> params(args.begin(), args.end());
    ExprHandler index = Index(params);
    return LoadValue(index);
  }

 private:
  ExprHandler Index(const ExprHandler& x) const {
    CHECK(ndim() == 1);
    return x;
  }
  ExprHandler Index(const ExprHandler& x, const ExprHandler& y) const {
    CHECK(ndim() == 2);
    return x * strides_[0] + y;
  }
  ExprHandler Index(const ExprHandler& x, const ExprHandler& y, const ExprHandler& z) const {
    CHECK(ndim() == 3);
    return x * strides_[0] + y * strides_[1] + z;
  }
  ExprHandler Index(const ExprHandler& x, const ExprHandler& y, const ExprHandler& z, const ExprHandler& w) const {
    CHECK(ndim() == 4);
    return x * strides_[0] + y * strides_[1] + z * strides_[2] + w;
  }
  ExprHandler Index(const std::vector<ExprHandler>& indices) const {
    CHECK(ndim() == (int)indices.size());
    ExprHandler total_index;
    for (size_t i = 0; i < indices.size(); i++) {
      ExprHandler index;
      if (i == indices.size() - 1) {
        index = indices[i];
      } else {
        index = indices[i] * strides_[i];
      }
      if (i == 0) {
        total_index = index;
      } else {
        total_index = total_index + index;
      }
    }
    return total_index;
  }

  ExprHandler LoadValue(const ExprHandler& index) const;

  VarHandler data_;
  Dtype dtype_;
  std::vector<ExprHandler> dims_;
  std::vector<ExprHandler> strides_;
  // TODO: add strides
};

inline ExprHandler Buffer::LoadValue(const ExprHandler& index) const {
  return Load::make(*this, index, ExprHandler(1));
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
