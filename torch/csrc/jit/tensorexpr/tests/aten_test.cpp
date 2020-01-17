#include <sstream>
#include <stdexcept>
#include <algorithm>

#include <gtest/gtest.h>

#include "torch/csrc/jit/tensorexpr/ir_printer.h"
#include "torch/csrc/jit/tensorexpr/tests/test_utils.h"

using namespace torch::jit::compiler;

TEST(ATenTest, _cast_Float) {
  const int kTotalSize = 128;
  Buffer a_buf(Var("A", kHandle), kInt32, {Expr(kTotalSize)});
  Buffer b_buf(Var("B", kHandle), kFloat32, {Expr(kTotalSize)});

  Var index = Var("index", kInt32);
  Expr load_a = Load::make(
      a_buf,
      index,
      1);
  Expr to_float = Cast::make(kFloat32, load_a);
  Stmt store_b = Store::make(
      b_buf,
      index,
      to_float,
      1);
  Stmt stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
      a_v(i) = i;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf);
  ir_eval(a_v, b_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i) << "index: " << i;
    EXPECT_EQ(b_v(i), static_cast<float>(i)) << "index: " << i;
  }
}

TEST(ATenTest, negInt) {
  const int kTotalSize = 128;
  Buffer a_buf(Var("A", kHandle), kInt32, {Expr(kTotalSize)});
  Buffer b_buf(Var("B", kHandle), kInt32, {Expr(kTotalSize)});

  Var index = Var("index", kInt32);
  Expr load_a = Load::make(
      a_buf,
      index,
      1);
  Expr to_float = Sub::make(0, load_a);
  Stmt store_b = Store::make(
      b_buf,
      index,
      to_float,
      1);
  Stmt stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<int> b_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
      a_v(i) = i;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf);
  ir_eval(a_v, b_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i) << "index: " << i;
    EXPECT_EQ(b_v(i), -static_cast<float>(i)) << "index: " << i;
  }
}

TEST(ATenTest, negFloat) {
  const int kTotalSize = 128;
  Buffer a_buf(Var("A", kHandle), kFloat32, {Expr(kTotalSize)});
  Buffer b_buf(Var("B", kHandle), kFloat32, {Expr(kTotalSize)});

  Var index = Var("index", kInt32);
  Expr load_a = Load::make(
      a_buf,
      index,
      1);
  Expr to_float = Sub::make(0, load_a);
  Stmt store_b = Store::make(
      b_buf,
      index,
      to_float,
      1);
  Stmt stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
      a_v(i) = i;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf);
  ir_eval(a_v, b_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i) << "index: " << i;
    EXPECT_EQ(b_v(i), -i) << "index: " << i;
  }
}

TEST(ATenTest, addInt) {
  const int kTotalSize = 128;
  Buffer a_buf(Var("A", kHandle), kInt32, {Expr(kTotalSize)});
  Buffer b_buf(Var("B", kHandle), kInt32, {Expr(kTotalSize)});
  Buffer c_buf(Var("C", kHandle), kInt32, {Expr(kTotalSize)});
  Buffer d_buf(Var("D", kHandle), kInt32, {Expr(kTotalSize)});

  Var index = Var("index", kInt32);
  Expr load_a = Load::make(
      a_buf,
      index,
      1);
  Expr load_b = Load::make(
      b_buf,
      index,
      1);
  Expr load_c = Load::make(
      c_buf,
      index,
      1);
  Stmt store_d = Store::make(
      d_buf,
      index,
      load_a + load_b * load_c,
      1);
  Stmt stmt = For::make(index, 0, kTotalSize, store_d);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<int> b_v(kTotalSize);
  PaddedBuffer<int> c_v(kTotalSize);
  PaddedBuffer<int> d_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
      a_v(i) = i;
      b_v(i) = 2*i+1;
      c_v(i) = 3*i+2;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf, d_buf);
  ir_eval(a_v, b_v, c_v, d_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i) << "index: " << i;
    EXPECT_EQ(b_v(i), 2*i+1) << "index: " << i;
    EXPECT_EQ(c_v(i), 3*i+2) << "index: " << i;
    EXPECT_EQ(d_v(i), a_v(i)+b_v(i)*c_v(i)) << "index: " << i;
  }
}

TEST(ATenTest, addFloat) {
  const int kTotalSize = 128;
  Buffer a_buf(Var("A", kHandle), kFloat32, {Expr(kTotalSize)});
  Buffer b_buf(Var("B", kHandle), kFloat32, {Expr(kTotalSize)});
  Buffer c_buf(Var("C", kHandle), kFloat32, {Expr(kTotalSize)});
  Buffer d_buf(Var("D", kHandle), kFloat32, {Expr(kTotalSize)});

  Var index = Var("index", kInt32);
  Expr load_a = Load::make(
      a_buf,
      index,
      1);
  Expr load_b = Load::make(
      b_buf,
      index,
      1);
  Expr load_c = Load::make(
      c_buf,
      index,
      1);
  Stmt store_d = Store::make(
      d_buf,
      index,
      load_a + load_b * load_c,
      1);
  Stmt stmt = For::make(index, 0, kTotalSize, store_d);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);
  PaddedBuffer<float> c_v(kTotalSize);
  PaddedBuffer<float> d_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
      a_v(i) = i;
      b_v(i) = 2*i+1;
      c_v(i) = 3*i+2;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf, d_buf);
  ir_eval(a_v, b_v, c_v, d_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i) << "index: " << i;
    EXPECT_EQ(b_v(i), 2*i+1) << "index: " << i;
    EXPECT_EQ(c_v(i), 3*i+2) << "index: " << i;
    EXPECT_EQ(d_v(i), a_v(i)+b_v(i)*c_v(i)) << "index: " << i;
  }
}

TEST(ATenTest, subInt) {
  const int kTotalSize = 128;
  Buffer a_buf(Var("A", kHandle), kInt32, {Expr(kTotalSize)});
  Buffer b_buf(Var("B", kHandle), kInt32, {Expr(kTotalSize)});
  Buffer c_buf(Var("C", kHandle), kInt32, {Expr(kTotalSize)});
  Buffer d_buf(Var("D", kHandle), kInt32, {Expr(kTotalSize)});

  Var index = Var("index", kInt32);
  Expr load_a = Load::make(
      a_buf,
      index,
      1);
  Expr load_b = Load::make(
      b_buf,
      index,
      1);
  Expr load_c = Load::make(
      c_buf,
      index,
      1);
  Stmt store_d = Store::make(
      d_buf,
      index,
      load_a - load_b * load_c,
      1);
  Stmt stmt = For::make(index, 0, kTotalSize, store_d);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<int> b_v(kTotalSize);
  PaddedBuffer<int> c_v(kTotalSize);
  PaddedBuffer<int> d_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
      a_v(i) = i;
      b_v(i) = 2*i+1;
      c_v(i) = 3*i+2;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf, d_buf);
  ir_eval(a_v, b_v, c_v, d_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i) << "index: " << i;
    EXPECT_EQ(b_v(i), 2*i+1) << "index: " << i;
    EXPECT_EQ(c_v(i), 3*i+2) << "index: " << i;
    EXPECT_EQ(d_v(i), a_v(i)-b_v(i)*c_v(i)) << "index: " << i;
  }
}

TEST(ATenTest, subFloat) {
  const int kTotalSize = 128;
  Buffer a_buf(Var("A", kHandle), kFloat32, {Expr(kTotalSize)});
  Buffer b_buf(Var("B", kHandle), kFloat32, {Expr(kTotalSize)});
  Buffer c_buf(Var("C", kHandle), kFloat32, {Expr(kTotalSize)});
  Buffer d_buf(Var("D", kHandle), kFloat32, {Expr(kTotalSize)});

  Var index = Var("index", kInt32);
  Expr load_a = Load::make(
      a_buf,
      index,
      1);
  Expr load_b = Load::make(
      b_buf,
      index,
      1);
  Expr load_c = Load::make(
      c_buf,
      index,
      1);
  Stmt store_d = Store::make(
      d_buf,
      index,
      load_a - load_b * load_c,
      1);
  Stmt stmt = For::make(index, 0, kTotalSize, store_d);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);
  PaddedBuffer<float> c_v(kTotalSize);
  PaddedBuffer<float> d_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
      a_v(i) = i;
      b_v(i) = 2*i+1;
      c_v(i) = 3*i+2;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf, d_buf);
  ir_eval(a_v, b_v, c_v, d_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i) << "index: " << i;
    EXPECT_EQ(b_v(i), 2*i+1) << "index: " << i;
    EXPECT_EQ(c_v(i), 3*i+2) << "index: " << i;
    EXPECT_EQ(d_v(i), a_v(i)-b_v(i)*c_v(i)) << "index: " << i;
  }
}

TEST(ATenTest, lerp) {
  const int kTotalSize = 128;
  Buffer a_buf(Var("A", kHandle), kFloat32, {Expr(kTotalSize)});
  Buffer b_buf(Var("B", kHandle), kFloat32, {Expr(kTotalSize)});
  Buffer c_buf(Var("C", kHandle), kFloat32, {Expr(kTotalSize)});
  Buffer d_buf(Var("D", kHandle), kFloat32, {Expr(kTotalSize)});

  Var index = Var("index", kInt32);
  Expr load_a = Load::make(
      a_buf,
      index,
      1);
  Expr load_b = Load::make(
      b_buf,
      index,
      1);
  Expr load_c = Load::make(
      c_buf,
      index,
      1);
  Stmt store_d = Store::make(
      d_buf,
      index,
      load_a + load_c * (load_b - load_a),
      1);
  Stmt stmt = For::make(index, 0, kTotalSize, store_d);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);
  PaddedBuffer<float> c_v(kTotalSize);
  PaddedBuffer<float> d_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
      a_v(i) = i;
      b_v(i) = 2*i+1;
      c_v(i) = 3*i+2;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf, d_buf);
  ir_eval(a_v, b_v, c_v, d_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i) << "index: " << i;
    EXPECT_EQ(b_v(i), 2*i+1) << "index: " << i;
    EXPECT_EQ(c_v(i), 3*i+2) << "index: " << i;
    EXPECT_EQ(d_v(i), a_v(i)+c_v(i)*(b_v(i) - a_v(i))) << "index: " << i;
  }
}

TEST(ATenTest, addcmulInt) {
  const int kTotalSize = 128;
  Buffer a_buf(Var("A", kHandle), kInt32, {Expr(kTotalSize)});
  Buffer b_buf(Var("B", kHandle), kInt32, {Expr(kTotalSize)});
  Buffer c_buf(Var("C", kHandle), kInt32, {Expr(kTotalSize)});
  Buffer d_buf(Var("D", kHandle), kInt32, {Expr(kTotalSize)});
  Buffer e_buf(Var("E", kHandle), kInt32, {Expr(kTotalSize)});

  Var index = Var("index", kInt32);
  Expr load_a = Load::make(
      a_buf,
      index,
      1);
  Expr load_b = Load::make(
      b_buf,
      index,
      1);
  Expr load_c = Load::make(
      c_buf,
      index,
      1);
  Expr load_d = Load::make(
      d_buf,
      index,
      1);
  Stmt store_e = Store::make(
      e_buf,
      index,
      load_a + load_b * load_c * load_d,
      1);
  Stmt stmt = For::make(index, 0, kTotalSize, store_e);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<int> b_v(kTotalSize);
  PaddedBuffer<int> c_v(kTotalSize);
  PaddedBuffer<int> d_v(kTotalSize);
  PaddedBuffer<int> e_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
      a_v(i) = i;
      b_v(i) = 2*i+1;
      c_v(i) = 3*i+2;
      d_v(i) = 5*i+3;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf, d_buf, e_buf);
  ir_eval(a_v, b_v, c_v, d_v, e_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i) << "index: " << i;
    EXPECT_EQ(b_v(i), 2*i+1) << "index: " << i;
    EXPECT_EQ(c_v(i), 3*i+2) << "index: " << i;
    EXPECT_EQ(d_v(i), 5*i+3) << "index: " << i;
    EXPECT_EQ(e_v(i), a_v(i) + b_v(i)*c_v(i)*d_v(i)) << "index: " << i;
  }
}

TEST(ATenTest, addcmulFloat) {
  const int kTotalSize = 128;
  Buffer a_buf(Var("A", kHandle), kFloat32, {Expr(kTotalSize)});
  Buffer b_buf(Var("B", kHandle), kFloat32, {Expr(kTotalSize)});
  Buffer c_buf(Var("C", kHandle), kFloat32, {Expr(kTotalSize)});
  Buffer d_buf(Var("D", kHandle), kFloat32, {Expr(kTotalSize)});
  Buffer e_buf(Var("E", kHandle), kFloat32, {Expr(kTotalSize)});

  Var index = Var("index", kInt32);
  Expr load_a = Load::make(
      a_buf,
      index,
      1);
  Expr load_b = Load::make(
      b_buf,
      index,
      1);
  Expr load_c = Load::make(
      c_buf,
      index,
      1);
  Expr load_d = Load::make(
      d_buf,
      index,
      1);
  Stmt store_e = Store::make(
      e_buf,
      index,
      load_a + load_b*load_c*load_d,
      1);
  Stmt stmt = For::make(index, 0, kTotalSize, store_e);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);
  PaddedBuffer<float> c_v(kTotalSize);
  PaddedBuffer<float> d_v(kTotalSize);
  PaddedBuffer<float> e_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
      a_v(i) = i;
      b_v(i) = 2*i+1;
      c_v(i) = 3*i+2;
      d_v(i) = 5*i+3;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf, d_buf, e_buf);
  ir_eval(a_v, b_v, c_v, d_v, e_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i) << "index: " << i;
    EXPECT_EQ(b_v(i), 2*i+1) << "index: " << i;
    EXPECT_EQ(c_v(i), 3*i+2) << "index: " << i;
    EXPECT_EQ(d_v(i), 5*i+3) << "index: " << i;
    EXPECT_EQ(e_v(i), a_v(i) + b_v(i)*c_v(i)*d_v(i)) << "index: " << i;
  }
}

TEST(ATenTest, mulInt) {
  const int kTotalSize = 128;
  Buffer a_buf(Var("A", kHandle), kInt32, {Expr(kTotalSize)});
  Buffer b_buf(Var("B", kHandle), kInt32, {Expr(kTotalSize)});
  Buffer c_buf(Var("C", kHandle), kInt32, {Expr(kTotalSize)});

  Var index = Var("index", kInt32);
  Expr load_a = Load::make(
      a_buf,
      index,
      1);
  Expr load_b = Load::make(
      b_buf,
      index,
      1);
  Stmt store_c = Store::make(
      c_buf,
      index,
      load_a * load_b,
      1);
  Stmt stmt = For::make(index, 0, kTotalSize, store_c);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<int> b_v(kTotalSize);
  PaddedBuffer<int> c_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
      a_v(i) = i;
      b_v(i) = 2*i+1;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf);
  ir_eval(a_v, b_v, c_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i) << "index: " << i;
    EXPECT_EQ(b_v(i), 2*i+1) << "index: " << i;
    EXPECT_EQ(c_v(i), a_v(i) * b_v(i)) << "index: " << i;
  }
}

TEST(ATenTest, mulFloat) {
  const int kTotalSize = 128;
  Buffer a_buf(Var("A", kHandle), kFloat32, {Expr(kTotalSize)});
  Buffer b_buf(Var("B", kHandle), kFloat32, {Expr(kTotalSize)});
  Buffer c_buf(Var("C", kHandle), kFloat32, {Expr(kTotalSize)});

  Var index = Var("index", kInt32);
  Expr load_a = Load::make(
      a_buf,
      index,
      1);
  Expr load_b = Load::make(
      b_buf,
      index,
      1);
  Stmt store_c = Store::make(
      c_buf,
      index,
      load_a * load_b,
      1);
  Stmt stmt = For::make(index, 0, kTotalSize, store_c);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);
  PaddedBuffer<float> c_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
      a_v(i) = i;
      b_v(i) = 2*i+1;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf);
  ir_eval(a_v, b_v, c_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i) << "index: " << i;
    EXPECT_EQ(b_v(i), 2*i+1) << "index: " << i;
    EXPECT_EQ(c_v(i), a_v(i) * b_v(i)) << "index: " << i;
  }
}

TEST(ATenTest, divInt) {
  const int kTotalSize = 128;
  Buffer a_buf(Var("A", kHandle), kInt32, {Expr(kTotalSize)});
  Buffer b_buf(Var("B", kHandle), kInt32, {Expr(kTotalSize)});
  Buffer c_buf(Var("C", kHandle), kInt32, {Expr(kTotalSize)});

  Var index = Var("index", kInt32);
  Expr load_a = Load::make(
      a_buf,
      index,
      1);
  Expr load_b = Load::make(
      b_buf,
      index,
      1);
  Stmt store_c = Store::make(
      c_buf,
      index,
      load_a / load_b,
      1);
  Stmt stmt = For::make(index, 0, kTotalSize, store_c);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<int> b_v(kTotalSize);
  PaddedBuffer<int> c_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
      a_v(i) = 2*i+1;
      b_v(i) = i+1;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf);
  ir_eval(a_v, b_v, c_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), 2*i+1) << "index: " << i;
    EXPECT_EQ(b_v(i), i+1) << "index: " << i;
    EXPECT_EQ(c_v(i), a_v(i) / b_v(i)) << "index: " << i;
  }
}

TEST(ATenTest, divFloat) {
  const int kTotalSize = 128;
  Buffer a_buf(Var("A", kHandle), kFloat32, {Expr(kTotalSize)});
  Buffer b_buf(Var("B", kHandle), kFloat32, {Expr(kTotalSize)});
  Buffer c_buf(Var("C", kHandle), kFloat32, {Expr(kTotalSize)});

  Var index = Var("index", kInt32);
  Expr load_a = Load::make(
      a_buf,
      index,
      1);
  Expr load_b = Load::make(
      b_buf,
      index,
      1);
  Stmt store_c = Store::make(
      c_buf,
      index,
      load_a / load_b,
      1);
  Stmt stmt = For::make(index, 0, kTotalSize, store_c);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);
  PaddedBuffer<float> c_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
      a_v(i) = 2*i+1;
      b_v(i) = i+1;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf);
  ir_eval(a_v, b_v, c_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), 2*i+1) << "index: " << i;
    EXPECT_EQ(b_v(i), i+1) << "index: " << i;
    EXPECT_EQ(c_v(i), a_v(i) / b_v(i)) << "index: " << i;
  }
}

TEST(ATenTest, maxInt) {
  const int kTotalSize = 128;
  Buffer a_buf(Var("A", kHandle), kInt32, {Expr(kTotalSize)});
  Buffer b_buf(Var("B", kHandle), kInt32, {Expr(kTotalSize)});
  Buffer c_buf(Var("C", kHandle), kInt32, {Expr(kTotalSize)});

  Var index = Var("index", kInt32);
  Expr load_a = Load::make(
      a_buf,
      index,
      1);
  Expr load_b = Load::make(
      b_buf,
      index,
      1);
  Stmt store_c = Store::make(
      c_buf,
      index,
      Max::make(load_a, load_b, true),
      1);
  Stmt stmt = For::make(index, 0, kTotalSize, store_c);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<int> b_v(kTotalSize);
  PaddedBuffer<int> c_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
      a_v(i) = i;
      b_v(i) = 2*i+1;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf);
  ir_eval(a_v, b_v, c_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i) << "index: " << i;
    EXPECT_EQ(b_v(i), 2*i+1) << "index: " << i;
    EXPECT_EQ(c_v(i), std::max(a_v(i), b_v(i))) << "index: " << i;
  }
}

TEST(ATenTest, maxFloat) {
  const int kTotalSize = 128;
  Buffer a_buf(Var("A", kHandle), kFloat32, {Expr(kTotalSize)});
  Buffer b_buf(Var("B", kHandle), kFloat32, {Expr(kTotalSize)});
  Buffer c_buf(Var("C", kHandle), kFloat32, {Expr(kTotalSize)});

  Var index = Var("index", kInt32);
  Expr load_a = Load::make(
      a_buf,
      index,
      1);
  Expr load_b = Load::make(
      b_buf,
      index,
      1);
  Stmt store_c = Store::make(
      c_buf,
      index,
      Max::make(load_a, load_b, true),
      1);
  Stmt stmt = For::make(index, 0, kTotalSize, store_c);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);
  PaddedBuffer<float> c_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
      a_v(i) = i;
      b_v(i) = 2*i+1;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf);
  ir_eval(a_v, b_v, c_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i) << "index: " << i;
    EXPECT_EQ(b_v(i), 2*i+1) << "index: " << i;
    EXPECT_EQ(c_v(i), std::fmax(a_v(i), b_v(i))) << "index: " << i;
  }
}

TEST(ATenTest, minInt) {
  const int kTotalSize = 128;
  Buffer a_buf(Var("A", kHandle), kInt32, {Expr(kTotalSize)});
  Buffer b_buf(Var("B", kHandle), kInt32, {Expr(kTotalSize)});
  Buffer c_buf(Var("C", kHandle), kInt32, {Expr(kTotalSize)});

  Var index = Var("index", kInt32);
  Expr load_a = Load::make(
      a_buf,
      index,
      1);
  Expr load_b = Load::make(
      b_buf,
      index,
      1);
  Stmt store_c = Store::make(
      c_buf,
      index,
      Min::make(load_a, load_b, true),
      1);
  Stmt stmt = For::make(index, 0, kTotalSize, store_c);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<int> b_v(kTotalSize);
  PaddedBuffer<int> c_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
      a_v(i) = i;
      b_v(i) = 2*i+1;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf);
  ir_eval(a_v, b_v, c_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i) << "index: " << i;
    EXPECT_EQ(b_v(i), 2*i+1) << "index: " << i;
    EXPECT_EQ(c_v(i), std::min(a_v(i), b_v(i))) << "index: " << i;
  }
}

TEST(ATenTest, minFloat) {
  const int kTotalSize = 128;
  Buffer a_buf(Var("A", kHandle), kFloat32, {Expr(kTotalSize)});
  Buffer b_buf(Var("B", kHandle), kFloat32, {Expr(kTotalSize)});
  Buffer c_buf(Var("C", kHandle), kFloat32, {Expr(kTotalSize)});

  Var index = Var("index", kInt32);
  Expr load_a = Load::make(
      a_buf,
      index,
      1);
  Expr load_b = Load::make(
      b_buf,
      index,
      1);
  Stmt store_c = Store::make(
      c_buf,
      index,
      Min::make(load_a, load_b, true),
      1);
  Stmt stmt = For::make(index, 0, kTotalSize, store_c);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);
  PaddedBuffer<float> c_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
      a_v(i) = i;
      b_v(i) = 2*i+1;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf);
  ir_eval(a_v, b_v, c_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i) << "index: " << i;
    EXPECT_EQ(b_v(i), 2*i+1) << "index: " << i;
    EXPECT_EQ(c_v(i), std::fmin(a_v(i), b_v(i))) << "index: " << i;
  }
}

TEST(ATenTest, _sigmoid_backward) {
  const int kTotalSize = 128;
  Buffer a_buf(Var("A", kHandle), kFloat32, {Expr(kTotalSize)});
  Buffer b_buf(Var("B", kHandle), kFloat32, {Expr(kTotalSize)});
  Buffer c_buf(Var("C", kHandle), kFloat32, {Expr(kTotalSize)});

  Var index = Var("index", kInt32);
  Expr load_a = Load::make(
      a_buf,
      index,
      1);
  Expr load_b = Load::make(
      b_buf,
      index,
      1);
  Stmt store_c = Store::make(
      c_buf,
      index,
      load_a * load_b * (FloatImm::make(1.0f) - load_b),
      1);
  Stmt stmt = For::make(index, 0, kTotalSize, store_c);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);
  PaddedBuffer<float> c_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
      a_v(i) = i;
      b_v(i) = 2*i+1;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf);
  ir_eval(a_v, b_v, c_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i) << "index: " << i;
    EXPECT_EQ(b_v(i), 2*i+1) << "index: " << i;
    EXPECT_EQ(c_v(i), a_v(i) * b_v(i) * (1.0f - b_v(i))) << "index: " << i;
  }
}

TEST(ATenTest, _tanh_backward) {
  const int kTotalSize = 128;
  Buffer a_buf(Var("A", kHandle), kFloat32, {Expr(kTotalSize)});
  Buffer b_buf(Var("B", kHandle), kFloat32, {Expr(kTotalSize)});
  Buffer c_buf(Var("C", kHandle), kFloat32, {Expr(kTotalSize)});

  Var index = Var("index", kInt32);
  Expr load_a = Load::make(
      a_buf,
      index,
      1);
  Expr load_b = Load::make(
      b_buf,
      index,
      1);
  Stmt store_c = Store::make(
      c_buf,
      index,
      load_a * (FloatImm::make(1.0f) - (load_b * load_b)),
      1);
  Stmt stmt = For::make(index, 0, kTotalSize, store_c);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);
  PaddedBuffer<float> c_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
      a_v(i) = i;
      b_v(i) = 2*i+1;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf);
  ir_eval(a_v, b_v, c_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i) << "index: " << i;
    EXPECT_EQ(b_v(i), 2*i+1) << "index: " << i;
    EXPECT_EQ(c_v(i), a_v(i) * (1.0f - (b_v(i) * b_v(i)))) << "index: " << i;
  }
}

TEST(ATenTest, reciprocal) {
  const int kTotalSize = 128;
  Buffer a_buf(Var("A", kHandle), kFloat32, {Expr(kTotalSize)});
  Buffer b_buf(Var("B", kHandle), kFloat32, {Expr(kTotalSize)});

  Var index = Var("index", kInt32);
  Expr load_a = Load::make(
      a_buf,
      index,
      1);
  Stmt store_b = Store::make(
      b_buf,
      index,
      FloatImm::make(1.0f) / load_a,
      1);
  Stmt stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
      a_v(i) = i;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf);
  ir_eval(a_v, b_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i) << "index: " << i;
    EXPECT_EQ(b_v(i), 1.0f / i) << "index: " << i;
  }
}

TEST(ATenTest, reluInt) {
  const int kTotalSize = 128;
  Buffer a_buf(Var("A", kHandle), kInt32, {Expr(kTotalSize)});
  Buffer b_buf(Var("B", kHandle), kInt32, {Expr(kTotalSize)});

  Var index = Var("index", kInt32);
  Expr load_a = Load::make(
      a_buf,
      index,
      1);
  Stmt store_b = Store::make(
      b_buf,
      index,
      Max::make(load_a, 0, false),
      1);
  Stmt stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<int> b_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
      a_v(i) = i - 64;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf);
  ir_eval(a_v, b_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i - 64) << "index: " << i;
    EXPECT_EQ(b_v(i), std::max(a_v(i), 0)) << "index: " << i;
  }
}

TEST(ATenTest, reluFloat) {
  const int kTotalSize = 128;
  Buffer a_buf(Var("A", kHandle), kFloat32, {Expr(kTotalSize)});
  Buffer b_buf(Var("B", kHandle), kFloat32, {Expr(kTotalSize)});

  Var index = Var("index", kInt32);
  Expr load_a = Load::make(
      a_buf,
      index,
      1);
  Stmt store_b = Store::make(
      b_buf,
      index,
      Max::make(load_a, 0, false), // relu does not propagate nans
      1);
  Stmt stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
      a_v(i) = i - 64;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf);
  ir_eval(a_v, b_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i - 64) << "index: " << i;
    EXPECT_EQ(b_v(i), std::fmax(a_v(i), 0)) << "index: " << i;
  }
}
