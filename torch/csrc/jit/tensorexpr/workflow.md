# Current workflow

## Step 1: input from the user.

User construct a kernel from tensor expressions, like:
```
    Buffer a_buf("a", kFloat32, {M, N});
    Buffer b_buf("b", kFloat32, {N, K});
    Buffer c_buf("c", kFloat32, {M, N});
    Buffer d_buf("d", kFloat32, {M, K});

    Tensor* x = Compute(
        "x",
        {{M, "m1"}, {N, "n1"}, {K, "k1"}},
        [&](const Var& m, const Var& n, const Var& k) {
          return a_buf(m, n) * b_buf(n, k);
        });
    Tensor* y = ...;
    Tensor* z = ...;
    std::vector<Tensor*> tensors_to_compute = {x, z}; // Tensor y might be used in x or z - in this case it will also be computed. 
```

## Step 2: Create schedule for the tensor expressions:
```
   Schedule s(tensors_to_compute);
```
This constructs a tree-like data structure (`TensorExprNode`) representing loop nests for the given tensor computation.
A node in this IR is either a loop-axis(LoopAxis) or a tensor expression (`TensorExprOp`).
If it is a loop-axis, it also contains children that again might be either a loop-axes or a tensor expression, and so on.
If it is a tensor-expression, it is lowered to a statement (`Stmt`). Currently, it just means that we're creating a `Store` for every tensor-expression. We also keep a pointer to the original tensor expression.
It could look like this:
```
loop-axis i
  loop-axis j
    Store(to: a[i, j], what: x[i] + y[j])
loop-axis k
  loop-axis l
    Store(to: b[k, l], what: a[i, j] + 1)
    loop-axis m
      Store(to: c[k,l,m], what: b[k,l] + z[m])
```

## Step 3: Apply scheduling primitives
Scheduling primitives mutate the tree structure: they can create or remove loop-axis, replace statements with other statements (updates `element_stmt` for each affected tensor expression) or remove them. The transformations also record the history.
The output of this step is a modified tree-like structure (same format as in step 2).

## Step 4: Lower the tree structure to statements.
This step creates a `For` statement for each loop-axis and emits `element_stmt` for bodies of the loops.

## Step 5: Pass the final statement for codegen (LLVM/CUDA/IREval)
Codegen is implemented as an IR visitor over the statements produced in the previous step.
