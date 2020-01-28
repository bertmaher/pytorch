import torch
import numpy as np

def test_easy():
    def easy(x, y):
        aaa = torch.add(x, y)
        return aaa

    traced = torch.jit.trace(easy, (torch.rand(1024), torch.rand(1024)))

    a = torch.rand(1024)
    b = torch.rand(1024)
    x = traced(a, b)
    np.testing.assert_allclose(a.numpy() + b.numpy(), x.numpy())

def test_three_arg():
    def easy(x, y, z):
        aaa = torch.add(x, y)
        bbb = torch.add(aaa, z)
        return bbb

    traced = torch.jit.trace(easy, (torch.rand(1024), torch.rand(1024), torch.rand(1024)))

    a = torch.rand(1024)
    b = torch.rand(1024)
    c = torch.rand(1024)
    x = traced(a, b, c)
    npr = a.numpy() + b.numpy() + c.numpy()
    np.testing.assert_allclose(npr, x.numpy())

def test_all_combos():
    def easy(x, y, z):
        a = torch.add(x, y)
        b = torch.add(a, z)
        c = torch.add(x, b)
        d = torch.add(c, a)
        return d

    def np_easy(x, y, z):
        a = x + y
        b = a + z
        c = x + b
        d = c + a
        return d

    traced = torch.jit.trace(easy, (torch.rand(1024), torch.rand(1024), torch.rand(1024)))

    a = torch.rand(1024)
    b = torch.rand(1024)
    c = torch.rand(1024)
    x = traced(a, b, c)
    npr = np_easy(a.numpy(), b.numpy(), c.numpy())
    np.testing.assert_allclose(npr, x.numpy())

def test_rank_two():
    def easy(x, y, z):
        a = torch.add(x, y)
        b = torch.add(a, z)
        c = torch.add(x, b)
        d = torch.add(c, a)
        return d

    def np_easy(x, y, z):
        a = x + y
        b = a + z
        c = x + b
        d = c + a
        return d

    shape = 32, 32
    traced = torch.jit.trace(easy, (torch.rand(shape), torch.rand(shape), torch.rand(shape)))

    a = torch.rand(shape)
    b = torch.rand(shape)
    c = torch.rand(shape)
    x = traced(a, b, c)
    npr = np_easy(a.numpy(), b.numpy(), c.numpy())
    np.testing.assert_allclose(npr, x.numpy())

def test_broadcast():
    def easy(x, y, z):
        a = torch.add(x, y)
        b = torch.add(a, z)
        return b

    def np_easy(x, y, z):
        a = x + y
        b = a + z
        return b

    N = 32
    traced = torch.jit.trace(easy, (torch.rand(N, N), torch.rand(N), torch.rand(N, N)))

    a = torch.rand(N, N)
    b = torch.rand(N)
    c = torch.rand(N, N)
    x = traced(a, b, c)
    npr = np_easy(a.numpy(), b.numpy(), c.numpy())
    np.testing.assert_allclose(npr, x.numpy())

def test_broadcast_2():
    zero = torch.tensor([0.0], dtype=torch.float)

    def foo(x, y, z):
        aaa = torch.add(x, y)
        bbb = torch.add(zero, aaa)
        return torch.add(bbb, z)

    def foo_np(x, y, z):
        a = x + y
        b = zero.numpy() + a
        return b + z

    x = torch.rand(3, 4)
    y = torch.ones(3, 1)
    z = torch.rand(4)
    traced = torch.jit.trace(foo, (x, y, z))

    r = traced(x, y, z)
    rnp = foo_np(x.numpy(), y.numpy(), z.numpy())
    np.testing.assert_allclose(r, rnp)

def test_broadcast_big2():
    zero = torch.tensor([0.0], dtype=torch.float)

    def foo(x, y, z):
        aaa = torch.add(x, y)
        bbb = torch.add(zero, aaa)
        return torch.add(bbb, z)

    def foo_np(x, y, z):
        a = x + y
        b = zero.numpy() + a
        return b + z

    x = torch.rand(32, 1024)
    y = torch.ones(32, 1)
    z = torch.rand(1024)
    traced = torch.jit.trace(foo, (x, y, z))

    r = traced(x, y, z)
    rnp = foo_np(x.numpy(), y.numpy(), z.numpy())
    np.testing.assert_allclose(r, rnp)

def test_alpha():
    def alpha(x):
        aaa = torch.add(x, x, alpha=2.0)
        return aaa

    traced = torch.jit.trace(alpha, (torch.tensor([1.0])))

    a = torch.tensor([1.0])
    x = traced(a)
    np.testing.assert_allclose(a.numpy() + 2.0 * a.numpy(), x.numpy())

def test_constant():
    def constant(x):
        bbb = torch.tensor([1.0])
        aaa = torch.add(x, bbb)
        return aaa

    traced = torch.jit.trace(constant, (torch.tensor([1.0])))

    a = torch.tensor([1.0])
    x = traced(a)
    np.testing.assert_allclose(a.numpy() + 1.0, x.numpy())

def test_add_sub():
    def easy(x, y, z):
        aaa = torch.add(x, y)
        bbb = torch.sub(aaa, z)
        return bbb

    traced = torch.jit.trace(easy, (torch.rand(1024), torch.rand(1024), torch.rand(1024)))

    a = torch.rand(1024)
    b = torch.rand(1024)
    c = torch.rand(1024)
    x = traced(a, b, c)
    np.testing.assert_allclose(a.numpy() + b.numpy() - c.numpy(), x.numpy())

def test_promotion():
    def easy(x, y):
        aaa = torch.add(x, y)
        return aaa

    traced = torch.jit.trace(easy, (torch.zeros(1024, dtype=torch.int32), torch.rand(1024, dtype=torch.float32)))

    a = torch.zeros(1024, dtype=torch.int32)
    b = torch.rand(1024, dtype=torch.float32)
    x = traced(a, b)
    np.testing.assert_allclose(a.numpy() + b.numpy(), x.numpy())

def test_eq():
    def easy(x, y):
        c = torch.eq(x, y)
        return c

    traced = torch.jit.trace(easy, (torch.zeros(1024), torch.zeros(1024)))
    a = torch.zeros(1024, dtype=torch.int32)
    b = torch.zeros(1024, dtype=torch.int32)
    x= traced(a, b)
    np.testing.assert_allclose(np.ones(1024), x.numpy())

def test_ne():
    def easy(x, y):
        c = torch.ne(x, y)
        return c

    traced = torch.jit.trace(easy, (torch.zeros(1024), torch.zeros(1024)))
    a = torch.zeros(1024, dtype=torch.int32)
    b = torch.ones(1024, dtype=torch.int32)
    x= traced(a, b)
    np.testing.assert_allclose(np.ones(1024), x.numpy())

def test_ge():
    def easy(x, y):
        c = torch.ge(x, y)
        return c

    traced = torch.jit.trace(easy, (torch.zeros(1024), torch.zeros(1024)))
    aa = np.array(1024, dtype=int)
    aa.fill(5)
    a = torch.from_numpy(aa)
    b = torch.zeros(1024, dtype=torch.int32)
    x= traced(a,b)
    np.testing.assert_allclose(np.ones(1024), x.numpy())

def test_gt():
    def easy(x, y):
        c = torch.gt(x, y)
        return c

    traced = torch.jit.trace(easy, (torch.zeros(1024), torch.zeros(1024)))
    a = torch.ones(1024, dtype=torch.int32)
    b = torch.zeros(1024, dtype=torch.int32)
    x= traced(a, b)
    np.testing.assert_allclose(np.ones(1024), x.numpy())

def test_le():
    def easy(x, y):
        c = torch.le(x, y)
        return c

    traced = torch.jit.trace(easy, (torch.zeros(1024), torch.zeros(1024)))
    aa = np.array(1024, dtype=int)
    aa.fill(5)
    a = torch.from_numpy(aa)
    b = torch.zeros(1024, dtype=torch.int32)
    x= traced(a, b)
    np.testing.assert_allclose(np.zeros(1024), x.numpy())

def test_lt():
    def easy(x, y):
        c = torch.lt(x, y)
        return c

    traced = torch.jit.trace(easy, (torch.zeros(1024), torch.zeros(1024)))
    a = torch.ones(1024, dtype=torch.int32)
    b = torch.zeros(1024, dtype=torch.int32)
    x= traced(a, b)
    np.testing.assert_allclose(np.zeros(1024), x.numpy())

def test_reps():
    def easy(x, y):
        c = torch.add(x, y)
        return c

    traced = torch.jit.trace(easy, (torch.rand(1024), torch.rand(1024)))

    for _ in range(32):
        a = torch.ones(1024)
        b = torch.zeros(1024)
        x = traced(a, b)
        np.testing.assert_allclose(np.ones(1024), x.numpy())

def test_add_const_rhs():
    def test(x):
        return x + 3.0
    traced = torch.jit.trace(test, torch.rand(4))
    x = torch.rand(4)
    y = traced(x)
    np.testing.assert_allclose(x.numpy() + 3.0, y.numpy())

def test_int_output():
    def test(x, y, z):
        return x * y * z
    xs = [(torch.rand(4) * 3 + 1).to(torch.int32) for i in range(3)]
    x, y, z = xs
    xn, yn, zn = [t.numpy() for t in xs]
    traced = torch.jit.trace(test, (x, y, z))
    res = traced(x, y, z)
    np.testing.assert_allclose(xn * yn * zn, res.numpy())
