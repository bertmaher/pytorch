import framework


class ElementMulBench(framework.Benchmark):
    def __init__(self, mode, device, N):
        super().__init__(mode, device)
        self.N = N
        self.d1 = self.rand([N], device=device, requires_grad=self.requires_grad)
        self.d2 = self.rand([N], device=device, requires_grad=self.requires_grad)

    def forward(self):
        y = self.mul(self.d1, self.d2)
        return y

    def reference(self):
        return self.numpy(self.d1) * self.numpy(self.d2)

    def config(self):
        return [self.N]

    @staticmethod
    def module():
        return 'element_mul'

    def memory_workload(self):
        if self.mode == 'fwd':
            sol_count = 2 + 1
            algorithmic_count = 2 + 1
        else:
            sol_count = (2 + 1) + (1 + 2)
            algorithmic_count = (2 + 1) + ((2 + 1) + (2 + 1))

        buffer_size = self.N * 4
        return {'sol': buffer_size * sol_count, 'algorithmic': buffer_size * algorithmic_count}

    @staticmethod
    def default_configs():
        return [[1 << 27]]


framework.register_benchmark_class(ElementMulBench)
