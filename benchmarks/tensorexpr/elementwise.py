import framework


class ElementMulBench(framework.Benchmark):
    def __init__(self, mode, device, N):
        super().__init__(mode, device)
        self.N = N
        self.d1 = self.rand([N], device=device, requires_grad=self.requires_grad)
        self.d2 = self.rand([N], device=device, requires_grad=self.requires_grad)
        self.d3 = self.rand([N], device=device, requires_grad=self.requires_grad)
        self.d4 = self.rand([N], device=device, requires_grad=self.requires_grad)
        self.inputs = [self.d1, self.d2, self.d3, self.d4]

    def forward(self, d1, d2, d3, d4):
        y = d1 * d2 + d3 * d4
        return y

    def reference(self):
        return self.numpy(self.d1) * self.numpy(self.d2) + self.numpy(self.d3) * self.numpy(self.d4)

    def config(self):
        return [self.N]

    @staticmethod
    def module():
        return 'element_mul'

    def memory_workload(self):
        if self.mode == 'fwd':
            sol_count = 4 + 1
            algorithmic_count = 3 + 1
        else:
            sol_count = (4 + 1) + (1 + 4)
            algorithmic_count = (4 + 1) + ((2 + 1) * 4)

        buffer_size = self.N * 4
        return {'sol': buffer_size * sol_count, 'algorithmic': buffer_size * algorithmic_count}

    @staticmethod
    def default_configs():
        return [[1 << 27]]


framework.register_benchmark_class(ElementMulBench)
