import framework


class BroadcastMulBench(framework.Benchmark):
    def __init__(self, mode, device, case, M, N, K):
        super().__init__(mode, device)
        self.case = case
        self.M = M
        self.N = N
        self.K = K

        if case == 'row':
            self.d1 = self.rand([M, N, 1], device=device, requires_grad=self.requires_grad)
            self.d2 = self.rand([M, 1, K], device=device, requires_grad=self.requires_grad)
        elif case == 'mid':
            self.d1 = self.rand([M, N, 1], device=device, requires_grad=self.requires_grad)
            self.d2 = self.rand([1, N, K], device=device, requires_grad=self.requires_grad)
        elif case == 'col':
            self.d1 = self.rand([M, 1, K], device=device, requires_grad=self.requires_grad)
            self.d2 = self.rand([1, N, K], device=device, requires_grad=self.requires_grad)
        else:
            raise ValueError('invalid case: %s' % (case))

    def forward(self):
        y = self.d1 + self.d2
        return y

    def reference(self):
        return self.numpy(self.d1) + self.numpy(self.d2)

    def config(self):
        return [self.M, self.N, self.K]

    @staticmethod
    def default_configs():
        return [[128, 256, 128]]

    def memory_workload(self):
        if self.mode == 'fwd':
            sol_count = 1
            algorithmic_count = 1
        else:
            sol_count = (1) + (1)
            algorithmic_count = 1 + (1 + 1)

        buffer_size = self.M * self.N * self.K * 4
        return {'sol': buffer_size * sol_count, 'algorithmic': buffer_size * algorithmic_count}

    
class BroadcastRowBench(BroadcastMulBench):
    def __init__(self, mode, device, M, N, K):
        super(BroadcastRowBench, self).__init__(mode, device, 'row', M, N, K)

    @staticmethod
    def module():
        return 'broadcast_row'

    
class BroadcastMidBench(BroadcastMulBench):
    def __init__(self, mode, device, M, N, K):
        super(BroadcastMidBench, self).__init__(mode, device, 'mid', M, N, K)

    @staticmethod
    def module():
        return 'broadcast_mid'

    
class BroadcastColBench(BroadcastMulBench):
    def __init__(self, mode, device, M, N, K):
        super(BroadcastColBench, self).__init__(mode, device, 'col', M, N, K)

    @staticmethod
    def module():
        return 'broadcast_col'


framework.register_benchmark_class(BroadcastRowBench)
framework.register_benchmark_class(BroadcastMidBench)
framework.register_benchmark_class(BroadcastColBench)
