#include <ATen/ATen.h>
#include <ATen/cpu/vec/vec.h>
#include <benchmark/benchmark.h>

void __attribute__((noinline)) loadu(float* __restrict__ ap, int64_t an) {
  benchmark::DoNotOptimize(at::vec::Vectorized<float>::loadu(ap, an));
}

static void Loadu(benchmark::State& state) {
  auto a = at::randn({7});
  auto ap = a.data_ptr<float>();
  auto an = a.numel();
  for (auto _ : state) {
    loadu(ap, an);
  }
  auto b = at::empty({7});
  auto vec = at::vec::Vectorized<float>::loadu(ap, an);
  vec.store(b.data_ptr<float>(), b.numel());
  TORCH_INTERNAL_ASSERT(at::equal(a, b));
}
BENCHMARK(Loadu);
BENCHMARK_MAIN();
