#include "matmul.h"

#include <cstdlib>
#include <cstring>

#include "matmul_backend.h"

#ifdef _WIN32
#include <malloc.h>
static void *alloc_aligned(std::size_t size) { return _aligned_malloc(size, 64); }
static void free_aligned(void *ptr) { _aligned_free(ptr); }
#else
static void *alloc_aligned(std::size_t size) {
  void *ptr = nullptr;
  if (posix_memalign(&ptr, 64, size) != 0) {
    return nullptr;
  }
  return ptr;
}
static void free_aligned(void *ptr) { std::free(ptr); }
#endif

namespace {

// Reference CPU matmul backend. Uses the same padded operand layout as the
// RKNN backend so all callers keep working unchanged.
struct CpuMatmulBackend : public MatmulBackend {
  void allocate(Matmul *m, int core) override {
    (void)core;
    m->A = static_cast<__fp16 *>(alloc_aligned(sizeof(__fp16) * m->M * m->K_padded));
    m->B = static_cast<__fp16 *>(
        alloc_aligned(sizeof(__fp16) * m->K_padded * m->N_padded));
    m->C = static_cast<float *>(alloc_aligned(sizeof(float) * m->M * m->N_padded));
  }

  void call(Matmul *m) override {
    const int M = m->M;
    const int K = m->K;
    const int N = m->N;
    const int K_padded = m->K_padded;
    const __fp16 *A = m->A;
    const __fp16 *B = m->B;
    float *C = m->C;

#pragma omp parallel for
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
          acc += static_cast<float>(A[((k / 8) * M + i) * 8 + (k % 8)]) *
                 static_cast<float>(B[(((j / 16) * (K_padded / 32) + (k / 32)) *
                                       16 +
                                       (j % 16)) *
                                          32 +
                                      (k % 32)]);
        }
        C[((j / 4) * M + i) * 4 + (j % 4)] = acc;
      }
    }
  }

  void deallocate(Matmul *m) override {
    free_aligned(m->A);
    free_aligned(m->B);
    free_aligned(m->C);
    m->A = nullptr;
    m->B = nullptr;
    m->C = nullptr;
  }

  const char *name() const override { return "cpu"; }
};

}  // namespace

MatmulBackend *create_cpu_backend() { return new CpuMatmulBackend(); }