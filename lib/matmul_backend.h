#ifndef _LIB_MATMUL_BACKEND_H_
#define _LIB_MATMUL_BACKEND_H_

struct Matmul;

// A backend provides the execution for a single Matmul. It owns the backing
// storage for the A, B and C operands as well as any backend-specific state
// (e.g. an RKNN matmul context and NPU memory).
//
// Each Matmul constructs its own backend instance through
// create_matmul_backend(). After allocate() the backend has set m->A, m->B
// and m->C to point at logical buffers in the layout documented in matmul.h;
// call() reads those logical pointers and computes C = A * B in-place.
struct MatmulBackend {
  virtual ~MatmulBackend() = default;

  // Allocate backing storage, set m->A/B/C to the logical buffers and stash
  // any backend-specific state. Called once while the Matmul is being
  // constructed.
  virtual void allocate(Matmul *m, int core) = 0;

  // Compute C = A * B into the logical buffer referenced by m->C.
  virtual void call(Matmul *m) = 0;

  // Release every resource held by the backend and null out m->A/B/C.
  virtual void deallocate(Matmul *m) = 0;

  virtual const char *name() const = 0;
};

MatmulBackend *create_cpu_backend();
#ifdef HAVE_RKNN
MatmulBackend *create_rknn_backend();
#endif

// Factory. Selects a backend from the USEFUL_TRANSFORMERS_BACKEND environment
// variable ("cpu" or "rknn"). Defaults to "rknn" when built with RKNN support,
// otherwise "cpu". Each call returns a freshly allocated backend.
MatmulBackend *create_matmul_backend();

#endif  // _LIB_MATMUL_BACKEND_H_