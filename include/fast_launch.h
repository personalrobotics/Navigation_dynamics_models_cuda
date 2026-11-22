// Packed pinned buffer and CUDA graph cache shared by the rollout launchers
// (kinodynamic.cu, kinematic.cu, holonomic_6D.cu). Procedure: ../KERNELS.md.
#pragma once
#include <algorithm>
#include <cstdio>
#include <cuda_runtime.h>
#include <map>
#include <mutex>
#include <unordered_map>
#include <utility>

// One packed buffer per stream: [state | intermediates | cost | valid].
// H2D is state only (the kernel overwrites valid and cost). D2H is the pack.
// Host mirror is pinned (cudaMallocHost) so the graph can capture both copies.
struct PackView {
  size_t state_bytes = 0, inter_off = 0, cost_off = 0, valid_off = 0, total = 0;
};

inline PackView pack_view(int rollouts, int timesteps, int NX) {
  PackView v;
  v.state_bytes = sizeof(float) * size_t(rollouts) * size_t(NX);
  size_t inter =
      sizeof(float) * size_t(rollouts) * size_t(timesteps) * size_t(NX);
  v.inter_off = v.state_bytes;
  v.cost_off = v.inter_off + inter;
  v.valid_off = v.cost_off + sizeof(float) * size_t(rollouts);
  v.total = v.valid_off + sizeof(bool) * size_t(rollouts);
  return v;
}

struct Buf {
  void *d = nullptr;
  void *h = nullptr;
};

// One instance per environment. `Launch` is that environment's kernel-argument
// struct; a graph keeps a copy so it can be updated when the map changes.
template <class Launch> struct FastLaunch {
  struct Entry {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaGraphNode_t dyn = nullptr;
    cudaGraphNode_t valid_node = nullptr;
    Launch args;
  };
  struct Slot {
    int cap = 0;
    bool mapped = false; // host alloc is also the device buffer
    void *d = nullptr;
    void *h = nullptr;
  };

  const char *name;
  std::mutex mu;
  std::unordered_map<cudaStream_t, Slot> slots;
  std::map<std::pair<void *, int>, Entry> graphs; // (device buffer, rollouts)
  bool broken = false;

  explicit FastLaunch(const char *n) : name(n) {}

  // Graphs that point at a freed device buffer must go.
  void drop_graphs_locked(void *d_base) {
    if (!d_base)
      return;
    for (auto it = graphs.begin(); it != graphs.end();) {
      if (it->first.first == d_base) {
        if (it->second.exec)
          cudaGraphExecDestroy(it->second.exec);
        if (it->second.graph)
          cudaGraphDestroy(it->second.graph);
        it = graphs.erase(it);
      } else {
        ++it;
      }
    }
  }

  // ponytail: a slot grows to the largest rollout count seen on that stream and
  // drops its graphs when it grows. Per-size slots if batch sizes keep growing.
  // Copies the device pointers out; the Slot can move if the map rehashes.
  // Do not switch this to cudaMallocManaged. On Orin, 10k best_f was about 3x
  // slower than this pinned pack (IGHA* 11397 vs 3519 exp/s), a CPU write into
  // the pointer segfaults, and prefetch returned invalid device ordinal.
  bool ensure_slot(cudaStream_t stream, int rollouts, int timesteps, int NX,
                   Buf *out, bool mapped = false) {
    std::lock_guard<std::mutex> lock(mu);
    Slot &s = slots[stream];
    if (!(s.d && s.cap >= rollouts && s.mapped == mapped)) {
      int cap = std::max(rollouts, s.cap);
      drop_graphs_locked(s.d);
      // Mapped slots have one allocation: d is an alias of h.
      if (s.mapped) {
        if (s.h)
          cudaFreeHost(s.h);
      } else {
        if (s.d)
          cudaFree(s.d);
        if (s.h)
          cudaFreeHost(s.h);
      }
      s.d = s.h = nullptr;
      s.cap = 0;
      s.mapped = false;
      size_t bytes = pack_view(cap, timesteps, NX).total;
      cudaError_t err = cudaSuccess;
      if (mapped) {
        err = cudaHostAlloc(&s.h, bytes, cudaHostAllocMapped);
        if (err == cudaSuccess)
          err = cudaHostGetDevicePointer(&s.d, s.h, 0);
      } else if ((err = cudaMalloc(&s.d, bytes)) == cudaSuccess) {
        err = cudaMallocHost(&s.h, bytes);
      }
      if (err != cudaSuccess) {
        std::fprintf(stderr, "[%s] pack alloc failed: %s\n", name,
                     cudaGetErrorString(err));
        if (!mapped && s.d)
          cudaFree(s.d);
        if (s.h)
          cudaFreeHost(s.h);
        s.d = s.h = nullptr;
        return false;
      }
      s.cap = cap;
      s.mapped = mapped;
    }
    out->d = s.d;
    out->h = s.h;
    return true;
  }
};
