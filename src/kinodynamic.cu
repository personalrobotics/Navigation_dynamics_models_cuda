#include "fast_launch.h"
#include "terrain_map.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#define x_index 0
#define y_index 1
#define yaw_index 2
#define vx_index 3

#define st_index 0
#define th_index 1
#define GRAVITY 9.81f

// Unroll count comes from node_info.timesteps via -DIGHASTAR_TIMESTEPS
// (kernel_cflags in ighastar/scripts/common_utils.py). A new value rebuilds.
#ifndef IGHASTAR_TIMESTEPS
#error "IGHASTAR_TIMESTEPS is not set. Build through create_planner or add kernel_cflags()."
#endif
constexpr int kTimesteps = IGHASTAR_TIMESTEPS;
constexpr int kNX = 4, kNC = 2; // match n_dims, n_cont in include/kinodynamic.h

__device__ float nan_to_num(float x, float replace) {
  return (std::isnan(x) || std::isinf(x)) ? replace : x;
}

__device__ float clamp(float x, float lower, float upper) {
  return std::min(std::max(x, lower), upper);
}

__device__ float wrap_to_pi(float x) {
  return std::fmod(x + M_PI, 2 * M_PI) - M_PI;
}

// Maximum number of steps the ESDF column march is allowed to take. The march
// makes at least one voxel of progress per step, and typically converges in
// two or three, so this is only a safety net.
#define MAX_MARCH_STEPS 32

// Index of the ESDF column holding world coordinates (x, y) (device function)
__device__ int esdf_column(float x, float y, const TerrainMap &map) {
  int img_X = clamp(static_cast<int>((x * map.res_inv)), 0, map.nx - 1);
  int img_Y = clamp(static_cast<int>((y * map.res_inv)), 0, map.ny - 1);
  return (img_Y * map.nx + img_X) * map.nz;
}

// Finds the terrain surface in an ESDF column: march down from the top of the
// column, stepping by the signed distance (which never exceeds the vertical
// gap to the surface, so we cannot step past it), then linearly interpolate
// between the two voxels straddling the zero crossing (device function).
__device__ float esdf_surface_z(float x, float y, const TerrainMap &map) {
  const int column = esdf_column(x, y, map);
  int k = map.nz - 1;
  float d = map.esdf[column + k];
  for (int step = 0; step < MAX_MARCH_STEPS && d > 0.0f && k > 0; step++) {
    k = max(k - max(1, static_cast<int>(d / map.voxel_z)), 0);
    d = map.esdf[column + k];
  }
  float d_lo = d;
  float d_hi = map.esdf[column + min(k + 1, map.nz - 1)];
  float delta = d_hi - d_lo;
  float frac = (fabsf(delta) > 1e-12f) ? (-d_lo / delta) : 0.0f;
  frac = clamp(frac, 0.0f, 1.0f);
  return map.z_min + (float(k) + frac) * map.voxel_z;
}

// Maps world coordinates to terrain height, from whichever representation the
// world was loaded with (device function)
__device__ float map_to_elev(float x, float y, const TerrainMap &map) {
  if (map.type == MAP_ELEVATION) {
    int img_X = clamp(static_cast<int>((x * map.res_inv)), 0, map.nx - 1);
    int img_Y = clamp(static_cast<int>((y * map.res_inv)), 0, map.ny - 1);
    return map.elev[img_Y * map.nx + img_X];
  }
  return esdf_surface_z(x, y, map);
}

// Maps world coordinates to a traversability value on the same 0-255 scale as
// the costmap: 255 is free, 0 is an obstacle (device function)
__device__ float map_to_cost(float x, float y, const TerrainMap &map) {
  if (map.type == MAP_ELEVATION) {
    int img_X = clamp(static_cast<int>((x * map.res_inv)), 0, map.nx - 1);
    int img_Y = clamp(static_cast<int>((y * map.res_inv)), 0, map.ny - 1);
    return map.cost[img_Y * map.nx + img_X];
  }
  // ESDF: the colour of the voxel at the recovered surface, as luminance.
  const int column = esdf_column(x, y, map);
  int k = clamp(static_cast<int>((esdf_surface_z(x, y, map) - map.z_min) /
                                 map.voxel_z),
                0, map.nz - 1);
  uchar4 c = map.color[column + k];
  return 0.299f * float(c.x) + 0.587f * float(c.y) + 0.114f * float(c.z);
}

// Computes the 3D footprint coordinates and center height of the car (device
// function)
__device__ void get_footprint_z(float *fl, float *fr, float *bl, float *br,
                                float &z, float x, float y, float cy, float sy,
                                const TerrainMap &map, float car_l2,
                                float car_w2) {
  fl[0] = car_l2 * cy - car_w2 * sy + x;
  fl[1] = car_l2 * sy + car_w2 * cy + y;

  fr[0] = car_l2 * cy - (-1) * car_w2 * sy + x;
  fr[1] = car_l2 * sy + (-1) * car_w2 * cy + y;

  bl[0] = (-1) * car_l2 * cy - car_w2 * sy + x;
  bl[1] = (-1) * car_l2 * sy + car_w2 * cy + y;

  br[0] = (-1) * car_l2 * cy - (-1) * car_w2 * sy + x;
  br[1] = (-1) * car_l2 * sy + (-1) * car_w2 * cy + y;

  float z_cent = map_to_elev(0, 0, map);
  z = map_to_elev(x, y, map) - z_cent;

  fl[2] = map_to_elev(fl[0], fl[1], map) - z_cent;
  fr[2] = map_to_elev(fr[0], fr[1], map) - z_cent;
  bl[2] = map_to_elev(bl[0], bl[1], map) - z_cent;
  br[2] = map_to_elev(br[0], br[1], map) - z_cent;
}

/*
list of constants:
    NX = 4
    NC = 2
    timesteps = 10
    n_succ = 1000
    patch_length_px = 20
    patch_width_px = 20
    map_size_px = 1000
    map_res = 0.1
    car_l2 = 1.0
    car_w2 = 0.5
*/
/*
list of reused variables:
    map = TerrainMap (costmap or ESDF colour layer)
    d_intermediate_states = d_intermediate_states
    patch_length_px = patch_length_px
    patch_width_px = patch_width_px
    car_l2 = car_l2
    car_w2 = car_w2
    d_valid = d_valid
*/

// Checks validity of multiple states against the traversability layer (CUDA
// kernel)
__global__ void
check_validity_batch_kernel(TerrainMap map, float *d_intermediate_states,
                            int patch_length_px, int patch_width_px,
                            float car_l2, float car_w2, int NX, int timesteps,
                            bool *d_valid) {
  int t = blockIdx.x;
  int k = blockIdx.y;
  int i = threadIdx.x;
  int j = threadIdx.y;

  if (i >= patch_length_px || j >= patch_width_px)
    return;
  if (!(d_valid[k]))
    return; // already invalid

  int intermediate_index = k * timesteps * NX + t * NX;

  float x = d_intermediate_states[intermediate_index + x_index];
  float y = d_intermediate_states[intermediate_index + y_index];
  float theta = d_intermediate_states[intermediate_index + yaw_index];

  float cy = cosf(theta);
  float sy = sinf(theta);
  float offset_x = (i * map.voxel_xy) - car_l2;
  float offset_y = (j * map.voxel_xy) - car_w2;

  float px = offset_x * cy - offset_y * sy + x;
  float py = offset_x * sy + offset_y * cy + y;

  if (px < 0 || px >= map.nx * map.voxel_xy || py < 0 ||
      py >= map.ny * map.voxel_xy) {
    d_valid[k] = false;
    return;
  }

  if (map_to_cost(px, py, map) <= 250.0f) {
    d_valid[k] = false;
  }
}

// kTimesteps is a compile-time constant (IGHASTAR_TIMESTEPS), so the step loop
// unrolls. `ok` stays in a register; the old kernel stored valid[k] every step.
__global__ void kinodynamic_kernel_unrolled(
    float *state, float *intermediate_states, float *controls, TerrainMap map,
    bool *valid, float *cost, float dt, int rollouts, float car_l2,
    float car_w2, float max_vel, float min_vel, float RI, float max_vert_acc,
    float max_theta, float gear_switch_time) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= rollouts)
    return;

  int state_base = k * kNX;
  float x = state[state_base + x_index];
  float y = state[state_base + y_index];
  float yaw = state[state_base + yaw_index];
  float vx = state[state_base + vx_index];
  float vz = 0.0f, vy = 0.0f;
  float wz = 0.0f;

  float cy = cosf(yaw), sy = sinf(yaw);
  float fl[3], fr[3], bl[3], br[3], z;
  get_footprint_z(fl, fr, bl, br, z, x, y, cy, sy, map, car_l2, car_w2);
  float last_roll = atan2f((fl[2] + bl[2]) - (fr[2] + br[2]), 4 * car_w2);
  float last_pitch = atan2f((bl[2] + br[2]) - (fl[2] + fr[2]), 4 * car_l2);
  float roll, pitch, wx, wy, cp, sp, cr, sr, ay, az;
  float initial_vx = vx;
  bool ok = true;

#pragma unroll
  for (int t = 1; t <= kTimesteps; t++) {
    int control_base = k * kTimesteps * kNC + (t - 1) * kNC;
    float curvature = controls[control_base + st_index];
    float ax = controls[control_base + th_index];
    wz = curvature * vx;

    cy = cosf(yaw);
    sy = sinf(yaw);
    get_footprint_z(fl, fr, bl, br, z, x, y, cy, sy, map, car_l2, car_w2);
    roll = atan2f((fl[2] + bl[2]) - (fr[2] + br[2]), 4 * car_w2);
    pitch = atan2f((bl[2] + br[2]) - (fl[2] + fr[2]), 4 * car_l2);

    wx = (roll - last_roll) / dt;
    wy = (pitch - last_pitch) / dt;
    last_pitch = pitch;
    last_roll = roll;

    cp = cosf(pitch), sp = sinf(pitch);
    cr = cosf(roll), sr = sinf(roll);

    vx += (ax * cp + sp * GRAVITY) * dt;
    ay = vx * wz - sr * GRAVITY;
    az = GRAVITY * cp * cr - vx * wy + wx * wx * car_w2;
    wz = curvature * vx;

    yaw = wrap_to_pi(yaw + wz * dt);
    cy = cosf(yaw);
    sy = sinf(yaw);

    x += dt * (vx * (cp * cy) + vy * (sr * sp * cy - cr * sy) +
               vz * (cr * sp * cy + sr * sy));
    y += dt * (vx * (cp * sy) + vy * (sr * sp * sy + cr * cy) +
               vz * (cr * sp * sy - sr * cy));

    ok = ok && fabsf(az - GRAVITY) < max_vert_acc;
    ok = ok && fabsf(ay / az) < RI;
    ok = ok && vx > min_vel && vx < max_vel;
    ok = ok && fabsf(pitch) < max_theta && fabsf(roll) < max_theta;

    int intermediate_index = k * kTimesteps * kNX + (t - 1) * kNX;
    intermediate_states[intermediate_index + x_index] = x;
    intermediate_states[intermediate_index + y_index] = y;
    intermediate_states[intermediate_index + yaw_index] = yaw;
    intermediate_states[intermediate_index + vx_index] = vx;
  }
  valid[k] = ok;
  cost[k] = fabsf(kTimesteps * dt) + gear_switch_time * (vx * initial_vx < 0);

  state[state_base + x_index] = x;
  state[state_base + y_index] = y;
  state[state_base + yaw_index] = yaw;
  state[state_base + vx_index] = vx;
}

struct KinoLaunch {
  void *d_base = nullptr;
  void *h_base = nullptr;
  size_t state_bytes = 0;
  size_t total_bytes = 0;
  float *d_state = nullptr;
  float *d_inter = nullptr;
  float *d_controls = nullptr;
  float *d_cost = nullptr;
  bool *d_valid = nullptr;
  TerrainMap map;
  float dt = 0;
  int rollouts = 0;
  int nx = 4;
  int ts = 5;
  int patch_length_px = 0, patch_width_px = 0;
  int blocks = 0, threads = 0;
  float car_l2 = 0, car_w2 = 0, max_vel = 0, min_vel = 0, RI = 0,
        max_vert_acc = 0, max_theta = 0, gear_switch_time = 0;
  bool zero_copy = false; // pack is cudaHostAllocMapped; no H2D/D2H nodes
};

static bool same_map(const TerrainMap &a, const TerrainMap &b) {
  return a.type == b.type && a.elev == b.elev && a.cost == b.cost &&
         a.esdf == b.esdf && a.color == b.color && a.nx == b.nx &&
         a.ny == b.ny && a.nz == b.nz && a.voxel_xy == b.voxel_xy &&
         a.voxel_z == b.voxel_z && a.z_min == b.z_min && a.res_inv == b.res_inv;
}

static FastLaunch<KinoLaunch> g("kinodynamic");

static void dyn_params(cudaKernelNodeParams *p, KinoLaunch *a, void **kp) {
  kp[0] = &a->d_state;
  kp[1] = &a->d_inter;
  kp[2] = &a->d_controls;
  kp[3] = &a->map;
  kp[4] = &a->d_valid;
  kp[5] = &a->d_cost;
  kp[6] = &a->dt;
  kp[7] = &a->rollouts;
  kp[8] = &a->car_l2;
  kp[9] = &a->car_w2;
  kp[10] = &a->max_vel;
  kp[11] = &a->min_vel;
  kp[12] = &a->RI;
  kp[13] = &a->max_vert_acc;
  kp[14] = &a->max_theta;
  kp[15] = &a->gear_switch_time;
  *p = {};
  p->func = (void *)kinodynamic_kernel_unrolled;
  p->gridDim = dim3(a->blocks);
  p->blockDim = dim3(a->threads);
  p->sharedMemBytes = 0;
  p->kernelParams = kp;
}

static void valid_params(cudaKernelNodeParams *p, KinoLaunch *a, void **kp) {
  kp[0] = &a->map;
  kp[1] = &a->d_inter;
  kp[2] = &a->patch_length_px;
  kp[3] = &a->patch_width_px;
  kp[4] = &a->car_l2;
  kp[5] = &a->car_w2;
  kp[6] = &a->nx;
  kp[7] = &a->ts;
  kp[8] = &a->d_valid;
  *p = {};
  p->func = (void *)check_validity_batch_kernel;
  p->gridDim = dim3(kTimesteps, a->rollouts);
  p->blockDim = dim3(a->patch_length_px, a->patch_width_px);
  p->sharedMemBytes = 0;
  p->kernelParams = kp;
}

static bool graph_launch(const KinoLaunch &in, cudaStream_t stream) {
  if (g.broken)
    return false;
  std::lock_guard<std::mutex> lock(g.mu);
  auto key = std::make_pair(in.d_base, in.rollouts);
  auto it = g.graphs.find(key);
  if (it == g.graphs.end()) {
    FastLaunch<KinoLaunch>::Entry e;
    e.args = in;
    void *dkp[16];
    void *vkp[9];
    cudaKernelNodeParams dp, vp;
    cudaGraphNode_t h2d;
    dyn_params(&dp, &e.args, dkp);
    valid_params(&vp, &e.args, vkp);
    cudaError_t err = cudaGraphCreate(&e.graph, 0);
    if (err == cudaSuccess && !e.args.zero_copy)
      err = cudaGraphAddMemcpyNode1D(&h2d, e.graph, nullptr, 0, e.args.d_base,
                                    e.args.h_base, e.args.state_bytes,
                                    cudaMemcpyHostToDevice);
    if (err == cudaSuccess)
      err = cudaGraphAddKernelNode(&e.dyn, e.graph,
                                  e.args.zero_copy ? nullptr : &h2d,
                                  e.args.zero_copy ? 0 : 1, &dp);
    if (err == cudaSuccess)
      err = cudaGraphAddKernelNode(&e.valid_node, e.graph, &e.dyn, 1, &vp);
    if (err == cudaSuccess && !e.args.zero_copy) {
      cudaGraphNode_t d2h;
      err = cudaGraphAddMemcpyNode1D(&d2h, e.graph, &e.valid_node, 1,
                                    e.args.h_base, e.args.d_base,
                                    e.args.total_bytes, cudaMemcpyDeviceToHost);
    }
    if (err == cudaSuccess)
      err = cudaGraphInstantiateWithFlags(&e.exec, e.graph, 0);
    if (err != cudaSuccess) {
      std::fprintf(stderr, "[kinodynamic] cuda graph capture failed: %s\n",
                   cudaGetErrorString(err));
      if (e.exec)
        cudaGraphExecDestroy(e.exec);
      if (e.graph)
        cudaGraphDestroy(e.graph);
      g.broken = true;
      return false;
    }
    it = g.graphs.emplace(key, std::move(e)).first;
  } else if (!same_map(it->second.args.map, in.map)) {
    it->second.args.map = in.map;
    void *dkp[16];
    void *vkp[9];
    cudaKernelNodeParams dp, vp;
    dyn_params(&dp, &it->second.args, dkp);
    valid_params(&vp, &it->second.args, vkp);
    cudaError_t err =
        cudaGraphExecKernelNodeSetParams(it->second.exec, it->second.dyn, &dp);
    if (err == cudaSuccess)
      err = cudaGraphExecKernelNodeSetParams(it->second.exec,
                                            it->second.valid_node, &vp);
    if (err != cudaSuccess) {
      std::fprintf(stderr, "[kinodynamic] cuda graph map update failed: %s\n",
                   cudaGetErrorString(err));
      g.broken = true;
      return false;
    }
  }
  cudaError_t err = cudaGraphLaunch(it->second.exec, stream);
  if (err != cudaSuccess) {
    std::fprintf(stderr, "[kinodynamic] cuda graph launch failed: %s\n",
                 cudaGetErrorString(err));
    g.broken = true;
    return false;
  }
  return true;
}

void kinodynamic_launcher(
    float *state, float *intermediate_states, const TerrainMap &map,
    bool *valid, float *cost, float dt, int timesteps, int n_succ, int NX,
    int NC, float car_l2, float car_w2, float max_vel, float min_vel, float RI,
    float max_vert_acc, float max_theta, float gear_switch_time,
    int patch_length_px, int patch_width_px, const int blocks,
    const int threads, float * /*d_state*/, float * /*d_intermediate_states*/,
    float *d_controls, bool * /*d_valid*/, float * /*d_cost*/,
    cudaStream_t stream) {
  // controls are already on d_controls. Caller device state/valid/cost buffers
  // are unused: this launcher keeps one packed allocation per stream.
  if (n_succ <= 0)
    return;
  if (timesteps != kTimesteps || NX != kNX || NC != kNC)
    throw std::runtime_error("[kinodynamic] kernel built for timesteps " +
                             std::to_string(kTimesteps) + ", NX " +
                             std::to_string(kNX) + ", NC " + std::to_string(kNC) +
                             "; called with " + std::to_string(timesteps) + ", " +
                             std::to_string(NX) + ", " + std::to_string(NC));
  // IGHASTAR_ZERO_COPY=1: one mapped pack, kernels read it, no graph copies.
  // Default stays the pinned buffer plus H2D/D2H. Do not add cudaMallocManaged;
  // see FastLaunch::ensure_slot.
  static const bool zero_copy = [] {
    const char *e = std::getenv("IGHASTAR_ZERO_COPY");
    return e && e[0] == '1' && e[1] == '\0';
  }();
  Buf buf;
  if (!g.ensure_slot(stream, n_succ, timesteps, NX, &buf, zero_copy)) {
    std::fill(valid, valid + n_succ, false);
    return;
  }
  PackView view = pack_view(n_succ, timesteps, NX);
  char *h = static_cast<char *>(buf.h);
  char *d = static_cast<char *>(buf.d);
  std::memcpy(h, state, view.state_bytes);

  KinoLaunch in;
  in.d_base = d;
  in.h_base = h;
  in.state_bytes = view.state_bytes;
  in.total_bytes = view.total;
  in.d_state = reinterpret_cast<float *>(d);
  in.d_inter = reinterpret_cast<float *>(d + view.inter_off);
  in.d_controls = d_controls;
  in.d_cost = reinterpret_cast<float *>(d + view.cost_off);
  in.d_valid = reinterpret_cast<bool *>(d + view.valid_off);
  in.map = map;
  in.dt = dt;
  in.rollouts = n_succ;
  in.nx = NX;
  in.ts = timesteps;
  in.patch_length_px = patch_length_px;
  in.patch_width_px = patch_width_px;
  in.blocks = blocks;
  in.threads = threads;
  in.car_l2 = car_l2;
  in.car_w2 = car_w2;
  in.max_vel = max_vel;
  in.min_vel = min_vel;
  in.RI = RI;
  in.max_vert_acc = max_vert_acc;
  in.max_theta = max_theta;
  in.gear_switch_time = gear_switch_time;
  in.zero_copy = zero_copy;

  bool graphed = graph_launch(in, stream);
  if (!graphed) {
    std::fill(valid, valid + n_succ, false);
    return;
  }
  // Spin instead of cudaStreamSynchronize: the blocking wait costs a sleep and
  // wake-up per batch (see holonomic_6D.cu).
  while (cudaStreamQuery(stream) == cudaErrorNotReady) {
  }
  std::memcpy(state, h, view.state_bytes);
  std::memcpy(intermediate_states, h + view.inter_off,
              view.cost_off - view.inter_off);
  std::memcpy(cost, h + view.cost_off, view.valid_off - view.cost_off);
  std::memcpy(valid, h + view.valid_off, view.total - view.valid_off);
}

void check_validity_launcher(const TerrainMap &map, float *states,
                             int patch_length_px, int patch_width_px,
                             float car_l2, float car_w2, int n_states, int NX,
                             bool *result) {
  dim3 threads(patch_length_px, patch_width_px);
  dim3 blocks(1, n_states);
  bool *d_result;
  float *d_validity_states;
  cudaMalloc(&d_result, n_states * sizeof(bool));
  cudaMalloc(&d_validity_states, NX * n_states * sizeof(float));
  cudaMemcpy(d_result, result, n_states * sizeof(bool), cudaMemcpyHostToDevice);
  cudaMemcpy(d_validity_states, states, n_states * NX * sizeof(float),
             cudaMemcpyHostToDevice);

  check_validity_batch_kernel<<<blocks, threads>>>(
      map, d_validity_states, patch_length_px, patch_width_px, car_l2, car_w2,
      NX, 1, d_result);

  cudaMemcpy(result, d_result, n_states * sizeof(bool), cudaMemcpyDeviceToHost);
  cudaFree(d_result);
  cudaFree(d_validity_states);
}