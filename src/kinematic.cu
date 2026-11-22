#include "fast_launch.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
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
constexpr int kNX = 4, kNC = 2; // match n_dims, n_cont in include/kinematic.h

__device__ float nan_to_num(float x, float replace) {
  return (std::isnan(x) || std::isinf(x)) ? replace : x;
}

__device__ float clamp(float x, float lower, float upper) {
  return std::min(std::max(x, lower), upper);
}

__device__ float wrap_to_pi(float x) {
  return std::fmod(x + M_PI, 2 * M_PI) - M_PI;
}

// Maps world coordinates to elevation value from heightmap (device function)
__device__ float map_to_elev(float x, float y, const float *elev,
                             int map_size_px, float res_inv) {
  int img_X = clamp(static_cast<int>((x * res_inv)), 0, map_size_px - 1);
  int img_Y = clamp(static_cast<int>((y * res_inv)), 0, map_size_px - 1);
  return elev[img_Y * map_size_px + img_X];
}

// Computes the 3D footprint coordinates and center height of the car (device
// function)
__device__ void get_footprint_z(float *fl, float *fr, float *bl, float *br,
                                float &z, float x, float y, float cy, float sy,
                                const float *elev, int map_size_px,
                                float res_inv, float car_l2, float car_w2) {
  fl[0] = car_l2 * cy - car_w2 * sy + x;
  fl[1] = car_l2 * sy + car_w2 * cy + y;

  fr[0] = car_l2 * cy - (-1) * car_w2 * sy + x;
  fr[1] = car_l2 * sy + (-1) * car_w2 * cy + y;

  bl[0] = (-1) * car_l2 * cy - car_w2 * sy + x;
  bl[1] = (-1) * car_l2 * sy + car_w2 * cy + y;

  br[0] = (-1) * car_l2 * cy - (-1) * car_w2 * sy + x;
  br[1] = (-1) * car_l2 * sy + (-1) * car_w2 * cy + y;

  float z_cent = map_to_elev(0, 0, elev, map_size_px, res_inv);
  z = map_to_elev(x, y, elev, map_size_px, res_inv) - z_cent;

  fl[2] = map_to_elev(fl[0], fl[1], elev, map_size_px, res_inv) - z_cent;
  fr[2] = map_to_elev(fr[0], fr[1], elev, map_size_px, res_inv) - z_cent;
  bl[2] = map_to_elev(bl[0], bl[1], elev, map_size_px, res_inv) - z_cent;
  br[2] = map_to_elev(br[0], br[1], elev, map_size_px, res_inv) - z_cent;
}

// Checks validity of multiple states against a bitmap (CUDA kernel)
__global__ void
check_validity_batch_kernel(const float *bitmap, int map_size_px, float map_res,
                            float *d_intermediate_states, int patch_length_px,
                            int patch_width_px, float car_l2, float car_w2,
                            int NX, int timesteps, bool *d_valid) {
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
  float offset_x = (i * map_res) - car_l2;
  float offset_y = (j * map_res) - car_w2;

  float px = offset_x * cy - offset_y * sy + x;
  float py = offset_x * sy + offset_y * cy + y;
  if (px < 0 || px >= map_size_px * map_res || py < 0 ||
      py >= map_size_px * map_res) {
    d_valid[k] = false;
    return;
  }

  if (map_to_elev(px, py, bitmap, map_size_px, 1.0f / map_res) <= 250.0f) {
    d_valid[k] = false;
  }
}

// kTimesteps is a compile-time constant (IGHASTAR_TIMESTEPS), so the step loop
// unrolls. `ok` stays in a register; the old kernel stored valid[k] every step.
__global__ void kinematic_kernel_unrolled(
    float *state, float *intermediate_states, float *controls,
    const float *BEVmap_height, const float *BEVmap_cost, bool *valid,
    float *cost, float dt, int rollouts, const int BEVmap_size_px,
    float BEVmap_res, float car_l2, float car_w2, float max_vel, float min_vel,
    float RI, float max_vert_acc, float max_theta, float gear_switch_time) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= rollouts)
    return;
  (void)BEVmap_cost;
  (void)max_vel;
  (void)min_vel;
  (void)RI;
  (void)max_vert_acc;

  float res_inv = 1.0f / BEVmap_res;
  int state_base = k * kNX;
  float x = state[state_base + x_index];
  float y = state[state_base + y_index];
  float yaw = state[state_base + yaw_index];
  float vx = state[state_base + vx_index];
  float wz = 0.0f;
  float vy = 0, vz = 0;

  float cy = cosf(yaw), sy = sinf(yaw);
  float fl[3], fr[3], bl[3], br[3], z;
  float roll, pitch, cp, sp, cr, sr;
  bool ok = true;

#pragma unroll
  for (int t = 1; t <= kTimesteps; t++) {
    int control_base = k * kTimesteps * kNC + (t - 1) * kNC;
    float curvature = controls[control_base + st_index];
    vx = controls[control_base + th_index];
    wz = curvature * vx;

    cy = cosf(yaw);
    sy = sinf(yaw);
    get_footprint_z(fl, fr, bl, br, z, x, y, cy, sy, BEVmap_height,
                    BEVmap_size_px, res_inv, car_l2, car_w2);
    roll = atan2f((fl[2] + bl[2]) - (fr[2] + br[2]), 4 * car_w2);
    pitch = atan2f((bl[2] + br[2]) - (fl[2] + fr[2]), 4 * car_l2);

    cp = cosf(pitch), sp = sinf(pitch);
    cr = cosf(roll), sr = sinf(roll);

    yaw = wrap_to_pi(yaw + wz * dt);
    cy = cosf(yaw);
    sy = sinf(yaw);

    x += dt * (vx * (cp * cy) + vy * (sr * sp * cy - cr * sy) +
               vz * (cr * sp * cy + sr * sy));
    y += dt * (vx * (cp * sy) + vy * (sr * sp * sy + cr * cy) +
               vz * (cr * sp * sy - sr * cy));

    ok = ok && fabsf(pitch) < max_theta && fabsf(roll) < max_theta;

    int intermediate_index = k * kTimesteps * kNX + (t - 1) * kNX;
    intermediate_states[intermediate_index + x_index] = x;
    intermediate_states[intermediate_index + y_index] = y;
    intermediate_states[intermediate_index + yaw_index] = yaw;
    intermediate_states[intermediate_index + vx_index] = vx;
  }
  valid[k] = ok;
  float gear_switch_cost = gear_switch_time * (vx < 0);
  cost[k] = fabs(kTimesteps * dt) * fabs(vx) + gear_switch_cost;
  state[state_base + x_index] = x;
  state[state_base + y_index] = y;
  state[state_base + yaw_index] = yaw;
  state[state_base + vx_index] = vx;
}

struct KineLaunch {
  void *d_base = nullptr;
  void *h_base = nullptr;
  size_t state_bytes = 0;
  size_t total_bytes = 0;
  float *d_state = nullptr;
  float *d_inter = nullptr;
  float *d_controls = nullptr;
  float *d_cost = nullptr;
  bool *d_valid = nullptr;
  const float *height = nullptr;
  const float *costmap = nullptr;
  int map_size = 0;
  float map_res = 0;
  float dt = 0;
  int rollouts = 0;
  int nx = 4;
  int ts = 10;
  int patch_length_px = 0, patch_width_px = 0;
  int blocks = 0, threads = 0;
  float car_l2 = 0, car_w2 = 0, max_vel = 0, min_vel = 0, RI = 0,
        max_vert_acc = 0, max_theta = 0, gear_switch_time = 0;
};

static bool same_map(const KineLaunch &a, const KineLaunch &b) {
  return a.height == b.height && a.costmap == b.costmap &&
         a.map_size == b.map_size && a.map_res == b.map_res;
}

static FastLaunch<KineLaunch> g("kinematic");

static void dyn_params(cudaKernelNodeParams *p, KineLaunch *a, void **kp) {
  kp[0] = &a->d_state;
  kp[1] = &a->d_inter;
  kp[2] = &a->d_controls;
  kp[3] = &a->height;
  kp[4] = &a->costmap;
  kp[5] = &a->d_valid;
  kp[6] = &a->d_cost;
  kp[7] = &a->dt;
  kp[8] = &a->rollouts;
  kp[9] = &a->map_size;
  kp[10] = &a->map_res;
  kp[11] = &a->car_l2;
  kp[12] = &a->car_w2;
  kp[13] = &a->max_vel;
  kp[14] = &a->min_vel;
  kp[15] = &a->RI;
  kp[16] = &a->max_vert_acc;
  kp[17] = &a->max_theta;
  kp[18] = &a->gear_switch_time;
  *p = {};
  p->func = (void *)kinematic_kernel_unrolled;
  p->gridDim = dim3(a->blocks);
  p->blockDim = dim3(a->threads);
  p->sharedMemBytes = 0;
  p->kernelParams = kp;
}

static void valid_params(cudaKernelNodeParams *p, KineLaunch *a, void **kp) {
  kp[0] = &a->costmap;
  kp[1] = &a->map_size;
  kp[2] = &a->map_res;
  kp[3] = &a->d_inter;
  kp[4] = &a->patch_length_px;
  kp[5] = &a->patch_width_px;
  kp[6] = &a->car_l2;
  kp[7] = &a->car_w2;
  kp[8] = &a->nx;
  kp[9] = &a->ts;
  kp[10] = &a->d_valid;
  *p = {};
  p->func = (void *)check_validity_batch_kernel;
  p->gridDim = dim3(kTimesteps, a->rollouts);
  p->blockDim = dim3(a->patch_length_px, a->patch_width_px);
  p->sharedMemBytes = 0;
  p->kernelParams = kp;
}

static bool graph_launch(const KineLaunch &in, cudaStream_t stream) {
  if (g.broken)
    return false;
  std::lock_guard<std::mutex> lock(g.mu);
  auto key = std::make_pair(in.d_base, in.rollouts);
  auto it = g.graphs.find(key);
  if (it == g.graphs.end()) {
    FastLaunch<KineLaunch>::Entry e;
    e.args = in;
    void *dkp[19];
    void *vkp[11];
    cudaKernelNodeParams dp, vp;
    cudaGraphNode_t h2d, d2h;
    dyn_params(&dp, &e.args, dkp);
    valid_params(&vp, &e.args, vkp);
    cudaError_t err = cudaGraphCreate(&e.graph, 0);
    if (err == cudaSuccess)
      err = cudaGraphAddMemcpyNode1D(&h2d, e.graph, nullptr, 0, e.args.d_base,
                                    e.args.h_base, e.args.state_bytes,
                                    cudaMemcpyHostToDevice);
    if (err == cudaSuccess)
      err = cudaGraphAddKernelNode(&e.dyn, e.graph, &h2d, 1, &dp);
    if (err == cudaSuccess)
      err = cudaGraphAddKernelNode(&e.valid_node, e.graph, &e.dyn, 1, &vp);
    if (err == cudaSuccess)
      err = cudaGraphAddMemcpyNode1D(&d2h, e.graph, &e.valid_node, 1,
                                    e.args.h_base, e.args.d_base,
                                    e.args.total_bytes, cudaMemcpyDeviceToHost);
    if (err == cudaSuccess)
      err = cudaGraphInstantiateWithFlags(&e.exec, e.graph, 0);
    if (err != cudaSuccess) {
      std::fprintf(stderr, "[kinematic] cuda graph capture failed: %s\n",
                   cudaGetErrorString(err));
      if (e.exec)
        cudaGraphExecDestroy(e.exec);
      if (e.graph)
        cudaGraphDestroy(e.graph);
      g.broken = true;
      return false;
    }
    it = g.graphs.emplace(key, std::move(e)).first;
  } else if (!same_map(it->second.args, in)) {
    it->second.args.height = in.height;
    it->second.args.costmap = in.costmap;
    it->second.args.map_size = in.map_size;
    it->second.args.map_res = in.map_res;
    void *dkp[19];
    void *vkp[11];
    cudaKernelNodeParams dp, vp;
    dyn_params(&dp, &it->second.args, dkp);
    valid_params(&vp, &it->second.args, vkp);
    cudaError_t err =
        cudaGraphExecKernelNodeSetParams(it->second.exec, it->second.dyn, &dp);
    if (err == cudaSuccess)
      err = cudaGraphExecKernelNodeSetParams(it->second.exec,
                                            it->second.valid_node, &vp);
    if (err != cudaSuccess) {
      std::fprintf(stderr, "[kinematic] cuda graph map update failed: %s\n",
                   cudaGetErrorString(err));
      g.broken = true;
      return false;
    }
  }
  cudaError_t err = cudaGraphLaunch(it->second.exec, stream);
  if (err != cudaSuccess) {
    std::fprintf(stderr, "[kinematic] cuda graph launch failed: %s\n",
                 cudaGetErrorString(err));
    g.broken = true;
    return false;
  }
  return true;
}

void kinematic_launcher(
    float *state, float *intermediate_states, const float *heightmap,
    const float *costmap, bool *valid, float *cost, float dt, int timesteps,
    int n_succ, int NX, int NC, const int map_size, float map_res, float car_l2,
    float car_w2, float max_vel, float min_vel, float RI, float max_vert_acc,
    float max_theta, float gear_switch_time, int patch_length_px,
    int patch_width_px, const int blocks, const int threads, float * /*d_state*/,
    float * /*d_intermediate_states*/, float *d_controls, bool * /*d_valid*/,
    float * /*d_cost*/, cudaStream_t stream) {
  if (n_succ <= 0)
    return;
  if (timesteps != kTimesteps || NX != kNX || NC != kNC)
    throw std::runtime_error("[kinematic] kernel built for timesteps " +
                             std::to_string(kTimesteps) + ", NX " +
                             std::to_string(kNX) + ", NC " + std::to_string(kNC) +
                             "; called with " + std::to_string(timesteps) + ", " +
                             std::to_string(NX) + ", " + std::to_string(NC));
  Buf buf;
  if (!g.ensure_slot(stream, n_succ, timesteps, NX, &buf)) {
    std::fill(valid, valid + n_succ, false);
    return;
  }
  PackView view = pack_view(n_succ, timesteps, NX);
  char *h = static_cast<char *>(buf.h);
  char *d = static_cast<char *>(buf.d);
  std::memcpy(h, state, view.state_bytes);

  KineLaunch in;
  in.d_base = d;
  in.h_base = h;
  in.state_bytes = view.state_bytes;
  in.total_bytes = view.total;
  in.d_state = reinterpret_cast<float *>(d);
  in.d_inter = reinterpret_cast<float *>(d + view.inter_off);
  in.d_controls = d_controls;
  in.d_cost = reinterpret_cast<float *>(d + view.cost_off);
  in.d_valid = reinterpret_cast<bool *>(d + view.valid_off);
  in.height = heightmap;
  in.costmap = costmap;
  in.map_size = map_size;
  in.map_res = map_res;
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

  bool graphed = graph_launch(in, stream);
  if (!graphed) {
    std::fill(valid, valid + n_succ, false);
    return;
  }
  cudaStreamSynchronize(stream);
  std::memcpy(state, h, view.state_bytes);
  std::memcpy(intermediate_states, h + view.inter_off,
              view.cost_off - view.inter_off);
  std::memcpy(cost, h + view.cost_off, view.valid_off - view.cost_off);
  std::memcpy(valid, h + view.valid_off, view.total - view.valid_off);
}

void check_validity_launcher(const float *costmap, int map_size_px,
                             float map_res, float *states, int patch_length_px,
                             int patch_width_px, float car_l2, float car_w2,
                             int n_states, int NX, bool *result) {
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
      costmap, map_size_px, map_res, d_validity_states, patch_length_px,
      patch_width_px, car_l2, car_w2, NX, 1, d_result);

  cudaMemcpy(result, d_result, n_states * sizeof(bool), cudaMemcpyDeviceToHost);
  cudaFree(d_result);
  cudaFree(d_validity_states);
}