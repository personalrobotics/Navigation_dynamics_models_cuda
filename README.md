# Navigation dynamics models

This repository holds the vehicle models for [IGHA*](https://github.com/personalrobotics/IGHAStar). IGHA* is the search. The search code stays in that repository. This repository holds the rollout code.

On the Allen Backside kinodynamic set, the successor call is about 3 times shorter after the packed graph. The before run is 625 expansions with the old direct launch. The after run is 5000 expansions with the packed graph. Both runs have preemption on. Successor time is the successor call divided by popped expansions. It does not include the bidirectional barrier.

| | Before | After | Faster |
|---|---|---|---|
| IGHA* successor | 71 µs | 26 µs | 2.7× |
| BiIGHA* successor | 84 µs | 24 µs | 3.6× |
| IGHA* wall rate | 9280 exp/s | 12031 exp/s | 1.3× |
| BiIGHA* wall rate | 7037 exp/s | 19067 exp/s | 2.7× |

The before run still waits on the bidirectional barrier. The after run does not. IGHA* never waits on that barrier, so the IGHA* rows are the model only. The BiIGHA* wall rate mixes the model with the barrier removal. The successor rows do not. Solved counts moved from 81/170 and 128/170 to 128/170 and 144/170. Part of that is the larger budget.

Three models are here.

- `kinematic` moves with a commanded speed and a curvature.
- `kinodynamic` integrates acceleration and curvature. The state includes speed.
- `holonomic_6D` is the R6 model. The state is x, y, yaw, body vx, body vy, and yaw rate.

Each model has a header and a CUDA file. `kinematic` and `kinodynamic` also have a CPU file.

## Connect a model to the search

`ighastar.cpp` includes one header. The compile flag `IGHASTAR_ENV_HEADER` names that header. The header must define an `Environment` class. The method list is in the [IGHAStar environment guide](https://github.com/personalrobotics/IGHAStar/blob/main/ighastar/src/Environments/README.md).

Do not copy the `Node` type into a new header. Define `n_dims`, `n_cont`, `hash_dims`, and `calc_hash`. Then include `node.h` and `trajectory_export.h` from IGHAStar. Pass `n_cont` to `near_meet_.load`. Build each child with `fill_substep_times`.

`g` is the cost on the child vertex. The edge cost is `child.g - parent.g`. The edge stores the intermediate poses, the controls, and the substep times.

`load_model.py` builds this link. It puts `include/` from this repository first. It compiles `ighastar.cpp` with the model source file.

## Add a model

1. Write the dynamics in a header and a `.cu` file. Keep `timesteps`, `NX`, and `NC` as runtime arguments.
2. Give each rollout its own CUDA thread. Loop from step 1 through `timesteps`.
3. Launch the dynamics kernel. Then launch the validity kernel.
4. Test this simple kernel on a fixed problem set. Record how many searches find a path.
5. Pack the buffers. Unroll the step loop. Capture a CUDA graph.
6. Run the same problems again. The solved count must stay in the same band.

Do not change the rollout math after step 4. A bidirectional search can differ by a few problems. The validity kernel writes `d_valid[k]` from many threads. Two search threads also run at the same time.

## Pack, unroll, and graph

These rules come from the kinodynamic runs on the Allen Backside problems. Packing and unrolling removed most of the successor time. The graph removes the launch cost of one expansion. Preemption already batches many expansions into one launch, so that launch cost is small.

Use one buffer per CUDA stream. The byte order is state, intermediates, cost, then valid. Allocate the device buffer with `cudaMalloc`. Allocate the host mirror with `cudaMallocHost`. Copy the caller state into the pinned prefix on the CPU. Copy only that prefix to the device. Copy the full pack back. Then copy each part into the caller arrays.

The CPU copy into the pinned prefix stays outside the graph. The graph has four nodes, in this order.

1. Copy the state prefix from the host to the device.
2. Run the dynamics kernel.
3. Run the validity kernel.
4. Copy the full pack from the device to the host.

Key each graph by the device base pointer and the rollout count. Each stream has its own slot. A bidirectional search uses two streams. If capture or replay fails, mark the rollouts invalid. Do not keep a second launch path.

`timesteps` becomes the macro `IGHASTAR_TIMESTEPS`. `load_model.py` passes it from `node_info.timesteps`. Put `#pragma unroll` on the step loop. Keep the valid flag in a register. Store `valid[k]` once, after the loop. Keep the order of the `&&` checks. When an earlier check fails, do not evaluate the later checks.

`kinodynamic` and `kinematic` integrate every step. `holonomic_6D` stops the step loop when a check fails. The cost is still the full edge duration. Write this choice in a new kernel. Do not copy the `holonomic_6D` stop by accident.

`check_validity_launcher` allocates device memory on every call. Near-meet checks use it. The expansion path does not. Leave that launcher in the simple form.

## Memory and the wait

Do not use `cudaMallocManaged`. On Orin, at 10000 expansions with `best_f`, the pinned pack did 11397 expansions per second. The managed pointer did 3519. A CPU write into the managed pointer caused a segmentation fault.

Zero-copy is optional. Only `kinodynamic.cu` reads it. Set the environment variable `IGHASTAR_ZERO_COPY` to the character `1`. This is not a yaml key. `cudaHostAlloc` maps the host buffer into the device address space. The graph then has no copy nodes. On Jetson this mapping is not cached for the GPU. On a discrete GPU the kernel reads the buffer across PCIe. Mapped memory is not CUDA unified memory.

`kinodynamic` and `holonomic_6D` wait with `cudaStreamQuery`. `kinematic` still calls `cudaStreamSynchronize`. A new model should use the query loop. The blocking wait sleeps.

## CPU

`kinodynamic_cpu.cpp` and `kinematic_cpu.cpp` use OpenMP on the rollout loop and the validity loop. The math matches the CUDA file. One call uses all cores. A bidirectional search runs two threads. Each thread runs a full team. Set `OMP_NUM_THREADS` if the machine runs too many threads.

## Problems, rate, and tests

`problems/Allen_backside_0/problems` holds 170 pairs. Each pair has an initial pose and a goal pose. `problems_8m` holds 162 pairs. The searches run through `hound_nav/hound_nav/run_bag_playback.py`. `summarize_exp_rate.py` reads the pickles. It prints total expansions divided by total solve time. The mean inside the playback script is a different number. That mean weights a one-expansion search the same as a long search.

The tests are in `tests/`. Run `python3 test_environment.py --env kinodynamic` from that directory. The same command accepts `kinematic` and `holonomic_6D`.
