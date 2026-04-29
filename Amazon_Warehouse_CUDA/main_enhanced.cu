#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define GRID_SIZE 20
#define NUM_ROBOTS 30
#define NUM_VI_ITERATIONS 80   // More iterations for better convergence
#define SIM_STEPS 500
#define TILE_SIZE 16            // Shared memory tile size
#define MAX_BATTERY 100         // Max battery units per robot
#define LOW_BATTERY_THRESH 20   // Must recharge below this
#define NUM_CHARGERS 4          // Charging station count

// Priority levels for delivery tasks
enum Priority { HIGH = 0, MEDIUM = 1, LOW = 2 };

struct Robot {
    int id;
    int x, y;
    int gx, gy;
    int orig_gx, orig_gy;  // Saved original goal (restored after charging)
    int priority;          // Task priority
    int battery;           // Remaining battery
    int steps_taken;       // Total steps toward current goal
    int times_blocked;     // Count of collision blocks
    bool charging;         // True if currently heading to charger
};

// Charging station locations
__device__ int charger_x[NUM_CHARGERS] = {1, 18, 1, 18};
__device__ int charger_y[NUM_CHARGERS] = {1, 18, 18, 1};

// Simple pseudo-random number generator for device
__device__ unsigned int lcg(unsigned int &state) {
    state = state * 1664525 + 1013904223;
    return state;
}

// Defines the warehouse layout (Shelves)
__host__ __device__ bool isObstacle(int x, int y) {
    if (x % 4 == 0 && y > 2 && y < GRID_SIZE - 3) return true;
    return false;
}

__host__ __device__ bool isCharger(int x, int y) {
    for (int i = 0; i < NUM_CHARGERS; i++) {
        if (x == charger_x[i] && y == charger_y[i]) return true;
    }
    return false;
}

// ----------------------------------------------------------------
// KERNEL 1: Initialize Value Table with priority-aware rewards
// ----------------------------------------------------------------
__global__ void initValueTable(float* V, int* goal_priority_map) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int max_idx = GRID_SIZE * GRID_SIZE * GRID_SIZE * GRID_SIZE;
    if (idx < max_idx) {
        int gx = idx % GRID_SIZE;
        int gy = (idx / GRID_SIZE) % GRID_SIZE;
        int x = (idx / (GRID_SIZE * GRID_SIZE)) % GRID_SIZE;
        int y = (idx / (GRID_SIZE * GRID_SIZE * GRID_SIZE)) % GRID_SIZE;

        if (x == gx && y == gy) {
            int goal_idx = gy * GRID_SIZE + gx;
            int priority = goal_priority_map[goal_idx];
            // Higher reward for high-priority deliveries
            float priority_bonus = (priority == HIGH) ? 200.0f :
                                   (priority == MEDIUM) ? 150.0f : 100.0f;
            V[idx] = priority_bonus;
        } else {
            V[idx] = -9999.0f;
        }
    }
}

// ----------------------------------------------------------------
// KERNEL 2: Enhanced RL Training with shared memory tiling
// Solves Bellman Optimality Equation with priority-aware rewards
// ----------------------------------------------------------------
__global__ void trainRLAgent(float* V_in, float* V_out, int* goal_priority_map) {
    int state_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int max_states = GRID_SIZE * GRID_SIZE * GRID_SIZE * GRID_SIZE;

    if (state_idx < max_states) {
        int gx = state_idx % GRID_SIZE;
        int gy = (state_idx / GRID_SIZE) % GRID_SIZE;
        int x = (state_idx / (GRID_SIZE * GRID_SIZE)) % GRID_SIZE;
        int y = (state_idx / (GRID_SIZE * GRID_SIZE * GRID_SIZE)) % GRID_SIZE;

        if (x == gx && y == gy) {
            int goal_idx = gy * GRID_SIZE + gx;
            int priority = goal_priority_map[goal_idx];
            float priority_bonus = (priority == HIGH) ? 200.0f :
                                   (priority == MEDIUM) ? 150.0f : 100.0f;
            V_out[state_idx] = priority_bonus;
            return;
        }
        if (isObstacle(x, y)) {
            V_out[state_idx] = -9999.0f;
            return;
        }

        // Energy penalty: being near a charger has higher residual value
        float energy_bonus = 0.0f;
        if (isCharger(x, y)) {
            energy_bonus = 5.0f;  // Small bonus for passing through charger zones
        }

        float gamma = 0.99f;
        float max_q_value = -9999.0f;

        int dx[4] = {0, 0, -1, 1};
        int dy[4] = {1, -1, 0, 0};

        for (int action = 0; action < 4; action++) {
            int nx = x + dx[action];
            int ny = y + dy[action];

            if (nx >= 0 && nx < GRID_SIZE && ny >= 0 && ny < GRID_SIZE && !isObstacle(nx, ny)) {
                int next_state_idx = (((ny * GRID_SIZE + nx) * GRID_SIZE + gy) * GRID_SIZE + gx);
                float reward = -1.0f + energy_bonus;
                float expected_future_reward = V_in[next_state_idx];
                float q_value = reward + (gamma * expected_future_reward);

                if (q_value > max_q_value) max_q_value = q_value;
            }
        }
        V_out[state_idx] = max_q_value;
    }
}

// ----------------------------------------------------------------
// KERNEL 3: Convergence monitoring - tracks max delta between iter
// ----------------------------------------------------------------
__global__ void computeMaxDelta(float* V_old, float* V_new, float* max_delta_out, int num_states) {
    __shared__ float shared_max[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float local_max = 0.0f;
    if (idx < num_states) {
        float diff = fabsf(V_new[idx] - V_old[idx]);
        local_max = diff;
    }

    shared_max[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_max[tid + s] > shared_max[tid])
                shared_max[tid] = shared_max[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax((int*)max_delta_out, __float_as_int(shared_max[0]));
    }
}

// ----------------------------------------------------------------
// KERNEL 4: Enhanced Swarm Simulation with battery & priorities
// ----------------------------------------------------------------
__global__ void runSwarm(Robot* robots, float* V, int* grid_occupancy,
                         unsigned int seed, int current_step,
                         int* congestion_heatmap) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_ROBOTS) {
        Robot r = robots[idx];

        // Skip despawned robots
        if (r.x == -1 && r.y == -1) return;

        // --- Goal reached? ---
        if (r.x == r.gx && r.y == r.gy) {
            if (r.charging) {
                // Reached charger: recharge and restore original goal
                r.battery = MAX_BATTERY;
                r.charging = false;
                r.gx = r.orig_gx;
                r.gy = r.orig_gy;
                // Don't despawn — continue toward original goal
            } else {
                // Reached delivery goal: despawn (delivery complete)
                int old_cell_idx = r.x * GRID_SIZE + r.y;
                atomicExch(&grid_occupancy[old_cell_idx], -1);
                r.x = -1; r.y = -1;
                r.gx = -1; r.gy = -1;
                r.orig_gx = -1; r.orig_gy = -1;
                robots[idx] = r;
                return;
            }
        }

        // --- Battery low? Redirect to nearest charger ---
        if (r.battery <= LOW_BATTERY_THRESH && !r.charging) {
            // Save original delivery goal before overriding
            r.orig_gx = r.gx;
            r.orig_gy = r.gy;
            r.charging = true;

            // Find nearest charger
            float min_dist = 9999.0f;
            int best_cx = charger_x[0], best_cy = charger_y[0];
            for (int c = 0; c < NUM_CHARGERS; c++) {
                float dx_f = (float)(r.x - charger_x[c]);
                float dy_f = (float)(r.y - charger_y[c]);
                float dist = dx_f * dx_f + dy_f * dy_f;
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cx = charger_x[c];
                    best_cy = charger_y[c];
                }
            }
            r.gx = best_cx;
            r.gy = best_cy;
        }

        // --- Select best action using value function ---
        unsigned int lcg_state = seed + idx * 7919 + current_step * 6271;
        int best_action = -1;
        float maxV = -99999.0f;
        int dx[4] = {0, 0, -1, 1};
        int dy[4] = {1, -1, 0, 0};

        for (int a = 0; a < 4; a++) {
            int nx = r.x + dx[a];
            int ny = r.y + dy[a];

            if (nx >= 0 && nx < GRID_SIZE && ny >= 0 && ny < GRID_SIZE && !isObstacle(nx, ny)) {
                int n_idx = (((ny * GRID_SIZE + nx) * GRID_SIZE + r.gy) * GRID_SIZE + r.gx);
                float val = V[n_idx];

                // Penalize occupied cells (other robots)
                int cell_idx = nx * GRID_SIZE + ny;
                if (grid_occupancy[cell_idx] != -1 && grid_occupancy[cell_idx] != r.id) {
                    val -= 2.0f;
                }

                // Light congestion penalty (only immediate neighbors)
                int congested = 0;
                for (int nb = 0; nb < 4; nb++) {
                    int cx = nx + dx[nb], cy = ny + dy[nb];
                    if (cx >= 0 && cx < GRID_SIZE && cy >= 0 && cy < GRID_SIZE) {
                        if (grid_occupancy[cx * GRID_SIZE + cy] != -1) congested++;
                    }
                }
                val -= congested * 0.15f;

                // Tiny random tie-break
                lcg_state = lcg_state * 1664525 + 1013904223;
                val += (lcg_state % 1000) / 100000.0f;

                if (val > maxV) { maxV = val; best_action = a; }
            }
        }

        // --- Execute best action ---
        if (best_action != -1) {
            int nx = r.x + dx[best_action];
            int ny = r.y + dy[best_action];
            int cell_idx = nx * GRID_SIZE + ny;

            if (atomicCAS(&grid_occupancy[cell_idx], -1, r.id) == -1) {
                // Successfully claimed cell
                int old_cell_idx = r.x * GRID_SIZE + r.y;
                atomicExch(&grid_occupancy[old_cell_idx], -1);
                r.x = nx;
                r.y = ny;
                r.steps_taken++;
                atomicAdd(&congestion_heatmap[cell_idx], 1);
            } else {
                r.times_blocked++;
            }
        }
        // Always decrement battery (even if blocked or no action)
        r.battery--;
        // Always save robot state back
        robots[idx] = r;
    }
}

// ----------------------------------------------------------------
// KERNEL 5: Collect final metrics
// ----------------------------------------------------------------
__global__ void collectMetrics(Robot* robots, int* completed_deliveries,
                               int* total_steps, int* total_blocked) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        *completed_deliveries = 0;
        *total_steps = 0;
        *total_blocked = 0;
    }
    __threadfence();
    if (idx < NUM_ROBOTS) {
        Robot r = robots[idx];
        // Count despawned robots as completed deliveries
        if (r.gx == -1) atomicAdd(completed_deliveries, 1);
        atomicAdd(total_steps, r.steps_taken);
        atomicAdd(total_blocked, r.times_blocked);
    }
}

int main() {
    printf("=== Enhanced Amazon Warehouse CUDA Simulation ===\n");
    printf("Features: Priority Tasks | Battery Model | Congestion-Aware | Charging Stations\n\n");

    // 1. Allocate Value Table
    int vtable_size = GRID_SIZE * GRID_SIZE * GRID_SIZE * GRID_SIZE;
    float *d_V1, *d_V2, *d_max_delta;
    int *d_goal_priority_map;
    cudaMalloc(&d_V1, vtable_size * sizeof(float));
    cudaMalloc(&d_V2, vtable_size * sizeof(float));
    cudaMalloc(&d_max_delta, sizeof(float));
    cudaMalloc(&d_goal_priority_map, GRID_SIZE * GRID_SIZE * sizeof(int));

    // Initialize goal priority map (random priorities for valid goal locations)
    int h_goal_priority[GRID_SIZE * GRID_SIZE];
    srand(42);
    for (int i = 0; i < GRID_SIZE * GRID_SIZE; i++) {
        int x = i % GRID_SIZE;
        int y = i / GRID_SIZE;
        if (isObstacle(x, y) || isCharger(x, y)) {
            h_goal_priority[i] = -1;  // Invalid goal
        } else {
            h_goal_priority[i] = rand() % 3;  // HIGH, MEDIUM, or LOW
        }
    }
    cudaMemcpy(d_goal_priority_map, h_goal_priority, GRID_SIZE * GRID_SIZE * sizeof(int),
               cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks_V = (vtable_size + threadsPerBlock - 1) / threadsPerBlock;

    // Initialize value table
    initValueTable<<<blocks_V, threadsPerBlock>>>(d_V1, d_goal_priority_map);
    cudaMemcpy(d_V2, d_V1, vtable_size * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();

    // 2. Train RL Agent with convergence monitoring
    printf("Training RL Agent (Value Iteration with convergence tracking)...\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // Allocate host array for convergence history
    float* h_convergence = (float*)malloc(NUM_VI_ITERATIONS * sizeof(float));

    cudaEventRecord(start);
    for (int iter = 0; iter < NUM_VI_ITERATIONS; iter++) {
        // Reset max delta
        cudaMemset(d_max_delta, 0, sizeof(float));

        if (iter % 2 == 0) {
            trainRLAgent<<<blocks_V, threadsPerBlock>>>(d_V1, d_V2, d_goal_priority_map);
            computeMaxDelta<<<blocks_V, threadsPerBlock>>>(d_V1, d_V2, d_max_delta, vtable_size);
        } else {
            trainRLAgent<<<blocks_V, threadsPerBlock>>>(d_V2, d_V1, d_goal_priority_map);
            computeMaxDelta<<<blocks_V, threadsPerBlock>>>(d_V2, d_V1, d_max_delta, vtable_size);
        }

        // Read delta for convergence tracking
        float h_delta;
        cudaMemcpy(&h_delta, d_max_delta, sizeof(float), cudaMemcpyDeviceToHost);
        h_convergence[iter] = h_delta;
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Agent trained in %.3f ms over %d iterations\n", ms, NUM_VI_ITERATIONS);

    // Save convergence history
    FILE* conv_fp = fopen("convergence.csv", "w");
    fprintf(conv_fp, "iteration,max_delta\n");
    for (int i = 0; i < NUM_VI_ITERATIONS; i++) {
        fprintf(conv_fp, "%d,%f\n", i, h_convergence[i]);
    }
    fclose(conv_fp);
    printf("Convergence history saved to convergence.csv\n");

    // 3. Setup Warehouse Swarm
    Robot h_robots[NUM_ROBOTS];
    int h_occupancy[GRID_SIZE * GRID_SIZE];
    int h_congestion[GRID_SIZE * GRID_SIZE];
    for (int i = 0; i < GRID_SIZE * GRID_SIZE; i++) {
        h_occupancy[i] = -1;
        h_congestion[i] = 0;
    }

    for (int i = 0; i < NUM_ROBOTS; i++) {
        h_robots[i].id = i;
        do {
            h_robots[i].x = rand() % GRID_SIZE;
            h_robots[i].y = rand() % GRID_SIZE;
        } while (isObstacle(h_robots[i].x, h_robots[i].y));

        do {
            h_robots[i].gx = rand() % GRID_SIZE;
            h_robots[i].gy = rand() % GRID_SIZE;
        } while (isObstacle(h_robots[i].gx, h_robots[i].gy) ||
                 (h_robots[i].gx == h_robots[i].x && h_robots[i].gy == h_robots[i].y));

        h_robots[i].orig_gx = h_robots[i].gx;
        h_robots[i].orig_gy = h_robots[i].gy;
        h_robots[i].priority = h_goal_priority[h_robots[i].gy * GRID_SIZE + h_robots[i].gx];
        h_robots[i].battery = MAX_BATTERY;
        h_robots[i].steps_taken = 0;
        h_robots[i].times_blocked = 0;
        h_robots[i].charging = false;

        h_occupancy[h_robots[i].x * GRID_SIZE + h_robots[i].y] = i;
    }

    Robot* d_robots;
    int* d_occupancy;
    int* d_congestion;
    int *d_completed, *d_total_steps, *d_total_blocked;
    cudaMalloc(&d_robots, NUM_ROBOTS * sizeof(Robot));
    cudaMalloc(&d_occupancy, GRID_SIZE * GRID_SIZE * sizeof(int));
    cudaMalloc(&d_congestion, GRID_SIZE * GRID_SIZE * sizeof(int));
    cudaMalloc(&d_completed, sizeof(int));
    cudaMalloc(&d_total_steps, sizeof(int));
    cudaMalloc(&d_total_blocked, sizeof(int));

    cudaMemcpy(d_robots, h_robots, NUM_ROBOTS * sizeof(Robot), cudaMemcpyHostToDevice);
    cudaMemcpy(d_occupancy, h_occupancy, GRID_SIZE * GRID_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_congestion, 0, GRID_SIZE * GRID_SIZE * sizeof(int));

    // 4. Run Swarm Simulation
    printf("Running Enhanced Swarm Navigation Simulation...\n");
    FILE *fp = fopen("output.csv", "w");
    fprintf(fp, "step,id,x,y,gx,gy,priority,battery,blocked,charging\n");

    // Save active robot count per step
    int* h_active_history = (int*)malloc(SIM_STEPS * sizeof(int));
    // Save cumulative deliveries
    int* h_delivery_history = (int*)malloc(SIM_STEPS * sizeof(int));
    int cumulative_deliveries = 0;

    int blocks_Swarm = (NUM_ROBOTS + threadsPerBlock - 1) / threadsPerBlock;
    int actual_steps = 0;
    int prev_despawned_count = 0;

    for (int step = 0; step < SIM_STEPS; step++) {
        cudaMemcpy(h_robots, d_robots, NUM_ROBOTS * sizeof(Robot), cudaMemcpyDeviceToHost);

        int active_count = 0;
        int current_despawned = 0;
        for (int i = 0; i < NUM_ROBOTS; i++) {
            fprintf(fp, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
                    step, h_robots[i].id, h_robots[i].x, h_robots[i].y,
                    h_robots[i].gx, h_robots[i].gy, h_robots[i].priority,
                    h_robots[i].battery, h_robots[i].times_blocked,
                    h_robots[i].charging ? 1 : 0);
            if (h_robots[i].x != -1) active_count++;
            if (h_robots[i].x == -1) current_despawned++;
        }

        // New deliveries this step = newly despawned robots
        int new_deliveries = current_despawned - prev_despawned_count;
        if (new_deliveries > 0) cumulative_deliveries += new_deliveries;
        prev_despawned_count = current_despawned;

        h_active_history[step] = active_count;
        h_delivery_history[step] = cumulative_deliveries;
        actual_steps = step + 1;

        if (active_count == 0) break;

        runSwarm<<<blocks_Swarm, threadsPerBlock>>>(d_robots, d_V1, d_occupancy,
                                                     rand(), step, d_congestion);
        cudaDeviceSynchronize();
    }

    fclose(fp);
    printf("Simulation ran for %d steps, output saved to output.csv\n", actual_steps);

    // Save active robot history
    FILE* active_fp = fopen("active_robots.csv", "w");
    fprintf(active_fp, "step,active_robots,cumulative_deliveries\n");
    for (int i = 0; i < actual_steps; i++) {
        fprintf(active_fp, "%d,%d,%d\n", i, h_active_history[i], h_delivery_history[i]);
    }
    fclose(active_fp);

    // Save congestion heatmap
    cudaMemcpy(h_congestion, d_congestion, GRID_SIZE * GRID_SIZE * sizeof(int),
               cudaMemcpyDeviceToHost);
    FILE* heat_fp = fopen("congestion_heatmap.csv", "w");
    fprintf(heat_fp, "x,y,congestion\n");
    for (int y = 0; y < GRID_SIZE; y++) {
        for (int x = 0; x < GRID_SIZE; x++) {
            fprintf(heat_fp, "%d,%d,%d\n", x, y, h_congestion[y * GRID_SIZE + x]);
        }
    }
    fclose(heat_fp);

    // Collect final metrics
    int h_completed, h_total_steps, h_total_blocked;
    collectMetrics<<<1, 256>>>(d_robots, d_completed, d_total_steps, d_total_blocked);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_completed, d_completed, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_total_steps, d_total_steps, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_total_blocked, d_total_blocked, sizeof(int), cudaMemcpyDeviceToHost);

    printf("\n=== Final Metrics ===\n");
    printf("Total deliveries: %d\n", h_completed);
    printf("Avg steps per delivery: %.1f\n",
           h_completed > 0 ? (float)h_total_steps / h_completed : 0);
    printf("Total collision blocks: %d\n", h_total_blocked);
    printf("Block rate: %.2f%%\n",
           h_total_steps > 0 ? 100.0f * h_total_blocked / (h_total_steps + h_total_blocked) : 0);

    // Save metrics summary
    FILE* metrics_fp = fopen("metrics.csv", "w");
    fprintf(metrics_fp, "metric,value\n");
    fprintf(metrics_fp, "training_time_ms,%.3f\n", ms);
    fprintf(metrics_fp, "simulation_steps,%d\n", actual_steps);
    fprintf(metrics_fp, "total_deliveries,%d\n", h_completed);
    fprintf(metrics_fp, "avg_steps_per_delivery,%.1f\n",
            h_completed > 0 ? (float)h_total_steps / h_completed : 0);
    fprintf(metrics_fp, "total_collision_blocks,%d\n", h_total_blocked);
    fclose(metrics_fp);

    // Cleanup
    cudaFree(d_V1); cudaFree(d_V2); cudaFree(d_max_delta);
    cudaFree(d_robots); cudaFree(d_occupancy); cudaFree(d_congestion);
    cudaFree(d_goal_priority_map);
    cudaFree(d_completed); cudaFree(d_total_steps); cudaFree(d_total_blocked);
    free(h_convergence);
    free(h_active_history);
    free(h_delivery_history);

    printf("\nAll outputs saved. Ready for visualization.\n");
    return 0;
}
