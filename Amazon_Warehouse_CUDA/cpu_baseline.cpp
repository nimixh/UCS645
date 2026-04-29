#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <random>
#include <omp.h>
#include <sys/time.h>
#include <time.h>

#define GRID_SIZE 20
#define NUM_ROBOTS 30
#define SIM_STEPS 300

struct Robot { int id, x, y, gx, gy; };
struct Point { int x, y; };
bool operator==(const Point& a, const Point& b) { return a.x==b.x && a.y==b.y; }

struct PointHash {
    size_t operator()(const Point& p) const { return p.y * 1000 + p.x; }
};

bool isObstacle(int x, int y) {
    return (x % 4 == 0) && (y > 2) && (y < GRID_SIZE - 3);
}

// BFS from start to goal, avoiding blocked cells (hash set of Point)
std::vector<Point> bfs_path(Point start, Point goal,
                             const std::unordered_set<Point, PointHash>& blocked) {
    if (start == goal) return {start};

    std::queue<Point> q;
    std::unordered_map<Point, Point, PointHash> parent;
    q.push(start);
    parent[start] = {-1, -1};

    int dx[4] = {0, 0, -1, 1};
    int dy[4] = {1, -1, 0, 0};

    while (!q.empty()) {
        Point cur = q.front(); q.pop();
        for (int a = 0; a < 4; a++) {
            int nx = cur.x + dx[a], ny = cur.y + dy[a];
            if (nx < 0 || nx >= GRID_SIZE || ny < 0 || ny >= GRID_SIZE) continue;
            if (isObstacle(nx, ny)) continue;
            Point nxt = {nx, ny};
            if (nxt == goal) {
                std::vector<Point> path = {goal, cur};
                while (!(parent[path.back()] == Point{-1,-1})) {
                    path.push_back(parent[path.back()]);
                }
                std::reverse(path.begin(), path.end());
                return path;
            }
            if (parent.find(nxt) == parent.end() && blocked.find(nxt) == blocked.end()) {
                parent[nxt] = cur;
                q.push(nxt);
            }
        }
    }
    return {};  // No path
}

double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

void run_cpu(int num_threads) {
    const char* mode = (num_threads == 1) ? "CPU Serial (1 thread)" : "CPU OpenMP";
    omp_set_num_threads(num_threads);

    // Setup robots (same seed as Python test)
    srand(42);
    Robot robots[NUM_ROBOTS];
    int occupancy[GRID_SIZE * GRID_SIZE];
    memset(occupancy, -1, sizeof(occupancy));

    for (int i = 0; i < NUM_ROBOTS; i++) {
        robots[i].id = i;
        do { robots[i].x = rand() % GRID_SIZE; robots[i].y = rand() % GRID_SIZE; }
        while (isObstacle(robots[i].x, robots[i].y));
        do {
            robots[i].gx = rand() % GRID_SIZE; robots[i].gy = rand() % GRID_SIZE;
        } while (isObstacle(robots[i].gx, robots[i].gy) ||
                 (robots[i].gx == robots[i].x && robots[i].gy == robots[i].y));
        occupancy[robots[i].y * GRID_SIZE + robots[i].x] = i;
    }

    // Simulation
    printf("  Running swarm simulation (%s)...\n", mode);
    double t_start = get_time_ms();

    FILE* fp = fopen("output_cpu.csv", "w");
    fprintf(fp, "step,id,x,y,gx,gy\n");

    int completed = 0, actual_steps = 0;
    std::mt19937 rng(42);

    for (int step = 0; step < SIM_STEPS; step++) {
        // Write current state
        for (int i = 0; i < NUM_ROBOTS; i++) {
            fprintf(fp, "%d,%d,%d,%d,%d,%d\n",
                    step, robots[i].id, robots[i].x, robots[i].y,
                    robots[i].gx, robots[i].gy);
        }

        int active_count = 0;
        for (int i = 0; i < NUM_ROBOTS; i++)
            if (robots[i].gx != -1) active_count++;

        actual_steps = step + 1;
        if (active_count == 0) break;

        // Random processing order
        std::vector<int> order(NUM_ROBOTS);
        for (int i = 0; i < NUM_ROBOTS; i++) order[i] = i;
        std::shuffle(order.begin(), order.end(), rng);

        // Build blocked set (all occupied cells)
        std::unordered_set<Point, PointHash> blocked_set;
        for (int y = 0; y < GRID_SIZE; y++)
            for (int x = 0; x < GRID_SIZE; x++)
                if (occupancy[y * GRID_SIZE + x] != -1)
                    blocked_set.insert({x, y});

        std::unordered_set<Point, PointHash> reserved;  // Cells claimed this step

        for (int idx = 0; idx < NUM_ROBOTS; idx++) {
            int i = order[idx];
            if (robots[i].gx == -1) continue;

            Robot* r = &robots[i];

            // Goal reached?
            if (r->x == r->gx && r->y == r->gy) {
                occupancy[r->y * GRID_SIZE + r->x] = -1;
                r->x = r->y = r->gx = r->gy = -1;
                completed++;
                blocked_set.erase({r->x, r->y});
                continue;
            }

            // BFS path avoiding all other robots
            blocked_set.erase({r->x, r->y});  // Don't block self
            auto path = bfs_path({r->x, r->y}, {r->gx, r->gy}, blocked_set);
            blocked_set.insert({r->x, r->y});

            if (path.size() < 2) continue;  // No path found

            Point nxt = path[1];
            if (occupancy[nxt.y * GRID_SIZE + nxt.x] == -1 &&
                reserved.find(nxt) == reserved.end()) {
                occupancy[r->y * GRID_SIZE + r->x] = -1;
                r->x = nxt.x; r->y = nxt.y;
                occupancy[nxt.y * GRID_SIZE + nxt.x] = i;
                reserved.insert(nxt);

                // Update blocked set
                blocked_set.erase({r->x, r->y});
                blocked_set.insert(nxt);
            }
        }
    }

    double t_total = get_time_ms() - t_start;
    fclose(fp);

    printf("  Done: %.1f ms (%d steps, %d deliveries)\n", t_total, actual_steps, completed);

    // Save metrics
    char fname[64];
    snprintf(fname, sizeof(fname), "metrics_cpu_t%d.csv", num_threads);
    FILE* mfp = fopen(fname, "w");
    fprintf(mfp, "metric,value\n");
    fprintf(mfp, "mode,%s\n", mode);
    fprintf(mfp, "num_threads,%d\n", num_threads);
    fprintf(mfp, "total_time_ms,%.1f\n", t_total);
    fprintf(mfp, "deliveries,%d\n", completed);
    fprintf(mfp, "simulation_steps,%d\n", actual_steps);
    fclose(mfp);
}

int main() {
    printf("=== CPU Baseline: Amazon Warehouse Pathfinding ===\n");
    printf("Grid: %dx%d | Robots: %d | Max Steps: %d\n\n",
           GRID_SIZE, GRID_SIZE, NUM_ROBOTS, SIM_STEPS);

    printf("--- CPU Serial (1 thread) ---\n");
    run_cpu(1);

    int max_threads = omp_get_max_threads();
    printf("--- CPU OpenMP (%d threads) ---\n", max_threads);
    run_cpu(max_threads);

    printf("Done.\n");
    return 0;
}
