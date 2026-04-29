const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const path = require("path");
const fs = require("fs");

// Icon rendering
const { FaWarehouse, FaRobot, FaCogs, FaBrain, FaShieldAlt, FaServer, FaChartBar, FaProjectDiagram, FaVideo, FaRocket } = require("react-icons/fa");
const { BsCpu, BsGpuCard, BsDiagram3 } = require("react-icons/bs");

function renderIconSvg(IconComponent, color = "#000000", size = 256) {
  return ReactDOMServer.renderToStaticMarkup(
    React.createElement(IconComponent, { color, size: String(size) })
  );
}

async function iconToBase64Png(IconComponent, color, size = 256) {
  const svg = renderIconSvg(IconComponent, color, size);
  const pngBuffer = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + pngBuffer.toString("base64");
}

// Image to base64 helper
function imageToBase64(filePath) {
  const buf = fs.readFileSync(filePath);
  const ext = path.extname(filePath).slice(1);
  const mime = ext === "jpg" ? "jpeg" : ext;
  return `image/${mime};base64,${buf.toString("base64")}`;
}

async function main() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9";
  pres.author = "Nimish";
  pres.title = "Accelerating Multi-Agent RL via CUDA-Native Zero-Copy Architecture";

  // --- Color Palette ---
  const C = {
    bg:     "0C1445",  // deep navy background
    bg2:    "1A237E",  // slightly lighter navy
    card:   "162060",  // card background
    white:  "FFFFFF",
    gold:   "F5C518",  // accent
    cyan:   "00D4FF",  // bright accent
    green:  "00E676",  // success
    red:    "FF5252",  // CPU color
    orange: "FF9100",  // warm accent
    purple: "B388FF",  // highlight
    gray:   "8892B0",  // muted text
    lgray:  "CCD6F6",  // light text
  };

  const FONT_TITLE = "Georgia";
  const FONT_BODY = "Calibri";

  // Factory functions for fresh shadow objects
  const mkShadow = () => ({ type: "outer", color: "000000", blur: 8, offset: 3, angle: 135, opacity: 0.25 });
  const mkCardShadow = () => ({ type: "outer", color: "000000", blur: 4, offset: 1, angle: 135, opacity: 0.2 });

  // --- Pre-render icons ---
  const icons = {
    warehouse:  await iconToBase64Png(FaWarehouse, "#F5C518", 256),
    robot:      await iconToBase64Png(FaRobot, "#00D4FF", 256),
    cpu:        await iconToBase64Png(BsCpu, "#FF5252", 256),
    gpu:        await iconToBase64Png(BsGpuCard, "#00E676", 256),
    cogs:       await iconToBase64Png(FaCogs, "#00D4FF", 256),
    brain:      await iconToBase64Png(FaBrain, "#B388FF", 256),
    shield:     await iconToBase64Png(FaShieldAlt, "#00E676", 256),
    server:     await iconToBase64Png(FaServer, "#F5C518", 256),
    chart:      await iconToBase64Png(FaChartBar, "#00D4FF", 256),
    diagram:    await iconToBase64Png(BsDiagram3, "#F5C518", 256),
    video:      await iconToBase64Png(FaVideo, "#FF9100", 256),
    rocket:     await iconToBase64Png(FaRocket, "#B388FF", 256),
  };

  // --- Helper: add a subtle decorative bar at top of slides ---
  function addTopBar(slide) {
    slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.04, fill: { color: C.gold } });
  }

  function addSlideNumber(slide, num) {
    slide.addText(`${num} / 10`, {
      x: 8.8, y: 5.15, w: 1, h: 0.35, fontSize: 9, color: C.gray, fontFace: FONT_BODY, align: "right"
    });
  }

  // ===================================================================
  // SLIDE 1: TITLE
  // ===================================================================
  let s1 = pres.addSlide();
  s1.background = { color: C.bg };
  // Decorative elements
  s1.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.gold } });
  s1.addShape(pres.shapes.RECTANGLE, { x: 0, y: 5.565, w: 10, h: 0.06, fill: { color: C.cyan } });
  // Icon row
  s1.addImage({ data: icons.warehouse, x: 4.5, y: 0.6, w: 0.5, h: 0.5 });
  s1.addImage({ data: icons.robot, x: 5.05, y: 0.6, w: 0.5, h: 0.5 });
  // Title
  s1.addText("Accelerating Multi-Agent\nReinforcement Learning via\nCUDA-Native Zero-Copy Architecture", {
    x: 0.8, y: 1.4, w: 8.4, h: 2.4, fontSize: 30, fontFace: FONT_TITLE, color: C.white,
    bold: true, align: "center", valign: "middle"
  });
  // Subtitle
  s1.addText("UCS645: Parallel and Distributed Computing — Project Defense", {
    x: 1, y: 3.7, w: 8, h: 0.5, fontSize: 14, fontFace: FONT_BODY, color: C.gray, align: "center"
  });
  // Author
  s1.addText("Nimish  |  102497027  |  Thapar Institute of Engineering & Technology", {
    x: 1, y: 4.3, w: 8, h: 0.5, fontSize: 12, fontFace: FONT_BODY, color: C.lgray, align: "center"
  });
  // Key stat callout
  s1.addText("29.6x", { x: 3.2, y: 4.85, w: 1.5, h: 0.4, fontSize: 22, fontFace: FONT_TITLE, color: C.gold, bold: true, align: "right" });
  s1.addText("GPU Speedup", { x: 4.8, y: 4.85, w: 1.8, h: 0.4, fontSize: 11, fontFace: FONT_BODY, color: C.lgray, align: "left" });
  s1.addText("17.6B", { x: 6.4, y: 4.85, w: 1.2, h: 0.4, fontSize: 22, fontFace: FONT_TITLE, color: C.cyan, bold: true, align: "right" });
  s1.addText("States/sec", { x: 7.65, y: 4.85, w: 1.5, h: 0.4, fontSize: 11, fontFace: FONT_BODY, color: C.lgray, align: "left" });

  s1.addNotes("Welcome everyone. Today I'm presenting my UCS645 project on accelerating multi-agent reinforcement learning for warehouse robotics using CUDA. Our GPU-native architecture achieves a 29.6x speedup over CPU serial baseline and processes over 17 billion state evaluations per second.");

  // ===================================================================
  // SLIDE 2: PROBLEM STATEMENT
  // ===================================================================
  let s2 = pres.addSlide();
  s2.background = { color: C.bg };
  addTopBar(s2);
  addSlideNumber(s2, 2);

  s2.addText("The Challenge: Multi-Agent Warehouse Navigation", {
    x: 0.8, y: 0.3, w: 8.4, h: 0.6, fontSize: 26, fontFace: FONT_TITLE, color: C.white, bold: true
  });

  // Three problem cards
  const problems = [
    { icon: icons.warehouse, title: "State-Space Explosion", desc: "20×20 grid × agent position × goal position = 160,000 distinct states for a SINGLE agent. CPU sequential processing cannot scale to real-time multi-agent coordination.", color: C.gold },
    { icon: icons.cpu, title: "PCIe Bottleneck", desc: "Standard RL frameworks split execution: CPU runs environment, GPU trains model. Continuous PCIe transfers dominate latency (~29.7 ms/step).", color: C.red },
    { icon: icons.robot, title: "Multi-Agent Deadlock", desc: "30 robots following greedy policies deadlock at choke points. Lock-free decentralized coordination with collision avoidance is essential.", color: C.cyan },
  ];

  problems.forEach((p, i) => {
    let cx = 0.5 + i * 3.15;
    s2.addShape(pres.shapes.RECTANGLE, {
      x: cx, y: 1.2, w: 2.9, h: 3.5, fill: { color: C.card }, shadow: mkCardShadow()
    });
    s2.addImage({ data: p.icon, x: cx + 0.95, y: 1.5, w: 0.45, h: 0.45 });
    s2.addShape(pres.shapes.RECTANGLE, { x: cx + 0.3, y: 2.15, w: 2.3, h: 0.03, fill: { color: p.color } });
    s2.addText(p.title, { x: cx + 0.3, y: 2.35, w: 2.3, h: 0.5, fontSize: 14, fontFace: FONT_BODY, color: C.white, bold: true });
    s2.addText(p.desc, { x: cx + 0.3, y: 2.85, w: 2.3, h: 1.6, fontSize: 11, fontFace: FONT_BODY, color: C.lgray, valign: "top" });
  });

  s2.addText("Core question: Can we keep the ENTIRE RL data lifecycle on the GPU and eliminate CPU-GPU transfers?", {
    x: 1, y: 4.9, w: 8, h: 0.4, fontSize: 12, fontFace: FONT_BODY, color: C.cyan, italic: true, align: "center"
  });

  s2.addNotes("The warehouse navigation problem has three key challenges. First, the state space grows as N to the fourth for an N-by-N grid — that's 160,000 states for our 20x20 grid. Second, standard ML frameworks constantly shuttle data across the PCIe bus. Third, multiple agents competing for the same cells inevitably deadlock without careful coordination.");

  // ===================================================================
  // SLIDE 3: SYSTEM ARCHITECTURE
  // ===================================================================
  let s3 = pres.addSlide();
  s3.background = { color: C.bg };
  addTopBar(s3);
  addSlideNumber(s3, 3);

  s3.addText("System Architecture: 100% GPU-Resident", {
    x: 0.8, y: 0.3, w: 8.4, h: 0.6, fontSize: 26, fontFace: FONT_TITLE, color: C.white, bold: true
  });

  // Architecture image (full width)
  s3.addImage({ path: "/home/nimish/dev/vscode/parallel project/UCS645_ProjectReport_Template/images/architecture.png", x: 0.5, y: 1.1, w: 9, h: 2.8 });

  // Bottom highlights
  const archPts = [
    { label: "Kernel 1", desc: "initValueTable()\n160K State Init", color: C.cyan },
    { label: "Kernel 2", desc: "trainRLAgent()\nBellman Solver", color: C.green },
    { label: "Kernel 3", desc: "runSwarm()\nCollision Avoidance", color: C.gold },
    { label: "Output", desc: "CSV + Metrics\n+ Visualization", color: C.purple },
  ];

  archPts.forEach((pt, i) => {
    let cx = 0.6 + i * 2.4;
    s3.addShape(pres.shapes.RECTANGLE, { x: cx, y: 4.15, w: 2.1, h: 0.95, fill: { color: C.card }, shadow: mkCardShadow() });
    s3.addShape(pres.shapes.RECTANGLE, { x: cx, y: 4.15, w: 2.1, h: 0.04, fill: { color: pt.color } });
    s3.addText(pt.label, { x: cx + 0.15, y: 4.25, w: 1.8, h: 0.3, fontSize: 12, fontFace: FONT_BODY, color: pt.color, bold: true });
    s3.addText(pt.desc, { x: cx + 0.15, y: 4.55, w: 1.8, h: 0.5, fontSize: 10, fontFace: FONT_BODY, color: C.lgray });
  });

  s3.addText("Zero PCIe transfers — All data stays in GPU VRAM from initialization through output", {
    x: 1, y: 5.2, w: 8, h: 0.3, fontSize: 11, fontFace: FONT_BODY, color: C.gray, align: "center"
  });

  s3.addNotes("Our architecture has three CUDA kernels that execute entirely on the GPU. Kernel 1 initializes the 160,000-state value table. Kernel 2 is the core — it runs parallel value iteration, solving the Bellman equation for all states simultaneously. Kernel 3 runs the multi-agent swarm, using atomic CAS for lock-free collision avoidance. The CPU only issues kernel launches and collects final logs.");

  // ===================================================================
  // SLIDE 4: VALUE ITERATION ON GPU
  // ===================================================================
  let s4 = pres.addSlide();
  s4.background = { color: C.bg };
  addTopBar(s4);
  addSlideNumber(s4, 4);

  s4.addText("Massively Parallel Value Iteration", {
    x: 0.8, y: 0.3, w: 8.4, h: 0.6, fontSize: 26, fontFace: FONT_TITLE, color: C.white, bold: true
  });

  // Left: Bellman equation
  s4.addShape(pres.shapes.RECTANGLE, { x: 0.4, y: 1.1, w: 4.5, h: 2.6, fill: { color: C.card }, shadow: mkCardShadow() });
  s4.addImage({ data: icons.brain, x: 0.65, y: 1.3, w: 0.35, h: 0.35 });
  s4.addText("Bellman Optimality Equation", {
    x: 1.1, y: 1.3, w: 3.5, h: 0.4, fontSize: 14, fontFace: FONT_BODY, color: C.purple, bold: true
  });
  s4.addText([
    { text: "V*(s) = max [ R(s,a) + γ · V*(s') ]", options: { fontSize: 16, color: C.white, fontFace: "Consolas", breakLine: true } },
    { text: "", options: { fontSize: 8, breakLine: true } },
    { text: "s = (x, y, gx, gy)  —  agent + goal position", options: { fontSize: 10, color: C.gray, breakLine: true } },
    { text: "|S| = 20⁴ = 160,000 states", options: { fontSize: 10, color: C.gray, breakLine: true } },
    { text: "γ = 0.99  |  R_step = -1  |  R_goal = +100", options: { fontSize: 10, color: C.gray, breakLine: true } },
    { text: "", options: { fontSize: 8, breakLine: true } },
    { text: "One CUDA thread per state → 160K threads", options: { fontSize: 11, color: C.green, bold: true, breakLine: true } },
    { text: "625 blocks × 256 threads = full coverage", options: { fontSize: 11, color: C.lgray } },
  ], { x: 0.65, y: 1.85, w: 4.0, h: 1.8, valign: "top" });

  // Right: Performance metrics
  s4.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 1.1, w: 4.4, h: 2.6, fill: { color: C.card }, shadow: mkCardShadow() });
  s4.addImage({ data: icons.gpu, x: 5.45, y: 1.3, w: 0.35, h: 0.35 });
  s4.addText("GPU Training Performance", {
    x: 5.9, y: 1.3, w: 3.5, h: 0.4, fontSize: 14, fontFace: FONT_BODY, color: C.green, bold: true
  });

  const perfStats = [
    { val: "0.545 ms", label: "60 Iterations × 160K States" },
    { val: "17.6B", label: "State Evaluations per Second" },
    { val: "625", label: "Thread Blocks (256 threads each)" },
    { val: "0", label: "PCIe Transfers During Training" },
  ];
  perfStats.forEach((s, i) => {
    let sy = 1.9 + i * 0.45;
    s4.addText(s.val, { x: 5.45, y: sy, w: 1.8, h: 0.3, fontSize: 16, fontFace: FONT_BODY, color: C.gold, bold: true });
    s4.addText(s.label, { x: 7.3, y: sy, w: 2.1, h: 0.3, fontSize: 10, fontFace: FONT_BODY, color: C.lgray });
  });

  // Bottom: Convergence
  s4.addImage({ path: "/home/nimish/dev/vscode/parallel project/UCS645_ProjectReport_Template/images/convergence.png", x: 0.4, y: 3.9, w: 9.2, h: 1.7 });

  s4.addNotes("The core innovation: we assign one CUDA thread to each of the 160,000 MDP states. Each thread independently evaluates all four actions, computes Q-values, and selects the maximum — all in parallel. After 60 iterations of ping-ponging between two value tables, the value function converges. Total training time: just 0.545 milliseconds.");

  // ===================================================================
  // SLIDE 5: COLLISION AVOIDANCE
  // ===================================================================
  let s5 = pres.addSlide();
  s5.background = { color: C.bg };
  addTopBar(s5);
  addSlideNumber(s5, 5);

  s5.addText("Lock-Free Multi-Agent Collision Avoidance", {
    x: 0.8, y: 0.3, w: 8.4, h: 0.6, fontSize: 26, fontFace: FONT_TITLE, color: C.white, bold: true
  });

  // Three mechanism cards
  const mechs = [
    { icon: icons.shield, title: "atomicCAS Locking", desc: "Each robot atomically claims its target cell using Compare-And-Swap. Only one robot can occupy a cell — others retry next step. Completely lock-free, no mutex overhead.", color: C.green },
    { icon: icons.diagram, title: "Occupancy Penalty (-100)", desc: "Cells occupied by other robots have their value reduced by 100 in the Q-value computation. This causes the value function to implicitly route agents around congestion zones.", color: C.orange },
    { icon: icons.cogs, title: "Random Tie-Breaking", desc: "Small random noise (~10⁻⁵) added to each action's Q-value prevents symmetric deadlocks where multiple robots choose identical paths simultaneously.", color: C.cyan },
  ];

  mechs.forEach((m, i) => {
    let cx = 0.35 + i * 3.2;
    s5.addShape(pres.shapes.RECTANGLE, { x: cx, y: 1.1, w: 3.0, h: 2.7, fill: { color: C.card }, shadow: mkCardShadow() });
    s5.addImage({ data: m.icon, x: cx + 0.2, y: 1.3, w: 0.4, h: 0.4 });
    s5.addText(m.title, { x: cx + 0.7, y: 1.35, w: 2.1, h: 0.35, fontSize: 14, fontFace: FONT_BODY, color: m.color, bold: true });
    s5.addShape(pres.shapes.RECTANGLE, { x: cx + 0.15, y: 1.85, w: 2.7, h: 0.02, fill: { color: m.color } });
    s5.addText(m.desc, { x: cx + 0.15, y: 2.0, w: 2.7, h: 1.6, fontSize: 10.5, fontFace: FONT_BODY, color: C.lgray, valign: "top" });
  });

  // Bottom code snippet
  s5.addShape(pres.shapes.RECTANGLE, { x: 0.4, y: 4.0, w: 9.2, h: 1.45, fill: { color: "0A1030" } });
  s5.addText([
    { text: "// CUDA kernel: collision avoidance logic", options: { fontSize: 9, color: C.gray, breakLine: true } },
    { text: "if (grid_occupancy[cell_idx] != -1 && grid_occupancy[cell_idx] != r.id) {", options: { fontSize: 11, color: C.lgray, fontFace: "Consolas", breakLine: true } },
    { text: "    val -= 100.0f;  // Heavy penalty for occupied cells", options: { fontSize: 11, color: C.orange, fontFace: "Consolas", breakLine: true } },
    { text: "}", options: { fontSize: 11, color: C.lgray, fontFace: "Consolas", breakLine: true } },
    { text: "val += (lcg(temp_state) % 1000) / 100000.0f;  // Random tie-break", options: { fontSize: 11, color: C.cyan, fontFace: "Consolas", breakLine: true } },
    { text: "if (atomicCAS(&grid_occupancy[cell_idx], -1, r.id) == -1) { move(); }", options: { fontSize: 11, color: C.green, fontFace: "Consolas" } },
  ], { x: 0.6, y: 4.1, w: 8.8, h: 1.3, valign: "top" });

  s5.addNotes("Collision avoidance uses three mechanisms working together. First, atomicCAS provides lock-free cell reservation. Second, we apply a heavy penalty of -100 to cells occupied by other robots in the value function, which causes agents to naturally route around each other. Third, tiny random noise breaks ties when multiple robots would otherwise choose identical paths — this is what finally solved our deadlock problem.");

  // ===================================================================
  // SLIDE 6: CPU BASELINE
  // ===================================================================
  let s6 = pres.addSlide();
  s6.background = { color: C.bg };
  addTopBar(s6);
  addSlideNumber(s6, 6);

  s6.addText("CPU Baseline: BFS-Based Pathfinding", {
    x: 0.8, y: 0.3, w: 8.4, h: 0.6, fontSize: 26, fontFace: FONT_TITLE, color: C.white, bold: true
  });

  // Left: BFS approach
  s6.addShape(pres.shapes.RECTANGLE, { x: 0.4, y: 1.1, w: 4.5, h: 3.4, fill: { color: C.card }, shadow: mkCardShadow() });
  s6.addImage({ data: icons.cpu, x: 0.65, y: 1.3, w: 0.4, h: 0.4 });
  s6.addText("CPU Serial (1 Thread) BFS", {
    x: 1.15, y: 1.35, w: 3.5, h: 0.35, fontSize: 14, fontFace: FONT_BODY, color: C.red, bold: true
  });
  s6.addText([
    { text: "Algorithm: Online BFS per robot, per step", options: { fontSize: 11, color: C.lgray, breakLine: true } },
    { text: "", options: { fontSize: 6, breakLine: true } },
    { text: "1. Shuffle robot order randomly each step", options: { fontSize: 11, color: C.lgray, fontFace: "Consolas", breakLine: true } },
    { text: "2. For each robot, build blocked set from:", options: { fontSize: 11, color: C.lgray, fontFace: "Consolas", breakLine: true } },
    { text: "   - Static obstacles (shelf columns)", options: { fontSize: 10, color: C.gray, fontFace: "Consolas", breakLine: true } },
    { text: "   - All other robots' current positions", options: { fontSize: 10, color: C.gray, fontFace: "Consolas", breakLine: true } },
    { text: "3. BFS shortest path to goal avoiding blocked cells", options: { fontSize: 11, color: C.lgray, fontFace: "Consolas", breakLine: true } },
    { text: "4. Claim first step of path via reservation system", options: { fontSize: 11, color: C.lgray, fontFace: "Consolas", breakLine: true } },
    { text: "5. If blocked, replan from scratch next step", options: { fontSize: 11, color: C.lgray, fontFace: "Consolas" } },
  ], { x: 0.65, y: 1.85, w: 4.0, h: 2.4, valign: "top" });

  // Right: Performance + OpenMP
  s6.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 1.1, w: 4.4, h: 1.8, fill: { color: C.card }, shadow: mkCardShadow() });
  s6.addText("Results", { x: 5.45, y: 1.2, w: 2, h: 0.35, fontSize: 14, fontFace: FONT_BODY, color: C.gold, bold: true });
  const cpuStats = [
    { val: "16.1 ms", label: "Total Wall Time" },
    { val: "30/30", label: "Deliveries Completed" },
    { val: "44 steps", label: "To Finish All Robots" },
    { val: "1.47", label: "Avg Steps per Delivery" },
  ];
  cpuStats.forEach((s, i) => {
    let sy = 1.6 + i * 0.32;
    s6.addText(s.val, { x: 5.45, y: sy, w: 1.5, h: 0.25, fontSize: 13, fontFace: FONT_BODY, color: C.white, bold: true });
    s6.addText(s.label, { x: 7.0, y: sy, w: 2.4, h: 0.25, fontSize: 10, fontFace: FONT_BODY, color: C.lgray });
  });

  // OpenMP box
  s6.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 3.1, w: 4.4, h: 1.4, fill: { color: C.card }, shadow: mkCardShadow() });
  s6.addText("OpenMP Multi-Threading", { x: 5.45, y: 3.2, w: 3.5, h: 0.3, fontSize: 13, fontFace: FONT_BODY, color: C.orange, bold: true });
  s6.addText("The baseline supports OpenMP parallelization via omp_set_num_threads(). On multi-core CPUs, robot processing can be distributed across threads for additional speedup beyond the serial baseline.", {
    x: 5.45, y: 3.55, w: 3.9, h: 0.85, fontSize: 10.5, fontFace: FONT_BODY, color: C.lgray, valign: "top"
  });

  // Bottom comparison note
  s6.addShape(pres.shapes.RECTANGLE, { x: 0.4, y: 4.7, w: 9.2, h: 0.55, fill: { color: C.card }, shadow: mkCardShadow() });
  s6.addText([
    { text: "Key Insight: ", options: { bold: true, color: C.gold } },
    { text: "CPU BFS is step-efficient (44 steps) but recomputes paths every step. GPU VI amortizes 0.545ms training across all simulation steps, enabling real-time replanning at scale.", options: { color: C.lgray } },
  ], { x: 0.6, y: 4.75, w: 8.8, h: 0.4, fontSize: 11, fontFace: FONT_BODY });

  s6.addNotes("For rigorous comparison, we implemented a CPU BFS-based baseline. Each robot runs BFS every step avoiding all other robots, with randomized processing order to prevent deadlocks. It achieves 30 out of 30 deliveries in just 44 steps with 16.1ms total time. However, BFS must be recomputed every step for every robot, while our GPU approach pre-computes a global value function once.");

  // ===================================================================
  // SLIDE 7: RESULTS - TRAINING PERFORMANCE
  // ===================================================================
  let s7 = pres.addSlide();
  s7.background = { color: C.bg };
  addTopBar(s7);
  addSlideNumber(s7, 7);

  s7.addText("Results: Training & Planning Performance", {
    x: 0.8, y: 0.3, w: 8.4, h: 0.6, fontSize: 26, fontFace: FONT_TITLE, color: C.white, bold: true
  });

  // Big stat callouts
  const bigStats = [
    { val: "0.545", unit: "ms", label: "GPU Training\n(60 VI Iterations)", color: C.green },
    { val: "16.1", unit: "ms", label: "CPU BFS Planning\n(Total Wall Time)", color: C.red },
    { val: "29.6x", unit: "", label: "GPU Speedup\nOver CPU Serial", color: C.gold },
    { val: "17.6B", unit: "", label: "State Evals/sec\nGPU Throughput", color: C.cyan },
  ];

  bigStats.forEach((bs, i) => {
    let cx = 0.2 + i * 2.45;
    s7.addShape(pres.shapes.RECTANGLE, { x: cx, y: 1.1, w: 2.25, h: 1.65, fill: { color: C.card }, shadow: mkCardShadow() });
    s7.addText([
      { text: bs.val, options: { fontSize: 32, color: bs.color, bold: true, fontFace: FONT_TITLE } },
      { text: bs.unit ? " " + bs.unit : "", options: { fontSize: 14, color: bs.color } },
    ], { x: cx + 0.1, y: 1.2, w: 2.05, h: 0.6, align: "center" });
    s7.addText(bs.label, { x: cx + 0.1, y: 1.95, w: 2.05, h: 0.6, fontSize: 10, fontFace: FONT_BODY, color: C.lgray, align: "center" });
  });

  // Comparison chart + table
  s7.addImage({ path: "/home/nimish/dev/vscode/parallel project/UCS645_ProjectReport_Template/images/speedup_bar.png", x: 0.3, y: 2.95, w: 5.5, h: 2.5 });

  // Right side table
  s7.addShape(pres.shapes.RECTANGLE, { x: 6.1, y: 2.95, w: 3.6, h: 2.5, fill: { color: C.card }, shadow: mkCardShadow() });
  s7.addText("Performance Table", { x: 6.3, y: 3.05, w: 3.2, h: 0.35, fontSize: 13, fontFace: FONT_BODY, color: C.white, bold: true });

  const tableRows = [
    [{ text: "Metric", options: { bold: true, color: C.white, fill: { color: C.bg2 }, fontSize: 10 } },
     { text: "CPU", options: { bold: true, color: C.red, fill: { color: C.bg2 }, fontSize: 10 } },
     { text: "GPU", options: { bold: true, color: C.green, fill: { color: C.bg2 }, fontSize: 10 } }],
    [{ text: "Time", options: { fontSize: 10, color: C.lgray } }, { text: "16.1 ms", options: { fontSize: 10, color: C.red } }, { text: "0.545 ms", options: { fontSize: 10, color: C.green } }],
    [{ text: "Speedup", options: { fontSize: 10, color: C.lgray } }, { text: "1.0x", options: { fontSize: 10, color: C.lgray } }, { text: "29.6x", options: { fontSize: 10, color: C.gold, bold: true } }],
    [{ text: "Throughput", options: { fontSize: 10, color: C.lgray } }, { text: "—", options: { fontSize: 10, color: C.lgray } }, { text: "17.6B/s", options: { fontSize: 10, color: C.cyan } }],
    [{ text: "Deliveries", options: { fontSize: 10, color: C.lgray } }, { text: "30/30", options: { fontSize: 10, color: C.red } }, { text: "30/30", options: { fontSize: 10, color: C.green } }],
  ];

  s7.addTable(tableRows, {
    x: 6.25, y: 3.5, w: 3.3, colW: [1.2, 1.0, 1.0],
    border: { pt: 0.5, color: C.bg2 },
    rowH: [0.3, 0.28, 0.28, 0.28, 0.28],
  });

  s7.addNotes("The GPU completes 60 value iterations over 160,000 states in just 0.545 milliseconds. That's 29.6 times faster than the CPU BFS baseline's planning time of 16.1 milliseconds. GPU throughput reaches 17.6 billion state evaluations per second. Both implementations achieve 100% delivery completion.");

  // ===================================================================
  // SLIDE 8: RESULTS - SWARM SIMULATION
  // ===================================================================
  let s8 = pres.addSlide();
  s8.background = { color: C.bg };
  addTopBar(s8);
  addSlideNumber(s8, 8);

  s8.addText("Results: Swarm Simulation Outcomes", {
    x: 0.8, y: 0.3, w: 8.4, h: 0.6, fontSize: 26, fontFace: FONT_TITLE, color: C.white, bold: true
  });

  // Active robots chart
  s8.addImage({ path: "/home/nimish/dev/vscode/parallel project/UCS645_ProjectReport_Template/images/gpu_vs_cpu_swarm.png", x: 0.2, y: 1.0, w: 5.3, h: 2.5 });

  // Delivery analysis
  s8.addImage({ path: "/home/nimish/dev/vscode/parallel project/UCS645_ProjectReport_Template/images/delivery_analysis.png", x: 5.6, y: 1.0, w: 4.2, h: 2.5 });

  // Bottom: Trade-off analysis card
  s8.addShape(pres.shapes.RECTANGLE, { x: 0.3, y: 3.7, w: 9.4, h: 1.75, fill: { color: C.card }, shadow: mkCardShadow() });
  s8.addText("Performance Trade-off Analysis", { x: 0.5, y: 3.8, w: 5, h: 0.35, fontSize: 15, fontFace: FONT_BODY, color: C.gold, bold: true });

  const tradeoffCols = [
    { title: "CPU BFS (Online)", items: ["Optimal shortest paths every step", "Full knowledge of all robot positions", "44 steps for 30 deliveries (efficient)", "BUT: O(grid² × robots) per step", "Cost scales poorly with more agents"], color: C.red },
    { title: "GPU VI (Offline)", items: ["Pre-computed global value function", "Static occupancy penalty approximation", "150 steps for 30 deliveries", "BUT: 0.545ms training amortized", "Scales to larger grids and robot counts"], color: C.green },
  ];

  tradeoffCols.forEach((col, i) => {
    let cx = 0.5 + i * 4.7;
    s8.addText(col.title, { x: cx, y: 4.2, w: 4.3, h: 0.3, fontSize: 12, fontFace: FONT_BODY, color: col.color, bold: true });
    col.items.forEach((item, j) => {
      s8.addText(`• ${item}`, { x: cx + 0.1, y: 4.5 + j * 0.2, w: 4.2, h: 0.2, fontSize: 9.5, fontFace: FONT_BODY, color: C.lgray });
    });
  });

  s8.addNotes("Both approaches achieve 30 out of 30 deliveries. The CPU BFS is actually more step-efficient — 44 steps vs 150 — because it computes optimal paths with full knowledge of all robot positions at every step. However, the GPU approach amortizes its one-time 0.545ms training cost, making it far more scalable as the number of agents and grid size increase.");

  // ===================================================================
  // SLIDE 9: VISUALIZATIONS
  // ===================================================================
  let s9 = pres.addSlide();
  s9.background = { color: C.bg };
  addTopBar(s9);
  addSlideNumber(s9, 9);

  s9.addText("Visualization & Analysis Suite", {
    x: 0.8, y: 0.3, w: 8.4, h: 0.6, fontSize: 26, fontFace: FONT_TITLE, color: C.white, bold: true
  });

  // 2x2 image grid
  const vizImgs = [
    { path: "/home/nimish/dev/vscode/parallel project/UCS645_ProjectReport_Template/images/active_robots.png", label: "Active Robots Timeline", x: 0.3, y: 1.1, w: 4.5, h: 1.65 },
    { path: "/home/nimish/dev/vscode/parallel project/UCS645_ProjectReport_Template/images/congestion_heatmap.png", label: "Congestion Heatmap", x: 5.2, y: 1.1, w: 4.5, h: 1.65 },
    { path: "/home/nimish/dev/vscode/parallel project/UCS645_ProjectReport_Template/images/state_space_scaling.png", label: "State-Space Scaling (O(N⁴))", x: 0.3, y: 2.85, w: 4.5, h: 1.65 },
    { path: "/home/nimish/dev/vscode/parallel project/UCS645_ProjectReport_Template/images/pcie_vs_zerocopy.png", label: "PCIe Bottleneck Analysis", x: 5.2, y: 2.85, w: 4.5, h: 1.65 },
  ];

  vizImgs.forEach(vi => {
    s9.addImage({ path: vi.path, x: vi.x, y: vi.y, w: vi.w, h: vi.h });
    s9.addText(vi.label, { x: vi.x, y: vi.y + vi.h - 0.3, w: vi.w, h: 0.25, fontSize: 9, fontFace: FONT_BODY, color: C.gold, align: "center" });
  });

  // Animation + MP4 note
  s9.addShape(pres.shapes.RECTANGLE, { x: 0.3, y: 4.65, w: 9.4, h: 0.8, fill: { color: C.card }, shadow: mkCardShadow() });
  s9.addImage({ data: icons.video, x: 0.5, y: 4.78, w: 0.35, h: 0.35 });
  s9.addText([
    { text: "Animation Output: ", options: { bold: true, color: C.gold } },
    { text: "Interactive HTML (warehouse_sim.html) + MP4 Video (warehouse_sim.mp4) generated via matplotlib FuncAnimation at 10 FPS. ", options: { color: C.lgray } },
    { text: "30 robots navigate the warehouse with color-coded paths, despawn on delivery, and the animation clearly shows collision avoidance in action.", options: { color: C.lgray } },
  ], { x: 0.95, y: 4.75, w: 8.5, h: 0.6, fontSize: 11, fontFace: FONT_BODY });

  s9.addNotes("Our visualization pipeline generates multiple analysis artifacts. The HTML animation shows the full swarm simulation with robots navigating around obstacles and each other. The MP4 video can be embedded in presentations. Static charts show active robot timelines, congestion patterns, state-space scaling behavior, and the PCIe bottleneck breakdown.");

  // ===================================================================
  // SLIDE 10: CONCLUSION & FUTURE WORK
  // ===================================================================
  let s10 = pres.addSlide();
  s10.background = { color: C.bg };
  s10.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.gold } });
  s10.addShape(pres.shapes.RECTANGLE, { x: 0, y: 5.565, w: 10, h: 0.06, fill: { color: C.cyan } });
  addSlideNumber(s10, 10);

  s10.addText("Conclusion & Future Work", {
    x: 0.8, y: 0.3, w: 8.4, h: 0.6, fontSize: 28, fontFace: FONT_TITLE, color: C.white, bold: true, align: "center"
  });

  // Achievements (left)
  s10.addShape(pres.shapes.RECTANGLE, { x: 0.3, y: 1.1, w: 4.6, h: 3.1, fill: { color: C.card }, shadow: mkCardShadow() });
  s10.addImage({ data: icons.rocket, x: 0.5, y: 1.25, w: 0.35, h: 0.35 });
  s10.addText("Key Achievements", { x: 0.95, y: 1.25, w: 3.5, h: 0.4, fontSize: 16, fontFace: FONT_BODY, color: C.gold, bold: true });

  const achievements = [
    "29.6× GPU speedup over CPU serial baseline",
    "100% delivery completion (30/30 robots)",
    "17.6B state evaluations per second throughput",
    "Zero-copy VRAM architecture eliminates PCIe bottleneck",
    "Lock-free collision avoidance with atomicCAS",
    "Complete visualization: HTML animation + MP4 + 10 analysis charts",
    "Rigorous CPU BFS baseline with OpenMP support",
  ];
  achievements.forEach((a, i) => {
    s10.addText(`▸ ${a}`, { x: 0.5, y: 1.8 + i * 0.32, w: 4.2, h: 0.3, fontSize: 11, fontFace: FONT_BODY, color: C.lgray });
  });

  // Future work (right)
  s10.addShape(pres.shapes.RECTANGLE, { x: 5.1, y: 1.1, w: 4.6, h: 3.1, fill: { color: C.card }, shadow: mkCardShadow() });
  s10.addImage({ data: icons.brain, x: 5.3, y: 1.25, w: 0.35, h: 0.35 });
  s10.addText("Future Directions", { x: 5.75, y: 1.25, w: 3.5, h: 0.4, fontSize: 16, fontFace: FONT_BODY, color: C.purple, bold: true });

  const futures = [
    { t: "Multi-GPU Domain Decomposition", d: "Partition warehouse into zones with NCCL boundary communication" },
    { t: "Deep RL Extension", d: "Neural value function approximator for continuous state spaces" },
    { t: "Dynamic Obstacle Handling", d: "Real-time replanning for humans, forklifts, moving obstacles" },
    { t: "Priority-Aware Scheduling", d: "Differentiated rewards for HIGH/MEDIUM/LOW urgency deliveries" },
    { t: "Battery-Constrained Navigation", d: "Finite energy model with autonomous charging station routing" },
    { t: "Hybrid GPU VI + CPU BFS", d: "Combine pre-computed value function with online BFS replanning" },
  ];
  futures.forEach((f, i) => {
    s10.addText(f.t, { x: 5.3, y: 1.8 + i * 0.38, w: 4.2, h: 0.2, fontSize: 11, fontFace: FONT_BODY, color: C.white, bold: true });
    s10.addText(f.d, { x: 5.3, y: 2.0 + i * 0.38, w: 4.2, h: 0.18, fontSize: 9, fontFace: FONT_BODY, color: C.gray });
  });

  // Bottom banner
  s10.addShape(pres.shapes.RECTANGLE, { x: 0.3, y: 4.4, w: 9.4, h: 0.7, fill: { color: C.card }, shadow: mkCardShadow() });
  s10.addText("GPU-native model-based RL achieves 29.6× speedup with 100% delivery success. The zero-copy architecture is ready for real-world warehouse deployment at scale.", {
    x: 0.5, y: 4.5, w: 9.0, h: 0.5, fontSize: 12, fontFace: FONT_BODY, color: C.lgray, align: "center", italic: true
  });

  s10.addText("Thank You!  |  Questions?", {
    x: 1, y: 5.2, w: 8, h: 0.35, fontSize: 16, fontFace: FONT_TITLE, color: C.white, align: "center", bold: true
  });

  s10.addNotes("To summarize: our CUDA-native architecture achieves 29.6x speedup with 100% delivery completion. Future work includes multi-GPU scaling for larger warehouses, deep RL for continuous state spaces, and battery-constrained navigation for real-world deployment. Thank you — I'm happy to take questions.");

  // ===================================================================
  // WRITE FILE
  // ===================================================================
  await pres.writeFile({ fileName: "/home/nimish/dev/vscode/parallel project/UCS645_Presentation.pptx" });
  console.log("Presentation saved: UCS645_Presentation.pptx");
}

main().catch(err => { console.error(err); process.exit(1); });
