import json

with open('main.cu', 'r') as f:
    cuda_code = f.read()

with open('cpu_baseline.cpp', 'r') as f:
    cpu_code = f.read()

with open('visualize.py', 'r') as f:
    vis_code = f.read()

# The notebook
cells = [
    # Cell 0: Title
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Amazon Warehouse CUDA Simulation\n",
            "## UCS645: Parallel and Distributed Computing\n\n",
            "**Run all cells in order.** Each cell depends on the previous one.\n\n",
            "This notebook:\n",
            "1. Writes source files to the Colab VM\n",
            "2. Compiles & runs **CPU Serial** (1 thread) and **CPU OpenMP** baselines\n",
            "3. Compiles & runs **CUDA GPU** simulation\n",
            "4. Generates visualization + speedup comparison graphs\n",
            "5. Zips and downloads all outputs"
        ]
    },
    # Cell 1: Install deps
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!pip install -q pandas matplotlib numpy\n",
            "!apt-get install -y -qq ffmpeg 2>/dev/null || true\n",
            "print(\"Dependencies ready.\")"
        ]
    },
    # Cell 2: Write source files
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "%%writefile main.cu\n" + cuda_code
        ]
    },
    # Cell 3: Write CPU baseline
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "%%writefile cpu_baseline.cpp\n" + cpu_code
        ]
    },
    # Cell 4: Write visualize
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "%%writefile visualize.py\n" + vis_code
        ]
    },
    # Cell 5: Compile & run CPU baseline
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 1: Compile & Run CPU Baseline"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!g++ -fopenmp -O3 -o cpu_baseline cpu_baseline.cpp -lm\n",
            "!./cpu_baseline"
        ]
    },
    # Cell 6: Compile & run GPU
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 2: Compile & Run CUDA GPU Simulation"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!nvcc -Wno-deprecated-gpu-targets -O3 -o warehouse_sim main.cu\n",
            "!./warehouse_sim"
        ]
    },
    # Cell 7: Visualize
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 3: Generate Animation"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "%run visualize.py"
        ]
    },
    # Cell 8: Speedup graphs
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 4: Speedup Comparison Graphs"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "from IPython.display import Image, display\n",
            "import os\n\n",
            "# Read metrics\n",
            "gpu_metrics = pd.read_csv('metrics.csv').set_index('metric')\n",
            "cpu_t1 = pd.read_csv('metrics_cpu_t1.csv').set_index('metric')\n",
            "import glob\n",
            "cpu_files = sorted(glob.glob('metrics_cpu_t*.csv'))\n",
            "cpu_omp = pd.read_csv(cpu_files[-1]).set_index('metric') if len(cpu_files) > 1 else cpu_t1\n\n",
            "# GPU uses 'training_time_ms', CPU uses 'total_time_ms'\n",
            "gpu_train = float(gpu_metrics.loc['training_time_ms', 'value'])\n",
            "cpu_serial_train = float(cpu_t1.loc['total_time_ms', 'value'])\n",
            "cpu_omp_train = float(cpu_omp.loc['total_time_ms', 'value'])\n",
            "omp_threads = int(cpu_omp.loc['num_threads', 'value'])\n\n",
            "print(f\"GPU training time:     {gpu_train:.2f} ms\")\n",
            "print(f\"CPU Serial (1t):       {cpu_serial_train:.1f} ms\")\n",
            "print(f\"CPU OpenMP ({omp_threads}t):       {cpu_omp_train:.1f} ms\")\n",
            "print(f\"GPU vs CPU Serial:     {cpu_serial_train/gpu_train:.1f}x speedup\")\n",
            "print(f\"GPU vs CPU OpenMP:     {cpu_omp_train/gpu_train:.1f}x speedup\")\n\n",
            "# Create comparison chart\n",
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n\n",
            "labels = [f'CPU Serial\\n(1 thread)', f'CPU OpenMP\\n({omp_threads} threads)', 'CUDA GPU\\n(Zero-Copy VRAM)']\n",
            "times = [cpu_serial_train, cpu_omp_train, gpu_train]\n",
            "colors = ['#e74c3c', '#f39c12', '#2ecc71']\n\n",
            "bars = ax1.bar(labels, times, color=colors, width=0.5, edgecolor='black', linewidth=1.5)\n",
            "ax1.set_ylabel('Training Time (ms)', fontsize=13)\n",
            "ax1.set_title('RL Training Time Comparison', fontsize=14, fontweight='bold')\n",
            "ax1.set_yscale('log')\n",
            "for bar, t in zip(bars, times):\n",
            "    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15,\n",
            "             f'{t:.1f} ms', ha='center', fontweight='bold', fontsize=12)\n\n",
            "speedups = [1.0, cpu_serial_train/max(cpu_omp_train,0.001), cpu_serial_train/max(gpu_train,0.001)]\n",
            "bars2 = ax2.bar(labels, speedups, color=colors, width=0.5, edgecolor='black', linewidth=1.5)\n",
            "ax2.set_ylabel('Speedup (vs CPU Serial)', fontsize=13)\n",
            "ax2.set_title('Speedup Over Serial Baseline', fontsize=14, fontweight='bold')\n",
            "ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)\n",
            "for bar, s in zip(bars2, speedups):\n",
            "    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(speedups)*0.02,\n",
            "             f'{s:.1f}x', ha='center', fontweight='bold', fontsize=12)\n\n",
            "plt.tight_layout()\n",
            "plt.savefig('speedup_comparison.png', dpi=150, bbox_inches='tight')\n",
            "plt.close()\n",
            "display(Image('speedup_comparison.png'))\n",
            "print('Saved speedup_comparison.png')\n\n",
            "# Throughput comparison\n",
            "vtable_size = 20**4\n",
            "gpu_tput = vtable_size * 60 / (gpu_train / 1000)\n",
            "cpu_serial_tput = vtable_size * 60 / (cpu_serial_train / 1000)\n",
            "cpu_omp_tput = vtable_size * 60 / (cpu_omp_train / 1000)\n\n",
            "print(f\"\\nThroughput (state evals/sec):\")\n",
            "print(f\"  CPU Serial: {cpu_serial_tput:,.0f}\")\n",
            "print(f\"  CPU OpenMP: {cpu_omp_tput:,.0f}\")\n",
            "print(f\"  CUDA GPU:   {gpu_tput:,.0f}\")"
        ]
    },
    # Cell 9: Deliveries comparison
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Delivery comparison\n",
            "gpu_deliveries = int(gpu_metrics.loc['total_deliveries', 'value']) if 'total_deliveries' in gpu_metrics.index else 'N/A'\n",
            "cpu_deliveries = int(cpu_t1.loc['deliveries', 'value']) if 'deliveries' in cpu_t1.index else 'N/A'\n\n",
            "print(f\"GPU deliveries:  {gpu_deliveries}\")\n",
            "print(f\"CPU deliveries:  {cpu_deliveries}\")\n\n",
            "# Active robots plot from GPU output\n",
            "if os.path.exists('output.csv'):\n",
            "    df = pd.read_csv('output.csv')\n",
            "    steps = sorted(df['step'].unique())\n",
            "    active = [len(df[(df['step']==s) & (df['x']!=-1)]) for s in steps]\n",
            "    deliveries = [len(df[(df['step']<=s) & (df['x']==-1) & (df['id']==i)]) for s in steps]\n",
            "    # Simpler: count despawned robots per step\n",
            "    cum_del = []\n",
            "    seen = set()\n",
            "    for s in steps:\n",
            "        frame = df[(df['step']==s) & (df['x']==-1)]\n",
            "        for rid in frame['id'].unique():\n",
            "            seen.add(rid)\n",
            "        cum_del.append(len(seen))\n",
            "    \n",
            "    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n",
            "    ax1.plot(steps, active, color='#e67e22', linewidth=2)\n",
            "    ax1.fill_between(steps, 0, active, alpha=0.15, color='#e67e22')\n",
            "    ax1.set_xlabel('Step', fontsize=12)\n",
            "    ax1.set_ylabel('Active Robots', fontsize=12)\n",
            "    ax1.set_title('Active Robots Over Time', fontsize=13, fontweight='bold')\n",
            "    ax1.grid(True, alpha=0.3)\n",
            "    ax2.plot(steps, cum_del, color='#27ae60', linewidth=2)\n",
            "    ax2.fill_between(steps, 0, cum_del, alpha=0.15, color='#27ae60')\n",
            "    ax2.set_xlabel('Step', fontsize=12)\n",
            "    ax2.set_ylabel('Deliveries', fontsize=12)\n",
            "    ax2.set_title('Cumulative Deliveries', fontsize=13, fontweight='bold')\n",
            "    ax2.grid(True, alpha=0.3)\n",
            "    plt.tight_layout()\n",
            "    plt.savefig('active_robots.png', dpi=150, bbox_inches='tight')\n",
            "    plt.close()\n",
            "    display(Image('active_robots.png'))\n",
            "    print('Saved active_robots.png')"
        ]
    },
    # Cell 10: Download all
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 5: Download All Outputs"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import glob, zipfile\n",
            "from google.colab import files\n\n",
            "# Create zip of all outputs\n",
            "outputs = ['output.csv', 'metrics.csv', 'metrics_cpu_t1.csv']\n",
            "outputs += glob.glob('metrics_cpu_t*.csv')\n",
            "outputs += ['warehouse_sim.html', 'warehouse_sim.mp4']\n",
            "outputs += ['speedup_comparison.png', 'active_robots.png']\n",
            "outputs = [f for f in outputs if os.path.exists(f)]\n\n",
            "with zipfile.ZipFile('warehouse_outputs.zip', 'w') as zf:\n",
            "    for f in outputs:\n",
            "        zf.write(f)\n",
            "        print(f\"  + {f} ({os.path.getsize(f)/1024:.1f} KB)\")\n\n",
            "print(f\"\\n{len(outputs)} files zipped.\")\n",
            "files.download('warehouse_outputs.zip')"
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": "Amazon_Warehouse_CUDA.ipynb", "provenance": []}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('Amazon_Warehouse_Simulation_Colab.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook written: Amazon_Warehouse_Simulation_Colab.ipynb")
