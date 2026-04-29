# Amazon Warehouse CUDA Simulation

This project simulates a massive swarm of autonomous warehouse robots using a custom CUDA backend for parallel physics/movement calculation.

## How to Run in Colab

If you are using the Colab extension in VS Code, create a new Jupyter Notebook (`.ipynb` file) inside this folder and run the following cells:

### Cell 1: Compile the CUDA Code
```bash
!nvcc -o warehouse_sim main.cu
```

### Cell 2: Run the Simulation
```bash
!./warehouse_sim
```
*(This will generate an `output.csv` file containing the robot positions for every timestep).*

### Cell 3: Visualize the Swarm
```python
# Run the python visualization script
!pip install pandas matplotlib numpy
%run visualize.py
```
*(This will generate a `warehouse_sim.html` video file and display the animation right inside your notebook).*

---

## Next Steps for the Project
1. **Collision Detection:** Update `main.cu` so robots cannot move into a cell that is already occupied.
2. **Task Assignment:** Give robots "goals" (X, Y coordinates) to reach instead of moving randomly.
3. **Reinforcement Learning:** Implement a Q-table or simple Neural Network inside CUDA to teach the robots the fastest route to their goals while avoiding traffic.
