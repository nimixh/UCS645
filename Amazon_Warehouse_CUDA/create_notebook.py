import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Amazon Warehouse CUDA Simulation\n",
    "Run the cells below to compile the CUDA code, execute the Reinforcement Learning swarm simulation, and visualize the results."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "!nvcc -o warehouse_sim main.cu\n",
    "!./warehouse_sim"
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
  }
 ],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 4
}

with open('Amazon_Warehouse_Simulation.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
