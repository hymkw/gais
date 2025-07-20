"""
GAIS (Governance & AI Symbiosis) Framework Simulation
Reproduces Figure 2 from "Friendly AI Symbiosis" paper

This script runs a Monte Carlo simulation to estimate the probability of catastrophic risk (CR)
across a grid of governance (G) and machine ethics investment (M) parameters.

Author: Anonymous (AAAI-26 submission)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ace_tools import display_dataframe_to_user

# ----------------------------------------------------------------------------
# Simulation Configuration
# ----------------------------------------------------------------------------
# Coefficients controlling dynamic interactions between human conflict (HC),
# unstoppable technology (UT), and friendly AI alignment (FAI).
ALPHA = 0.05    # Effect of UT on increasing human conflict
BETA = 0.05     # Effect of FAI on reducing human conflict
GAMMA = 0.05    # Effect of HC accelerating UT development
DELTA = 0.10    # Governance strength reducing UT growth
ETA = 0.40      # Probability of control failure as a function of UT
LAMBDA = 0.08   # Effect of machine ethics investment on increasing FAI
MU = 0.06       # Effect of self-preservation drive decreasing FAI
SP = 0.5        # Constant self-preservation drive

# Simulation parameters
N_RUNS = 150    # Number of simulation runs per (G, M) combination
T_MAX = 50      # Maximum discrete time steps per run
THETA = 0.4     # Control failure threshold for catastrophe
PHI = 1.0       # Human conflict threshold for catastrophe

# Sweep grid for governance (G) and machine ethics (M) levels, from 0.0 to 0.8
G_VALUES = np.linspace(0.0, 0.8, 5)
M_VALUES = np.linspace(0.0, 0.8, 5)

# ----------------------------------------------------------------------------
# Single Simulation Function
# ----------------------------------------------------------------------------
def run_single_simulation(G, M):
    """
    Execute one Monte Carlo trial of the GAIS model for given G and M.

    Args:
        G (float): Governance strength [0, 1]
        M (float): Machine ethics investment level [0, 1]

    Returns:
        bool: True if a catastrophic event occurs within T_MAX, else False
    """
    # Initialize state variables randomly within specified ranges
    HC = np.random.uniform(0.1, 0.3)  # Human conflict intensity
    UT = np.random.uniform(0.1, 0.3)  # Unstoppable technology advancement
    FAI = np.random.uniform(0.0, 0.2) # Friendly AI alignment level

    for _ in range(T_MAX):
        # Calculate next-step dynamics
        HC_new = HC + ALPHA * UT - BETA * FAI
        UT_new = UT + GAMMA * HC - DELTA * G
        CF = ETA * UT * (1 - G)              # Instantaneous control failure metric
        FAI_new = FAI + LAMBDA * M - MU * SP  # Update alignment considering machine ethics and self-preservation

        # Check catastrophe criteria: control failure or runaway conflict
        if CF > THETA or HC_new > PHI:
            return True

        # Update state variables, clamping to valid ranges
        HC = max(0.0, HC_new)
        UT = max(0.0, UT_new)
        FAI = max(0.0, min(1.0, FAI_new))

    # No catastrophe within time horizon
    return False

# ----------------------------------------------------------------------------
# Parameter Sweep and Aggregation
# ----------------------------------------------------------------------------
# Prepare matrix to record catastrophic risk probability for each grid point
results = np.zeros((len(M_VALUES), len(G_VALUES)))

for i, M in enumerate(M_VALUES):
    for j, G in enumerate(G_VALUES):
        catastrophic_count = 0
        # Run multiple trials for statistical estimation
        for _ in range(N_RUNS):
            if run_single_simulation(G, M):
                catastrophic_count += 1
        # Probability of catastrophe = fraction of runs triggering CR
        results[i, j] = catastrophic_count / N_RUNS

# ----------------------------------------------------------------------------
# Display Results as a DataFrame
# ----------------------------------------------------------------------------
# Construct labeled DataFrame for easy reading
df = pd.DataFrame(
    results,
    index=[f"M={m:.1f}" for m in M_VALUES],
    columns=[f"G={g:.1f}" for g in G_VALUES]
)
display_dataframe_to_user("Catastrophic Risk Probability Matrix", df)

# ----------------------------------------------------------------------------
# Visualization: Heatmap of Catastrophic Risk
# ----------------------------------------------------------------------------
plt.figure()
plt.imshow(
    results,
    origin='lower',             # Align bottom-left corner as (G[0], M[0])
    extent=[G_VALUES[0], G_VALUES[-1], M_VALUES[0], M_VALUES[-1]],
    aspect='auto'               # Stretch to fill axes
)
plt.xlabel('Governance Level (G)')
plt.ylabel('Machine Ethics Level (M)')
plt.title('Catastrophic Risk Probability')
plt.colorbar(label='CR Probability')  # Legend for risk values
plt.tight_layout()
plt.show()  
