"""
Simulation of Deep Brain Stimulation (DBS).

Uses the Forced Temporal Spike-Time Stimulation (FTSTS) DBS strategy.
"""

import numpy as np
from dbssim import run_simulation
from kuramoto_syn import kuramoto_syn
from plotting import plot_kop, plot_avg_synaptic_weight, plot_synchrony


def main():
    """Example run"""

    data = run_simulation(
        duration=5000,
        N_E=160,
        N_I=40,
        seed=42,
        cache=False,
    )

    sptime, step_size, duration, ne, J_I, W_IE, synchrony = data

    t = np.linspace(
        0.1,
        duration,
        int(round((duration - 0.1) / step_size)) + 1
    )
    t = np.ascontiguousarray(t, dtype=np.float64)

    # Compute Neuron Synchronization.
    re = kuramoto_syn(
        sptime=sptime,
        t=t,
        step_size=step_size,
        duration=duration,
        num_neurons=ne,
        fast=True,
    )

    plot_kop(t, re)
    plot_avg_synaptic_weight(t, J_I, W_IE, duration)
    plot_synchrony(synchrony)


if __name__ == "__main__":
    main()
