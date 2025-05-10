"""
Simulation of Deep Brain Stimulation (DBS).

Uses the Forced Temporal Spike-Time Stimulation (FTSTS) DBS strategy.
"""

import numpy as np
from config import SimulationConfig
from dbssim import run_simulation
from neural_model import NeuralModel
from kuramoto_syn import kuramoto_syn
from plotting import plot_kop, plot_avg_synaptic_weight, plot_synchrony


def main():
    """Example run"""

    # todo: idk how to handle shared params yet
    # for now, i will consolidate them
    simconfig = SimulationConfig(
        duration=5000,
        step_size=0.1,
        sample_duration=20,
        seed=42,
    )

    neural_model = NeuralModel(
        shared_params=simconfig,
        num_e=160,
        num_i=50,
        seed=42,
    )

    data = run_simulation(
        config=simconfig,
        model=neural_model,
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
