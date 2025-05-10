"""
Simulation of Deep Brain Stimulation (DBS).

Uses the Forced Temporal Spike-Time Stimulation (FTSTS) DBS strategy.
"""
# pylint: disable=invalid-name

import os
import time
from datetime import datetime
import numpy as np
from tqdm import tqdm
from config import SimulationConfig
from neural_model import NeuralModel

PRESIM_STATE_DIR = "data/presim_state/"
POSTSIM_STATE_DIR = "data/postsim_state/"

INHIBITION_THRESHOLD = 75  # (mV) threshold for inhibition
STIMULATION_ONSET_TIME_RATIO = 0.08  # begin after 8% of the simulation time
PLASTICITY_ONSET_TIME_RATIO = 0.004  # begin after 0.4% of the simulation time
SAMPLE_RATIO = 0.0008  # sample window is 0.08% of the simulation time


def run_simulation(
        config: SimulationConfig,
        model: NeuralModel,
        cache: bool = False
):
    """
    Run a round of open-loop FTSTS simulation.
    """

    duration = config.duration
    step_size = config.step_size
    sample_duration = config.sample_duration
    num_steps = int(duration / step_size)
    num_samples = int(duration / sample_duration)
    num_steps_per_sample = int(sample_duration / step_size)

    if cache:
        os.makedirs(PRESIM_STATE_DIR, exist_ok=True)
        os.makedirs(POSTSIM_STATE_DIR, exist_ok=True)
        # os.mkdir(
        #     f"./data/ode/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        # )

    if cache:  # save presim state
        assert os.path.isdir(PRESIM_STATE_DIR)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{model.num_e}-{model.num_i}-{timestamp}.mat"
        path = os.path.join(PRESIM_STATE_DIR, filename)
        print(f"Saving presim state to {path}")
        model.save_state(path)

    stim_onset_time = int(duration * STIMULATION_ONSET_TIME_RATIO)
    plast_onset_time = int(duration * PLASTICITY_ONSET_TIME_RATIO)

    tic = time.time()

    for i in tqdm(range(1, num_samples + 1), "Simulating", unit="sample"):
        model.comp_time = (i - 1) * sample_duration

        # Calculate Average Effective Inhibition.
        if np.mean(model.w_ie0) * model.j_i < INHIBITION_THRESHOLD:
            model.cross_100 = 0

        # Toggle Stimulation and Plasticity.
        stim_on = ((i * sample_duration) >= stim_onset_time) * model.cross_100
        plast_on = (i * sample_duration) >= plast_onset_time
        assert stim_on in (0, 1)
        assert plast_on in (0, 1)

        # Define the Sample Window.
        sample_start = (i >= 2) * (i - 1) * num_steps_per_sample
        sample_end = i * num_steps_per_sample
        assert sample_end - sample_start == num_steps_per_sample

        # Define the Stimulation Input.
        vstim = 100  # mV (todo: rename: Ustim?)
        ue = vstim * model.u_e[:, sample_start:sample_end]
        ui = vstim * model.u_i[:, sample_start:sample_end]
        assert ue.shape[1] == num_steps_per_sample
        assert ui.shape[1] == num_steps_per_sample

        # Run the ODE Neuron Model.
        (
            step_times,
            v_e,
            v_i,
            s_ei,
            s_ie,
            x_ei,
            x_ie,
            apost,
            apre,
            w_ie,
            spike_e,
            spike_i,
            ref_e,
            ref_i,
            synchrony,
            spt_e,
            phif
        ) = model.step(
            ue=ue,
            ui=ui,
            plast_on=plast_on,
            stim_on=stim_on,
            percent_V_stim=1
        )

        # Recorded Variables.
        model.time_array[sample_start:sample_end, 0] = (
            step_times[0:-1, 0] + (i - 1) * sample_duration
        )
        model.w_ie[sample_start:sample_end, 0] = np.mean(
            w_ie[0:-1, :],
            axis=1
        )
        # model.spike_e[sample_start:sample_end, :] = spike_e_m[0:num_steps_per_sample, :]
        # model.spike_i[sample_start:sample_end, :] = spike_i_m[0:num_steps_per_sample, :]
        model.synchrony[i - 1, 0] = synchrony
        model.time_syn[i - 1, 0] = sample_duration * (i)
        model.spike_time_e[sample_start:sample_end, :] = spt_e[0:-1, :]

        # Update Initial Conditions (for next run).
        model.v_e0 = v_e[-1, :].reshape(1, -1)
        model.v_i0 = v_i[-1, :].reshape(1, -1)
        model.s_ei0 = s_ei[-1, :].reshape(1, -1)
        model.s_ie0 = s_ie[-1, :].reshape(1, -1)
        model.x_ei0 = x_ei[-1, :].reshape(1, -1)
        model.x_ie0 = x_ie[-1, :].reshape(1, -1)
        model.apost0 = apost[-1, :].reshape(1, -1)
        model.apre0 = apre[-1, :].reshape(1, -1)
        model.w_ie0 = w_ie[-1, :].reshape(1, -1)
        model.w_ei0 = model.weight_0
        offset = num_steps_per_sample - int(5 / step_size)
        model.leftover_s_ei = s_ei[offset:-1, :]
        model.leftover_s_ie = s_ie[offset:-1, :]
        model.spt_e0 = spt_e[-1, :]

    minute = (time.time() - tic) / 60
    print('Simulation complete.')
    print("Run time (minutes):", minute)

    if cache:  # save postsim state
        assert os.path.isdir(POSTSIM_STATE_DIR)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{model.num_e}-{model.num_i}-{timestamp}.mat"
        path = os.path.join(POSTSIM_STATE_DIR, filename)
        print(f"Saving postsim state to {path}")
        model.save_state(path)

    return (  # for example run
        model.spike_time_e,
        step_size,
        duration,
        model.num_e,
        model.j_i,
        model.w_ie,
        model.synchrony,
    )
