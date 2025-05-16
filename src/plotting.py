"""
Code to plot data from the DBS simulations.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_pulsatile_input(Ue, Ui, duration, step_size):
    """
    Plot the Ue and Ui inputs.
    """

    num_steps = int(duration / step_size)
    t = [i * step_size for i in range(num_steps)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].plot(t, Ui, label='Ui', color='red')
    axes[0].set_title('FTSTS Waveform for inhibitory (I) neurons')
    axes[0].set_xlabel('time (s)')
    axes[0].set_ylabel('Vstim(t) (V)')
    axes[0].set_xlim([0, duration / 1000])
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].legend()

    axes[1].plot(t, Ue, label='Ue', color='blue')
    axes[1].set_title('FTSTS Waveform for excitatory (E) neurons')
    axes[1].set_xlabel('time (s)')
    axes[1].set_ylabel('Vstim(t) (V)')
    axes[1].set_xlim([0, duration / 1000])
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_kop(t, re):
    plt.figure(figsize=(10, 5))
    plt.plot(t / 1000, re, label='Kuramoto Order Parameter', color='blue')
    plt.title('Time Series Kuramoto Order Parameter', fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('KOP', fontsize=12)
    # plt.xlim([0.2, 0.8])
    plt.ylim([0, 1.1])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()


def plot_avg_synaptic_weight(time, J_I, W_IE, duration):
    plt.figure(figsize=(10, 5))
    plt.plot(time / 1000, J_I * W_IE,
             label='Average Synaptic Weight', color='blue')
    plt.title('Time Series Average Synaptic Weight', fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Average Synaptic Weight (IE)', fontsize=12)
    plt.xlim([0, duration / 1000])
    plt.ylim([0, 300])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()


def plot_synchrony(syn):
    t = np.arange(len(syn))
    plt.figure(figsize=(10, 4))
    plt.title("Local Synchrony Measurements")
    plt.plot(t, syn)
    plt.xlabel("Time")
    plt.ylabel("Synchrony")
    plt.tight_layout()
    plt.show()


def plot_spike_patterns(spike_e, spike_i, step_size):
    """
    Plot the spiking patterns before, during, and after the FTSTS protocol,
    respectively.
    """

    # time_windows = [  # todo: change with simulation params
    #     (200, 600),
    #     (5000, 5400),
    #     (24000, 24400),
    # ]
    time_windows = [
        (0, 350),
        (400, 750),
        (4650, 5000),
    ]

    num_e = spike_e.shape[1]
    num_i = spike_i.shape[1]

    fig, axs = plt.subplots(
        len(time_windows),
        1,
        figsize=(12, 4 * len(time_windows)),
        sharey=True,
    )

    if len(time_windows) == 1:
        acs = [axs]  # ensure iterable

    for ax, (start_ms, end_ms) in zip(axs, time_windows):
        a = int(start_ms / step_size)
        b = int(end_ms / step_size)

        time_range = np.arange(a, b) * step_size / 1000  # (s)
        spikes_e = spike_e[a:b, :]
        spikes_i = spike_i[a:b, :]
        te, ne = np.nonzero(spikes_e)
        ti, ni = np.nonzero(spikes_i)

        te_time = (a + te) * step_size / 1000
        ti_time = (a + ti) * step_size / 1000

        ax.plot(te_time, ne + 1, 'k.', markersize=1)
        ax.plot(ti_time, num_e + ni + 1, 'r.', markersize=1)

        ax.set_xlim(start_ms / 1000, end_ms / 1000)
        ax.set_ylim(0.9, num_e + num_i + 0.1)
        ax.set_ylabel('Neuron Index')
        ax.set_xlabel('Time (s)')
        ax.set_title(f'Spike Raster Plot: {start_ms}–{end_ms} ms')

    plt.tight_layout()
    plt.show()


def plot_avg_synaptic_input(S_EI, S_IE, duration, step_size):
    num_steps = int(duration / step_size)
    t = [i * step_size for i in range(num_steps)]

    plt.figure(figsize=(10, 5))
    plt.plot(t, S_EI, 'k', t, S_IE, 'r')
    plt.title('Time Series Average Synaptic Input', fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Average Synaptic Input', fontsize=12)
    plt.xlim([0, duration / 1000])
    plt.ylim([-0.5, 0.5])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(['I-to-E', 'E-to-I'])
    plt.show()

    # plt.figure(3)
    # plt.plot(t, S_EI, 'k', t, S_IE, 'r')
    # plt.legend(['I-to-E', 'E-to-I'])
    # plt.xlabel('Time (sec)')
    # plt.ylabel('Average Synaptic Input')


# def todo():

#     # t = np.arange(0.1, duration+step_size, step_size)
#     t = np.linspace(0.1,  # precision error with np.arange
#                     duration,
#                     int(round((duration - 0.1) / step_size)) + 1)

#     # calculate the rates

#     # dt = 100
#     # [rate_E, rate_I, t_rate] = rate_calc(spike_E, spike_I, step, dt, N_E, N_I);
#     #
#     # plt.figure(4)
#     # plt.plot(t_rate/1000, 1000*rate_E, 'k', t_rate/1000, 1000*rate_I, 'r', linewidth=1.2)
#     # plt.ylabel('Average Firing Rate (sp/sec)')
#     # plt.xlabel('Time (sec)')
#     # plt.legend(['Excitatory Neurons', 'Inhibitory Neurons'])

#     # plt.figure(4)
#     # plt.plot(t/1000, spike_E, 'k.', t/1000, spike_I, 'r.')
#     # plt.xlim([400/1000, 500/1000])
#     # plt.ylim([0.9, 2000.1])
#     # plt.ylabel('Neuron Index')
#     # plt.xlabel('Time (sec)')
