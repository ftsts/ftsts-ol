"""
TODO: docstring
"""
# pylint: disable=invalid-name

import numpy as np

V_REST = 0  # (mV) resting potential
V_THRESHOLD = 20  # (mV) threshold potential for action potential
V_RESET = 14  # (mV) reset potential after action potential
REFRACTORY = 2  # (ms) refractory period after action potential
MAX_WEIGHT_IE = 10  # (mV) maximum synaptic weight for E->I connections
MAX_WEIGHT_EI = 290  # (mV) maximum synaptic weight for I->E connections


def _dVdt(v, z, mu, sigma, tau_m, x, stim):
    """TODO: docstring"""
    # todo: link to paper/equation
    # todo: include equation in docstring

    # (eq1&2): τm dV(t)/dt = -V(t) + Z(t) + μ + (σ * sqrt(τm) * X(t)) + Vstim(t)

    return (
        V_REST - v + z + mu + (sigma * np.sqrt(tau_m) * x) + stim
    ) / tau_m


def ode_neuron_model(
        plast_on,
        ON,
        vE0,
        vI0,
        S_EI0,
        S_IE0,
        X_EI0,
        X_IE0,
        Apost0,
        Apre0,
        W_IE0,
        W_EI0,
        mew_e,
        sigma_e,
        ue,
        ui,
        mew_i,
        sigma_i,
        J_E,
        J_I,
        C_E,
        C_I,
        tau_LTP,
        tau_LTD,
        step_size,
        sample_duration,
        N_E,
        N_I,
        S_key_EI,
        S_key_IE,
        leftover_S_EI,
        leftover_S_IE,
        ref_E,
        ref_I,
        tau_E_m,
        tau_I_m,
        percent_V_stim,
        comp_time,
        spt_E0,
        phif,
):
    """
    Function to simulate a network of excitatory and inhibitory neurons with synaptic plasticity.
    tau_E_m: membrane time constant for excitatory neurons
    """

    num_steps = int(sample_duration / step_size)  # steps per sample

    # Neuron Parameters.
    # tau_E_m = 10; %ms
    # tau_I_m = 10;
    tau_d = 1
    tau_r = 1
    vrest = 0  # mV
    whitenoise_E = np.random.randn(num_steps + 1, N_E)  # X(t)
    whitenoise_I = np.random.randn(num_steps + 1, N_I)  # X(t)
    vreset = 14
    refractory = 2  # ms
    # ref_E = zeros(1, N_E);
    # ref_I = zeros(1, N_I);

    # Synaptic Parameters.
    WEI = W_EI0
    syn_delay = 5  # ms
    num_synapses_IE = int(np.max(S_key_IE))
    num_synapses_EI = int(np.max(S_key_EI))

    # Plasticity Parameters.
    dApre_0 = 0.005*1
    dApost_0 = dApre_0*1
    Apre_i = 0
    Apost_i = 0
    a_LTD = -1 * plast_on * 1.1
    a_LTP = 1 * plast_on * 1
    eta = 0.25

    # Initialize State Vectors.
    # todo: can reduce memory with curr, next instead of full arrays
    # todo: maybe not... np.mean(v_e, axis=1)
    v_E = np.zeros((num_steps + 1, N_E))
    v_I = np.zeros((num_steps + 1, N_I))
    S_EI = np.zeros((num_steps + 1, N_E))
    S_IE = np.zeros((num_steps + 1, N_I))
    X_EI = np.zeros((num_steps + 1, N_E))
    X_IE = np.zeros((num_steps + 1, N_I))
    Apost = np.zeros((num_steps + 1, num_synapses_IE))
    Apre = np.zeros((num_steps + 1, num_synapses_IE))
    W_IE = np.zeros((num_steps + 1, num_synapses_IE))
    time = np.zeros((num_steps + 1, 1))
    spike_E = np.zeros((num_steps + 1, N_E))
    spike_I = np.zeros((num_steps + 1, N_I))
    # spt_E = np.zeros((1, N_E))
    spt_E = np.zeros((num_steps + 1, N_E))

    v_E[0, :] = vE0
    v_I[0, :] = vI0
    S_EI[0, :] = S_EI0
    S_IE[0, :] = S_IE0
    X_EI[0, :] = X_EI0
    X_IE[0, :] = X_IE0
    Apost[0, :] = Apost0
    Apre[0, :] = Apre0
    W_IE[0, :] = W_IE0
    spt_E[0, :] = spt_E0

    # Stimulation Parameters.
    # number of E neurons stimulating
    stim_percent_E = np.zeros(N_E)
    a = 0
    b = int(np.floor(percent_V_stim * N_E))
    stim_percent_E[a:b] = 1  # TO MAKE FIGURES 3,4,7
    # stim_percent_E[a:b] = 1 + 0.1*np.random.randn(b)  # TO MAKE FIGURE 6

    # number of I neurons stimulating
    stim_percent_I = np.zeros(N_I)
    a = 0
    b = int(np.floor(percent_V_stim * N_I))
    stim_percent_I[a:b] = 1  # FIGURES 3,4,7
    # stim_percent_I[a:b] = 1 + 0.1*np.random.randn(b)  # FIGURE 6

    spike_I_time = np.zeros((num_steps + 1, N_I))
    delay_index = int(syn_delay / step_size) - 1

    for t in range(num_steps):
        time[t + 1, 0] = time[t, 0] + step_size

        # Excitatory-Inhibitory Network Model Updates.
        # Voltage (membrane) potentials.
        # (eq1&2): τm dV(t)/dt = -V(t) + Z(t) μ + (σ * sqrt(τm) * X(t)) + Vstim(t)
        # FIGURES 3,4,7
        dv_Edt = _dVdt(  # equation (1) in paper
            v=v_E[t, :],
            z=J_E / C_E * (  # equation (3) in paper
                S_EI[t - delay_index, :] if t > delay_index
                else leftover_S_EI[t, :]
            ),
            mu=mew_e,
            sigma=sigma_e,
            tau_m=tau_E_m[0, :],
            x=whitenoise_E[t, :],
            stim=ON * ue[0, t],
        )
        dv_Idt = _dVdt(  # equation (2) in paper
            v=v_I[t, :],
            z=J_I / C_I * (  # equation (3) in paper
                S_IE[t-delay_index, :] if t > delay_index
                else leftover_S_IE[t, :]
            ),
            mu=mew_i,
            sigma=sigma_i,
            tau_m=tau_I_m[0, :],
            x=whitenoise_I[t, :],
            stim=ON * ui[0, t],
        )
        # FIGURE 6
        # dv_Edt[i, :] = (vrest - v_E[i, :] + J_E/C_E * S_EI[i-delay_index, :] + mew_e + ON1 * ue[0, i] + ON1 * 10 * np.random.randn(1, N_E) + sigma_e * (tau_E_m[0, :]**0.5) * whitenoise_E[i, :]) / tau_E_m[0, :]
        # dv_Edt[i, :] = (vrest - v_E[i, :] + J_E/C_E * leftover_S_EI[i, :] + mew_e + ON1 * ue[0, i] + ON1 * 10 * np.random.randn(1, N_E) + sigma_e * (tau_E_m[0, :]**0.5) * whitenoise_E[i, :]) / tau_E_m[0, :]
        # dv_Idt[i, :] = (vrest - v_I[i, :] + J_I/C_I * S_IE[i-delay_index, :] + mew_i + ON1 * ui[0, i] + ON1 * 10 * np.random.randn(1, N_I) + sigma_i * (tau_I_m[0, :]**0.5) * whitenoise_I[i, :]) / tau_I_m[0, :]
        # dv_Idt[i, :] = (vrest - v_I[i, :] + J_I/C_I * leftover_S_IE[i, :] + mew_i + ON1 * ui[0, i] + ON1 * 10 * np.random.randn(1, N_I) + sigma_i * (tau_I_m[0, :]**0.5) * whitenoise_I[i, :]) / tau_I_m[0, :]
        v_E[t + 1, :] = v_E[t, :] + step_size * dv_Edt
        v_I[t + 1, :] = v_I[t, :] + step_size * dv_Idt

        # Conductance Updates.
        # (eq5): τd dS(t)/dt = -S(t) + X(t)
        dS_EIdt = (  # equation (5) in paper
            -S_EI[t, :] + X_EI[t, :]
        ) / tau_d
        dS_IEdt = (  # equation (5) in paper
            -S_IE[t, :] + X_IE[t, :]
        ) / tau_d
        S_EI[t + 1, :] = S_EI[t, :] + step_size * dS_EIdt
        S_IE[t + 1, :] = S_IE[t, :] + step_size * dS_IEdt

        # (eq6): τr dX(t)/dt = -X(t) + W(t) * δ(t - tpre + tdelay)
        dX_EIdt = (  # equation (6) in paper
            -X_EI[t, :]  # + W(t) * δ(t - tpre + tdelay) in spike check
        ) / tau_r
        dX_IEdt = (  # equation (6) in paper
            -X_IE[t, :]  # + W(t) * δ(t - tpre + tdelay) in spike check
        ) / tau_r
        X_EI[t + 1, :] = X_EI[t, :] + step_size * dX_EIdt
        X_IE[t + 1, :] = X_IE[t, :] + step_size * dX_IEdt

        # Spike-Timing Dependence Plasticity (STDP) Updates.
        # (eq7): W(t + Δt) = W(t) + ΔW(t)
        # (eq8): ΔW(t) = η * aLTP * Apost(t) if tpre - tpost < 0
        # (eq9): ΔW(t) = η * aLTD * Apre(t) if tpre - tpost > 0
        W_IE[t + 1, :] = (  # equation (7) in paper
            W_IE[t, :]  # + ΔW(t) in spike check
        )

        # (eq10): τLTP dApost/dt = -Apost + A0 * δ(t - tpost)
        dApostdt = (
            -Apost[t, :]  # + A0 * δ(t - tpost) in spike check
        ) / tau_LTD
        Apost[t + 1, :] = Apost[t, :] + step_size * dApostdt

        # (eq11): τLTD dApre/dt = -Apre + A0 * δ(t - tpre)
        dApredt = (
            -Apre[t, :]  # + A0 * δ(t - tpre) in spike check
        ) / tau_LTP
        Apre[t + 1, :] = Apre[t, :] + step_size * dApredt

        # Refractory Updates.
        ref_E[0, :] -= step_size
        ref_I[0, :] -= step_size

        # Calculate Kuramoto Order (but not really).
        spt_E[t + 1, :] = spt_E[t, :]
        # phif[i+1+int(comp_time/step), :] = 2*np.pi * (time[i+1, 0] + comp_time - spt_E[i, :])

        # Check for Spikes.
        for k in range(N_E):  # excitatory
            if v_E[t, k] < V_THRESHOLD <= v_E[t+1, k]:
                v_E[t + 1, k] = V_RESET
                ref_E[0, k] = REFRACTORY
                spike_E[t + 1, k] = k + 1
                spt_E[t + 1, k] = time[t + 1, 0] + comp_time
                # spt_E[0, k] = time[i+1, 0] + comp_time  # converter

                for j in range(N_I):  # E to I
                    if S_key_IE[k, j] != 0:
                        index = int(S_key_IE[k, j]) - 1
                        X_IE[t + 1, j] = (  # + W(t) * δ(t - tpre + tdelay)
                            X_IE[t, j] + W_IE[t, index]
                        )
                        # plasticity update - "on_pre"
                        Apre[t + 1, index] = (  # + A0 * δ(t - tpre)
                            Apre[t, index] + dApre_0
                        )
                        W_IE[t + 1, index] = (  # equation (9) in paper
                            W_IE[t, index]
                            + eta * a_LTD * Apost[t, index]
                        )
                        # max synaptic weight check
                        if (J_I * W_IE[t + 1, index]) < MAX_WEIGHT_IE:
                            W_IE[t + 1, index] = MAX_WEIGHT_IE / J_I
            elif ref_E[0, k] >= 0:  # in refractory period
                v_E[t + 1, k] = V_RESET
            elif v_E[t + 1, k] < V_REST:
                v_E[t + 1, k] = V_REST

        # Check for Action Potentials (inhibitory).
        for k in range(N_I):
            if v_I[t, k] < V_THRESHOLD <= v_I[t + 1, k]:
                v_I[t + 1, k] = V_RESET
                ref_I[0, k] = REFRACTORY
                spike_I[t + 1, k] = k + 1 + N_E
                spike_I_time[t + 1, k] = time[t + 1, 0]

                for j in range(N_E):  # I to E
                    if S_key_EI[k, j] != 0:
                        index = int(S_key_EI[k, j]) - 1
                        X_EI[t+1, j] = X_EI[t, j] - WEI
                    # plasticity update - "on_post"
                    if S_key_IE[j, k] != 0:
                        index = int(S_key_IE[j, k]) - 1
                        Apost[t + 1, index] = Apost[t, index] + dApost_0
                        W_IE[t + 1, index] = (  # equation (8) in paper
                            W_IE[t, index]
                            + eta * a_LTP * Apre[t, index]
                        )

                        # max synaptic weight check
                        if (J_I * W_IE[t + 1, index]) > MAX_WEIGHT_EI:
                            W_IE[t + 1, index] = MAX_WEIGHT_EI / J_I
            elif ref_I[0, k] >= 0:  # in refractory period
                v_I[t + 1, k] = V_RESET
            elif v_I[t + 1, k] < V_REST:
                v_I[t + 1, k] = V_REST

    # Calculate Synchrony.
    # todo: what synchrony measurement is this?
    # synchrony = sqrt(Var[V(t)] / Var[Vi(t)])
    N = N_E  # + N_I;
    Vcomb = np.zeros((num_steps + 1, N))
    Vcomb[:, 0:N_E] = v_E
    V1 = np.mean(Vcomb, axis=1)

    # variance of average voltage over whole run
    # sigma_v^2 = <(V(t)^2>t -[<V(t)>]^2
    sigma_squ_v = np.mean(V1**2) - (np.mean(V1))**2

    # variance of voltage at each time step
    sigma_vi = np.zeros(N)
    sum_sig = 0
    for j in range(N):
        sigma_vi[j] = np.mean(Vcomb[:, j]**2) - (np.mean(Vcomb[:, j]))**2
        sum_sig = sum_sig + sigma_vi[j]

    syn_squ = sigma_squ_v / (sum_sig / N)
    synchrony = np.sqrt(syn_squ)

    assert 0 <= synchrony <= 1

    return time, v_E, v_I, S_EI, S_IE, X_EI, X_IE, Apost, Apre, W_IE, spike_E, spike_I, ref_E, ref_I, synchrony, spt_E, phif
