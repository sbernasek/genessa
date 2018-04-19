__author__ = 'Sebi'

from modules.algorithms import Simulation
import numpy as np
import matplotlib.pyplot as plt

"""
TO DO: implement a pareto fitness function checking for sustained oscillations
"""


def get_frequency_modulation_capacity(system, ic='random', period=10, amplitude=1, offset=1, duration=50, dt=1,
                                      trials=1, method='bd-leaping', show_output=False):
    """
    Run simulation with specified sinusoidal input then determines the extent to which the system's output frequency differs.

    Parameters:
        system (network object) - system of interest
        ic (str) - define initial value, either 'ones', 'zeros', or 'random'
        period (float) - period of input signal
        amplitude (float) - amplitude of input signal
        offset (float) - mean offset of input signal
        duration (float) - simulation duration
        dt (float) - time step
        trials (int) - number of trials
        method (str) - solution algorithm
        show_output (bool) - if True, plot system output dynamics

    Returns:
        modulation_extent (float) - fraction of total signal power spectrum attributed to an alternate frequency
    """

    # mount system into simulation object
    sim = Simulation(system)

    # define input signal as sinusoid
    amplitude = 5
    offset = 10
    frequency = 1/period

    input_generator = lambda t: amplitude*np.sin(frequency*(2*np.pi)*t) + offset

    fitnesses, states_list = [], []
    for trial in range(trials):

        # define initial condition if none provided
        if ic == 'random':
            ic = np.random.randint(0, offset+amplitude, size=system.nodes.size)
        elif ic == 'ones':
            ic = np.ones(system.nodes.size)
        else:
            ic = np.zeros(system.nodes.size)

        # run simulation
        times, states = sim.simulate(ic=ic, input_function=input_generator, method=method, duration=duration, dt=dt)
        states_list.append(states)

        # get modulation extent via spectral power density
        fitness = get_modulation_extent(frequency, states[system.output_node, int(len(times)/4):], dt)

        # get binary similarity of signal to reference signal with same phase but twice the period (exclude first 25% of time series)
        #fitness = get_synchronization_with_target_frequency(times[int(len(times)/4):], states[system.output_node, int(len(times)/4):], target_frequency=1/period/2)
        fitnesses.append(fitness)

    if show_output is True:
        fig, ax = plt.subplots()
        ax.plot(times, np.apply_along_axis(input_generator, 0, times), '-b', label='input', linewidth=3)
        ax.plot(times, states_list[0][system.output_node, :], '-r', label='output', linewidth=3)
        for states in states_list[1:]:
            ax.plot(times, states[system.output_node, :], '-r', linewidth=3)
        ax.legend(loc=0)
        ax.tick_params(labelsize=14)

    return np.mean(fitnesses)


def get_modulation_extent(input_frequency, output_signal, dt):
    """
    Determine the extent to which an output signal has been modulated.

    Parameters:
        input_frequency (float) - frequency of input signal
        output_signal (array like) - discretized output
        dt (float) - time step for output signal

    Returns:
        modulation_extent (float) - fraction of total signal power spectrum attributed to an alternate frequency
    """

    # get ordered spectral power density
    ranked_frequencies, ranked_power = get_signal_frequencies(output_signal, dt)

    # compute total power
    total_power = np.sum(ranked_power[1:])

    # if output is unaffected, return score of zero
    if total_power < 10:
        modulation_extent = 0

    else:
        # determine ratio of highest alternate nonzero frequency power density to total power
        input_freq_index = np.argmin(abs(ranked_frequencies - input_frequency))
        for index, f in enumerate(ranked_frequencies):
            if f != 0 and index != input_freq_index:
                next_highest_power = ranked_power[index]
                break

        modulation_extent = next_highest_power/total_power

    return modulation_extent


def get_signal_frequencies(signal, dt, positive_only=True):
    """
    Determine frequencies within a signal by rank ordering spectral power density.

    Parameters:
        signal (array like) - signal magnitude at each timepoint
        dt (float) - signal discretization
    """

    # decompose signal via fft
    a_m = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(n=signal.size, d=dt)
    power_spectrum = np.abs(a_m)**2

    # get dominant fourier mode
    if positive_only is True:
        selected_indices = np.where(frequencies >= 0)
    else:
        selected_indices = np.arange(0, len(frequencies), step=1)

    ranked_indices = np.argsort(power_spectrum[selected_indices])[::-1]
    ranked_power = power_spectrum[selected_indices][ranked_indices]
    ranked_frequencies = frequencies[selected_indices][ranked_indices]

    return ranked_frequencies, ranked_power


def get_synchronization_with_target_frequency(times, signal, target_frequency=1/10):

    # convert signal to binary
    binary_signal = signal >= np.mean(signal)

    # get binary reference
    reference = np.sin(target_frequency*(2*np.pi)*times)
    binary_reference = reference >= 0

    # compute sum of squared residuals
    ssr = np.sum((binary_signal-binary_reference**2))

    return 1/ssr