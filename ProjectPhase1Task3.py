from scipy.io import wavfile
from scipy.signal import butter, filtfilt, decimate, sosfilt
import matplotlib.pyplot as plt
import numpy as np
import simpleaudio as sa
import os


def processButterworthBandpass(lowcut, highcut, fs, order=10):
    """
    Creates the second order sections of the Butterworth filter
    :param lowcut: (Int) The low cutoff frequency
    :param highcut: (Int) The high cutoff frequency
    :param fs: (Int) The sampling rate
    :param order: (Int) The order of the Butterworth filter
    :return: (Array) The second order sections of the Butterworth filter
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos


def applyButterworthBandPass(data, lowcut, highcut, fs, order=10):
    """
    Filters data through a Butterworth Bandpass Filter
    :param data: (Array) The audio data
    :param lowcut: (Int) The low cutoff frequency
    :param highcut: (Int) The high cutoff frequency
    :param fs: (Int) The sampling rate
    :param order: (Int) The order of the Butterworth filter
    :return: The filtered audio data
    """
    sos = processButterworthBandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y


def processButterworthLowPass(cutoff, fs, order=10):
    """
    Creates the polynomials for the Butterworth Lowpass Filter
    :param cutoff: (Int) The cutoff frequency
    :param fs: (Int) The sampling rate
    :param order: (Int) The order of the filter
    :return: (Array, Array) The polynomials for the Butterworth Lowpass Filter
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(N=order, Wn=normal_cutoff, btype='low')
    return b, a


def applyButterworthLowPass(data, cutoff, fs, order=10):
    """
    Filters data through a Butterworth Lowpass Filter
    :param data: (Array) The audio data
    :param cutoff: (Int) The cutoff frequncy
    :param fs: (Int) Sampling rate
    :param order: (Int) The order of the filter
    :return: The filtered audio data
    """
    b, a = processButterworthLowPass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def stereoToMono(audio):
    """
    Converts stereo audio data to mono audio data by averaging the two channels
    :param audio: (Array) The stereo audio data
    :return: The mono audio data
    """
    new_audio_data = [audio.sum(axis=1) / 2]
    return np.array(new_audio_data)


def playSoundFromFile(path):
    """
    Plays an audio file
    :param path: (String) The path to the audio file within the project
    :return: No Return
    """
    wave_obj = sa.WaveObject.from_wave_file(path)
    play_obj = wave_obj.play()
    play_obj.wait_done()


def playSoundFromData(data, rate, channels=1):
    """
    Plays the sound from data as an array
    :param data: (Array) The audio to be played
    :param rate: (Int) The sampling rate that the audio should be played at
    :param channels: (Int) 1 for mono, 2 for stereo
    :return: No Return
    """
    sa.play_buffer(audio_data=data, num_channels=channels, bytes_per_sample=1, sample_rate=rate)
    # play_obj.wait_done()


def plotCosineFunction(rate):
    """
    Plots a two cosine waves
    :param rate: (Int) The rate at which we want to plot to cosine wave
    :return: No Return
    """
    cosine_rate = 1000  # target 1kHz
    num_cosine_cycles = 2  # plot 2 cycles
    time_domain = num_cosine_cycles / cosine_rate  # Getting time domain for cosine
    cos_time_domain = np.linspace(0, time_domain, int(rate * time_domain))  # Getting time domain for cosine
    cos_signal_plot = np.cos(2 * np.pi * 1000 * cos_time_domain)  # Cosine function to plot
    plt.subplot(2, 1, 2)
    plt.stem(cos_time_domain, cos_signal_plot, 'b', markerfmt=' ')
    plt.ylabel("Cosine, cos(2000*pi*t)")
    plt.xlabel("Time, t (s)")


def getChannelCosineSignal(frequency, length):
    """

    :param frequency:
    :param length:
    :param sampling_rate:
    :return:
    """
    time_domain = length
    cos_time_domain = np.linspace(0, time_domain, time_domain)
    cos_signal = np.cos(2 * np.pi * frequency * cos_time_domain)
    return cos_signal


def downsampleData(data, downsample_factor):
    """
    Down samples data by a certain Q factor using a SciPy function, decimate
    :param data: (Array) The file to be downsampled
    :param downsample_factor: (Int) The factor in which the data is to be downsampled by
    :return: (Array) The downsampled audio
    """
    return decimate(x=data, q=int(downsample_factor), ftype="fir", axis=0)


def writeAudioFile(file_name, rate, data):
    """
    Writes an array out to an audio file
    :param file_name: (String) file name including the file type
    :param rate: (Int) rate in Hz to write the file in
    :param data: (Array) The data to write to an audio file
    :return: No Return
    """
    wavfile.write(file_name, rate, data)


def plotPhaseOne(unprocessed_audio, final_audio):
    """
    Used in Phase One to plot the unprocessed signal and the processed signal
    :param unprocessed_audio: (Array) The raw audio file
    :param final_audio: (Array) The mono-down-sampled audio file
    :return: No Return
    """
    sample_number = np.linspace(0, unprocessed_audio.size, unprocessed_audio.size)  # Getting sample number domain

    plt.subplot(2, 1, 1)
    plt.stem(sample_number, unprocessed_audio, 'b', markerfmt=' ')
    plt.ylabel("Input audio data, x[n]")
    plt.xlabel("Sample number, n")

    plt.subplot(2, 1, 2)
    plt.stem(sample_number, final_audio, 'b', markerfmt=' ')
    plt.ylabel("Filtered audio data, x[n]")
    plt.xlabel("Sample number, n")

    # ***Play Cosine Signal
    # plotCosineFunction(rate=sample_rate)
    # playSoundFromData(data=np.cos(2 * np.pi * 1000 * sample_number), rate=target_rate)

    plt.show()


def plotSingularAudioSignal(audio, x_label, y_label):
    """
    Plots an audio signal and labels the inputted axes
    :param audio: (Array) Signal to be plotted
    :param x_label: (String) X axis label
    :param y_label: (String) Y axis label
    :return: No Return
    """
    sample_number = np.linspace(0, audio.size, audio.size)
    plt.stem(sample_number, audio, 'b', markerfmt=' ')
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.show()


def applyEnvelopeDetection(input_signal, cutoff_frequency, current_rate):
    """
    Takes an signal and performs envelope detection using the following algorithm
    1. Rectifying the input signal by taking the absolute value of each sample
    1. Finding the number of samples within the cutoff frequency inside the current frequency
    2. Finding the max value inside the sub-range of samples from the rectified signal
    3. For each max value found, we assign that value to the sub-range in the resulting signal
    4. Then we pass the the resulting signal to a low-pass filter to smooth out the envelope signal
    :param input_signal: (Array) Input signal
    :param cutoff_frequency: (Int) Cutoff frequency for the lowpass signal
    :param current_rate: (Int) The sampling rate of the rectified signal
    :return: The enveloped signal
    """
    # Full wave rectification
    rectified_signal = np.absolute(input_signal)

    # Peak detection
    interval_length = current_rate / cutoff_frequency
    resulting_signal = np.zeros(rectified_signal.size)
    low = 0
    high = interval_length
    while high <= resulting_signal.size:
        max_num = rectified_signal[int(low):int(high)].max()
        resulting_signal[range(int(low), int(high))] = max_num
        low = high
        high += interval_length

    # Applying Lowpass Filter
    lowpass = applyButterworthLowPass(data=resulting_signal, cutoff=2*cutoff_frequency, fs=current_rate, order=2)
    return lowpass


def plotPhaseTwo(y_label, filtered_signal, envelope):
    """
    Plots the first and last channel
    :param y_label: (String) The y-axis label
    :param filtered_signal: (Array) The signal to be plotted
    :param envelope: (Array) The envelope of the signal to be plotted
    :return: No Return
    """
    sample_number = np.linspace(0, filtered_signal.size, filtered_signal.size)
    plt.stem(sample_number, filtered_signal, 'b', markerfmt=' ', label="Filtered Signal")
    plt.plot(sample_number, envelope, 'r', label="Upper Envelope")
    plt.xlabel("Sample Number, n")
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


def getCentralFrequency(channel_range):
    """
    Gets the central frequency of a frequency band
    :param channel_range: (Array) The frequency range of the channel
    :return: (Int) The central frequency or zero if invalid input
    """
    if len(channel_range) == 0:
        return 0
    elif len(channel_range) == 1:
        return channel_range[0]
    elif len(channel_range) == 2:
        return int((channel_range[1] + channel_range[0]) / 2)
    else:
        return 0


def modulateSignal(carrier, modulator):
    """
    Amplitude Modulates (AM) the signal
    :param carrier: (Array) The carrier signal, cosine in our case
    :param modulator: (Array) The modulator signal, the enveloped signal in out case
    :return: (Array) Returns the modulated signal
    """
    return_arr = np.zeros(len(carrier))
    for i in range(len(carrier)):
        amp_cos = carrier[i]
        amp_mod = modulator[i]
        return_arr[i] = amp_cos * amp_mod
    return return_arr


def combineAllChannels(modulated_channels):
    """
    Combines all signals in each channel by summing all signals
    :param modulated_channels: (Array) N*M array of the list of all modulated channels
    :return: (Array) 1D array of the output signal
    """
    combined = np.zeros(len(modulated_channels[0]))
    for i in range(len(combined)):
        sum_of_channels = 0
        for j in range(len(modulated_channels)):
            sum_of_channels += modulated_channels[j][i]
        combined[i] = sum_of_channels

    return combined


def normaliseSignal(signal):
    """
    Normalise the signal amplitude by dividing the signal by the abs value such that the max amplitude is 1
    :param signal: (Array) The modulated output signal
    :return: (Array) The normalised signal
    """
    maximum = max(signal)
    minimum = min(signal)
    abs_max = max(maximum, abs(minimum))
    signal /= abs_max
    return signal


def processAudioFile(file, channel_ranges, envelope_detection_freq):
    """
    PLAYING AUDIO IS COMMENTED OUT, IF YOU WANT TO ENABLE THE PLAYING, UNCOMMENT THE COMMENT
    BELOW COMMENTS DENOTED WITH ***
    Turns stereo audio files into mono audio files with a sampling rate of 16KHz.
    Saves a mono version and a mono + down-sampled version of the clip.
    Plots the down-sampled, mono audio file as well as a cosine function with f=1kHz and sampling rate of 16kHz
    :param channel_ranges: (Array) The array of channel ranges / frequency bands
    :param envelope_detection_freq: (Int) The envelope detection frequency in the filtering process
    :param file: (String) Name of input audio file
    :return: No return
    """
    enveloped = 'channels/enveloped/'
    filtered = 'channels/filtered/'
    modulated = 'channels/modulated/'
    audio = 'audio/'
    channel = "channel"
    filt = "_filt.wav"
    env = "_env.wav"
    mod = "_mod.wav"

    wav = audio + 'raw/' + file + '.wav'  # Name of file to read
    wav_mono = audio + 'mono/' + file + '_MONO.wav'  # Name of file to write
    wav_mono_downsampled = audio + 'mono_ds/' + file + '_MONO_DS.wav'  # Name of final file
    output = audio + 'output/' + file + '_final.wav'
    output_norm = audio + 'output_normalised/' + file + '_final_norm.wav'

    signal_range = np.array([100, 7999])
    target_rate = 16000  # target 16kHz
    num_of_channels = len(channel_ranges)
    channels_filt_file_name = np.full(num_of_channels, "", dtype=object)
    channels_env_file_name = np.full(num_of_channels, "", dtype=object)
    channels_mod_file_name = np.full(num_of_channels, "", dtype=object)

    for i in range(num_of_channels):
        channels_filt_file_name[i] = (str(channel + str(i + 1) + filt))
        channels_env_file_name[i] = (str(channel + str(i + 1) + env))
        channels_mod_file_name[i] = (str(channel + str(i + 1) + mod))

    """Read Input Signal"""
    sample_rate, data = wavfile.read(wav)
    # playSoundFromData(data=data, channels=2, rate=sample_rate)

    """Stereo to Mono"""
    mono = np.array(data)
    if data[0].size == 2:
        mono = stereoToMono(audio=data)[0]
    writeAudioFile(file_name=wav_mono, rate=sample_rate, data=mono)
    # playSoundFromData(data=mono, rate=sample_rate)

    """Downsample Mono Audio"""
    signal = mono
    if sample_rate > target_rate:
        signal = downsampleData(data=mono, downsample_factor=sample_rate / target_rate)
    writeAudioFile(file_name=wav_mono_downsampled, rate=target_rate, data=signal)

    """Creating the channels of BPFs"""
    signal_range = applyButterworthBandPass(signal, signal_range[0], signal_range[1], target_rate)
    writeAudioFile(file_name="signal.wav", rate=target_rate, data=signal_range)

    channels_filtered = np.full((num_of_channels, len(signal_range)), 0, dtype=float)
    channels_enveloped = np.full((num_of_channels, len(signal_range)), 0, dtype=float)
    channels_cosine = np.full((num_of_channels, len(signal_range)), 0, dtype=float)
    channels_modulated = np.full((num_of_channels, len(signal_range)), 0, dtype=float)

    for i in range(num_of_channels):
        channels_filtered[i] = applyButterworthBandPass(
            data=signal_range,
            lowcut=channel_ranges[i][0],
            highcut=channel_ranges[i][1],
            fs=target_rate
        )
        channels_enveloped[i] = applyEnvelopeDetection(
            input_signal=channels_filtered[i],
            cutoff_frequency=envelope_detection_freq,
            current_rate=target_rate
        )
        channels_cosine[i] = getChannelCosineSignal(
            frequency=getCentralFrequency(channel_range=channel_ranges[i]),
            length=len(signal_range)
        )
        channels_modulated[i] = modulateSignal(
            carrier=channels_cosine[i],
            modulator=channels_enveloped[i]
        )
        writeAudioFile(file_name=filtered + channels_filt_file_name[i], rate=target_rate, data=channels_filtered[i])
        writeAudioFile(file_name=enveloped + channels_env_file_name[i], rate=target_rate, data=channels_enveloped[i])
        writeAudioFile(file_name=modulated + channels_mod_file_name[i], rate=target_rate, data=channels_modulated[i])
        if i == 0:
            plotPhaseTwo(y_label="Channel 1 Signal, 264-314Hz, x[n]", filtered_signal=channels_filtered[i],
                         envelope=channels_enveloped[i])
        elif i == num_of_channels - 1:
            plotPhaseTwo(
                y_label="Channel " + str(num_of_channels) + " Signal, " + str(
                    channel_ranges[num_of_channels - 1][0]) + "-" +
                        str(channel_ranges[num_of_channels - 1][1]) + "Hz , x[n]", filtered_signal=channels_filtered[i],
                envelope=channels_enveloped[i])

    output_signal = combineAllChannels(modulated_channels=channels_modulated)
    normalized_output = normaliseSignal(signal=output_signal)
    writeAudioFile(file_name=output, rate=target_rate, data=output_signal)
    writeAudioFile(file_name=output_norm, rate=target_rate, data=normalized_output)


"""
To process audio files,
(1) place all audio files into the ./audio/raw directory of the project
(2) run the code and follow the console instructions, it will ask you (in this order):
    (a) Number of channels
    (b) For each channel, what the low and high frequencies are
    (c) The envelope detection frequency
    (d) If you want all the audio files, or just one to be processed
(3) Wait for the magic to happen
(4) To play an audio file, go to ./audio/output_normalised for the final processed audio file.
"""
if __name__ == "__main__":
    num_of_channels = int(input('How many audio channels?'))
    audio_files = np.array(os.listdir('./audio/raw'))
    channels = np.zeros((num_of_channels, 2))

    for i in range(num_of_channels):
        low = 0
        high = 8000
        while low < high and (low < 100 and high > 7999):
            low = int(input('For channel ' + str(i + 1) + ', what is the low frequency?'))
            high = int(input('For channel ' + str(i + 1) + ', what is the high frequency?'))
        channels[i][0] = low
        channels[i][1] = high

    env_det_freq = int(input('What should the envelope detection frequency be?'))

    run_all = input('Do you want to run all the files or just one? (y for all, n for one)')
    if run_all.lower() == 'y':
        for i in range(len(audio_files)):
            if audio_files[i] != '.DS_Store':
                audio = audio_files[i][:-4]
                processAudioFile(file=audio, channel_ranges=channels,
                                 envelope_detection_freq=env_det_freq)
    else:
        new = "MS_N_Timmy"
        processAudioFile(file=new, channel_ranges=channels,
                         envelope_detection_freq=env_det_freq)
