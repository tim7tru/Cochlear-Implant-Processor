from scipy.io import wavfile
from scipy.signal import butter, lfilter, decimate
import matplotlib.pyplot as plt
import numpy as np
import simpleaudio as sa


def processButterworthBandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def applyButterworthBandPass(data, lowcut, highcut, fs, order=5):
    b, a = processButterworthBandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def processButterworthLowPass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def applyButterworthLowPass(data, cutoff, fs, order=5):
    b, a = processButterworthLowPass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def stereoToMono(audio):
    new_audio_data = [audio.sum(axis=1) / 2]
    return np.array(new_audio_data)


def playSoundFromFile(path):
    wave_obj = sa.WaveObject.from_wave_file(path)
    play_obj = wave_obj.play()
    play_obj.wait_done()


def playSoundFromData(data, rate, channels=1):
    sa.play_buffer(audio_data=data, num_channels=channels, bytes_per_sample=1, sample_rate=rate)
    # play_obj.wait_done()


def plotCosineFunction(rate):
    cosine_rate = 1000  # target 1kHz
    num_cosine_cycles = 2  # plot 2 cycles
    time_domain = num_cosine_cycles / cosine_rate  # Getting time domain for cosine
    cos_time_domain = np.linspace(0, time_domain, int(rate * time_domain))  # Getting time domain for cosine
    cos_signal_plot = np.cos(2 * np.pi * 1000 * cos_time_domain)  # Cosine function to plot
    plt.subplot(2, 1, 2)
    plt.stem(cos_time_domain, cos_signal_plot, 'b', markerfmt=' ')
    plt.ylabel("Cosine, cos(2000*pi*t)")
    plt.xlabel("Time, t (s)")


def downsampleData(data, downsample_factor):
    return decimate(x=data, q=int(downsample_factor), ftype="fir", axis=0)


def writeAudioFile(file_name, rate, data):
    wavfile.write(file_name, rate, data)


def combineAudioChannels(channels):
    return np.mean(channels, axis=0)


def plotAudioSignals(unprocessed_audio, final_audio):
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


def processAudioFile(file):
    """
    PLAYING AUDIO IS COMMENTED OUT, IF YOU WANT TO ENABLE THE PLAYING, UNCOMMENT THE COMMENT
    BELOW COMMENTS DENOTED WITH ***
    Turns stereo audio files into mono audio files with a sampling rate of 16KHz.
    Saves a mono version and a mono + down-sampled version of the clip.
    Plots the down-sampled, mono audio file as well as a cosine function with f=1kHz and sampling rate of 16kHz
    :param file: (String) Name of input audio file
    :return: No return
    """
    wav = file + '.wav'  # Name of file to read
    wav_mono = file + '_MONO.wav'  # Name of file to write
    wav_mono_downsampled = file + '_MONO_DS.wav'  # Name of final file
    target_rate = 16000  # target 16kHz

    """Read Input Signal"""
    sample_rate, data = wavfile.read(wav)
    # playSoundFromData(data=data, channels=2, rate=sample_rate)

    """Stereo to Mono"""
    mono = np.array(data)
    if data[0].size == 2:
        mono = stereoToMono(audio=data)
    writeAudioFile(file_name=wav_mono, rate=sample_rate, data=mono)
    # playSoundFromData(data=mono, rate=sample_rate)

    """Downsample Mono Audio"""
    signal = downsampleData(data=mono, downsample_factor=sample_rate/target_rate)
    writeAudioFile(file_name=wav_mono_downsampled, rate=target_rate, data=signal)
    # playSoundFromData(data=signal, rate=sample_rate)

    """Creating the channels of BPFs"""
    channel1 = applyButterworthBandPass(signal, 100, 7999, target_rate)
    channel2 = applyButterworthBandPass(signal, 600, 665, target_rate)
    channel3 = applyButterworthBandPass(signal, 700, 750, target_rate)
    channel4 = applyButterworthBandPass(signal, 800, 825, target_rate)
    all_channels = np.array([
        channel1,
        channel2,
        channel3,
        channel4,
    ])
    
    """Combine BPFs and apply to processed audio"""
    new_audio = combineAudioChannels(all_channels)
    writeAudioFile(file_name="final.wav", rate=target_rate, data=new_audio)
    # playSoundFromData(data=new_audio, rate=sample_rate)

    plotAudioSignals(unprocessed_audio=signal, final_audio=new_audio)


if __name__ == "__main__":
    processAudioFile(file='CS_N_Timmy')
