import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import welch
import os
import soundfile as sf

# Function to compute the PSD of an audio file
def compute_psd(audio, sr, n_fft=1024, hop_length=512):
    # Compute the Short-Time Fourier Transform (STFT)
    stft = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))**2
    # Compute the Power Spectral Density (PSD)
    psd = np.mean(stft, axis=1)
    # Convert to frequency axis
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    return freqs, psd

# Function to load and process multiple audio files
def compute_average_psd(file_paths, n_fft=1024, hop_length=512):
    psds = []
    for file_path in file_paths:
        audio, sr = librosa.load(file_path, sr=None)
        freqs, psd = compute_psd(audio, sr, n_fft, hop_length)
        psds.append(psd)
    # Average PSD across all files
    average_psd = np.mean(psds, axis=0)
    return freqs, average_psd

# Function to apply Wiener filter
def apply_wiener_filter(audio, sr, target_psd, input_psd, n_fft=1024, hop_length=512):
    # Compute the STFT of the input audio
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    # Compute the PSD of the input audio
    #input_psd = np.mean(np.abs(stft)**2, axis=1)
    # Compute the Wiener filter gain function
    H = target_psd / (input_psd + 10e-15)
    # Apply the Wiener filter in the frequency domain
    filtered_stft = stft * H[:, np.newaxis]
    # Convert back to time domain
    filtered_audio = librosa.istft(filtered_stft, hop_length=hop_length)
    
    return filtered_audio

# List of audio file paths
input_dir_movies = 'movies_10061'
audio_files_movies = [os.path.join(input_dir_movies,audio) for audio in os.listdir(input_dir_movies)]
input_dir_celeb = 'movies_celeb_10061'
audio_files_celeb = [os.path.join(input_dir_celeb,audio) for audio in os.listdir(input_dir_celeb)]

# Create a directory for filtered audios if it doesn't exist
output_dir = input_dir_movies+"_filtered"
os.makedirs(output_dir, exist_ok=True)

# Compute the average PSD for each list of audio files
freqs1, noisy_psd = compute_average_psd(audio_files_movies)
freqs2, clean_psd = compute_average_psd(audio_files_celeb)

filtered_audios =[]
for i, file_path in enumerate(audio_files_movies):
    # Load the original audio
    audio, sr = librosa.load(file_path, sr=None)
    
    # Compute the PSD of the original audio
    freqs, psd = compute_psd(audio, sr)
    
    # Apply Wiener filter to the original audio
    filtered_audio = apply_wiener_filter(audio, sr, clean_psd, noisy_psd)
    output_path = os.path.join(output_dir, f'filtered_{os.path.basename(file_path)}')
    sf.write(output_path, filtered_audio, sr)

    filtered_audios.append(filtered_audio)

    # Compute the PSD of the filtered audio
    filtered_freqs, filtered_psd = compute_psd(filtered_audio, sr)
    
    # Plot the original and filtered spectrograms
    fig =plt.figure(figsize=(12, 8))
    
    # Original audio spectrogram
    plt.subplot(2, 1, 1)
    S = librosa.feature.melspectrogram(y=audio, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    filename = os.path.basename(audio_files_movies[i]).split('.')[0]
    plt.title(f'Original Spectrogram {filename}')
    plt.xlabel('Time')
    plt.ylabel('Frequency (Hz)')
    
    # Filtered audio spectrogram
    plt.subplot(2, 1, 2)
    S_filtered = librosa.feature.melspectrogram(y=filtered_audio, sr=sr)
    S_filtered_dB = librosa.power_to_db(S_filtered, ref=np.max)
    librosa.display.specshow(S_filtered_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Filtered Spectrogram {filename}')
    plt.xlabel('Time')
    plt.ylabel('Frequency (Hz)')
    
    plt.tight_layout()
    #plt.show()
    fig.savefig(f'{filename}.png')


# Compute the new average PSD for the filtered audios
filtered_psds = [compute_psd(audio, sr)[1] for audio in filtered_audios]
new_avg_psd1 = np.mean(filtered_psds, axis=0)

# Plot the PSDs
fig = plt.figure(figsize=(10, 6))
plt.semilogy(freqs1, noisy_psd, label=input_dir_movies)
plt.semilogy(freqs2, clean_psd, label=input_dir_celeb)
plt.semilogy(freqs1, new_avg_psd1, label='Filtered Audios')
plt.title('Average Power Spectral Density Comparison')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density')
plt.legend()
plt.grid()
plt.show()

fig.savefig('speaker_10061.png')

