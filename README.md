# Springboard_Capstone_2_Final

## Background: 
In this competition, we were challenged to use the Speech Commands Dataset to build an algorithm that understands
simple spoken commands.By improving the recognition accuracy of open-sourced voice interface tools,
we can improve product effectiveness and their accessibility.

The <b>training set </b> provided contains a few informational files and a folder of audio files. 
The audio folder contains subfolders with 1 second clips of voice commands, with the folder name being the label of the audio clip. 
There are more labels that should be predicted. The labels that we need to predict in Test are yes, no, up, down, left, right, on, off, 
stop, go. Everything else should be considered either unknown or silence. 
The folder _background_noise_ contains longer clips of "silence" that we can break up and use as training input.

The <b>test set </b> contains an audio folder with 150,000+ files in the format clip_000044442.wav. 
The task is to predict the correct label.

## Pre-processing: 
Exploratory data analysis showed that spectrogram representations of audio files could serve as better input to a deep learning network 
compared to raw audio inputs or fourier representations of the audio files. Further exploration showed that Mel power spectrograms 
, which better mimic the way the human ear perceives sound, could potentially be better than log power spectrograms. MFCC's which further abstract
away information from the mel spectrogram can also be potentially used. <br>
I generated log spectrograms, mel spectrograms and mfccs and stored them in the form of numpy arrays. I had previously tried to convert log spectrograms
into color images and store them to be used as input. But I found storing them as numpy arrays to be more memory efficient and it yielded 
the same results. <br>
So ultimately I had three numpy arrays 1) Log Spec 2) Mel Spec 3) MFCC corresponding to different representations of the input 

~~~
def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return np.log(spec.T.astype(np.float32) + eps)

def mel_specgram(samples, sample_rate):
    
    S = librosa.feature.melspectrogram(samples, sr=sample_rate, n_mels=128)
    # Convert to log scale (dB). We'll use the peak power (max) as reference.
    log_S = librosa.power_to_db(S, ref=np.max)
    return log_S

def mfcc(samples,sample_rate):
    S_spec = librosa.feature.melspectrogram(samples, sr=sample_rate, n_mels=128)
    log_S= librosa.power_to_db(S_spec, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
    return mfcc
 ~~~ 
![Image of Mel Spec](https://github.com/ravimaranganti/Springboard_Capstone_2_Final/blob/master/images/melspec_yes.png)
 <p align="center"> <b> Mel Spectrograms of Randomly Selected audiofiles labeled 'yes' </b> </p>
<br>
<br>
