from torchaudio import transforms
import torchaudio
import torch
import random


class AudioUtil():
    @staticmethod
    def open(audio_file):
        sig, sr = torchaudio.load(audio_file)
        return (sig,sr)
    
    # ----------------------------
    # Standardizing sample rate to 44100Hz
    # ----------------------------
    def resample(audio, srate):
        sig, sr = audio
        if (sr == srate):
            return audio
        no_channels = sig.shape[0]

        #Resample 1st channel:
        resig = torchaudio.transforms.Resample(sr, srate)(sig[:1,:])
        if (no_channels > 1):
            #Resample 2nd channel and merge both
            retwo = torchaudio.transforms.Resample(sr, srate)(sig[1:,:])
            resig = torch.cat([resig, retwo])

        return ((resig, srate))


    # ----------------------------
    # Some audios are mono, some are stereo. We need everything to have the same dimensions.
    # Thus, we can either only select the first channel of stereo or duplicate the first channel of mono
    # ----------------------------
    @staticmethod
    def rechannel(audio, channel):
        sig, sr = audio
        if (sig.shape[0]==channel):
            return audio
        if (channel==1):
            resig = sig[:1,:]
        else:
            resig = torch.cat([sig,sig])

        return ((resig, sr))

    

    # ----------------------------
    # Standardize the length of the audio - that is, either pad or truncate the audio
    # ----------------------------
    @staticmethod
    def resize_aud(audio, ms):
        sig, sr = audio
        no_rows, sig_len = sig.shape
        max_len = sr // 1000 * ms

        #Truncate
        if (sig_len > max_len):
            sig = sig[:, :max_len]
        #Padding
        elif (sig_len < max_len):
            #Length of the paddings at the start and end of the signal
            len_start = random.randint(0, max_len-sig_len)
            len_end = max_len - len_start - sig_len

            pad_start = torch.zeros((no_rows, len_start))
            pad_end = torch.zeros((no_rows, len_end))

            sig = torch.cat((pad_start, sig, pad_end), 1)

        return (sig, sr)


    # ----------------------------
    # Refer to textbox_1 for the reasoning of this method
    # ----------------------------
    @staticmethod
    def time_shift(aud, shift_limit):
        sig,sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return (sig.roll(shift_amt), sr)

    # ----------------------------
    # Generating Spectrogram
    # ----------------------------
    @staticmethod
    def spectro_gram(audio, n_mels=64, n_fft=1024, hop_len=None):
        sig, sr = audio
        top_db = 80 #if we have more time, we can try 80
        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)
        #shape of spec is [channel (mono or stereo etc), n_mels, time]
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return (spec)


    # ----------------------------
    # Augment the Spectrogram by masking out some sections of it in both the frequency
    # dimension (ie. horizontal bars) and the time dimension (vertical bars) to prevent
    # overfitting and to help the model generalise better. The masked sections are
    # replaced with the mean value.
    # ----------------------------
    @staticmethod
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

        return aug_spec
    

                