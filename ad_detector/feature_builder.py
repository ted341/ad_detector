import wave
import yaml
import numpy as np
from antropy import spectral_entropy
from sklearn.cluster import KMeans
import audioop


class FeatureBuilder:
    def __init__(self, shots, input_video, input_audio):
        self.input_video = input_video
        self.input_audio = input_audio
        
        with open("config.yaml") as file:
            self.rate = yaml.safe_load(file)['audio']['rate']
        
        self.shots = shots
        self.audio_segments = self._segment_audio()
    
    def _segment_audio(self):
        print('\tsegmenting audio data...', end='')
        segments = []
        with wave.open(self.input_audio, 'rb') as wf:
            for shot in self.shots:
                start_position = int(shot.start_timestamp * self.rate)
                wf.setpos(start_position)
                
                total_audio_frames = int((shot.duration) * self.rate)
                raw_data = wf.readframes(total_audio_frames)
                segments.append(np.frombuffer(raw_data, dtype=np.int16))
            # self._test_play_audio(wf, segments[12])
        print('done')
        return segments
    
    def _test_play_audio(self, wf, data):
        import pyaudio
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),  # 2
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)
        stream.write(data[:48000*3])  # play for 3 seconds
        
        stream.stop_stream()
        stream.close()
        p.terminate()
            
    def build(self):
        for i, audio in enumerate(self.audio_segments):
            shot = self.shots[i]
            shot.features['entropy'] = self.build_spectral_entropy(audio)
            shot.features['duration'] = self.build_duration(shot)
            shot.features['snr'] = self.build_snr(audio)
            shot.features['edratio'] = shot.features['entropy'] / shot.features['duration']
    
    def build_spectral_entropy(self, audio):
        return spectral_entropy(audio, self.rate)
    
    def build_rms(self, audio):
        return audioop.rms(audio, 2)  # 2: 16bits per sample
    
    def build_duration(self, shot):
        return shot.duration
    
    def build_snr(self, audio, axis=0, ddof=0):
        """signal-to-noise. Not a good feature"""
        a = np.asanyarray(audio)
        m = a.mean(axis)
        sd = a.std(axis=axis, ddof=ddof)
        return np.where(sd == 0, 0, m/sd)

            