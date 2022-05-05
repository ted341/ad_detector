import wave
import yaml
import numpy as np
from antropy import spectral_entropy, sample_entropy
from sklearn.cluster import KMeans
import audioop
import cv2


class FeatureBuilder:
    def __init__(self, shots, frames, input_audio):
        self.frames = frames
        self.input_audio = input_audio
        
        with open("config.yaml") as file:
            self.rate = yaml.safe_load(file)['audio']['frame_rate']
        
        self.shots = shots
        self.audio_segments = self._segment_audio()
        self.saturation_segments = []
        # self.brightness = []
        self.bri_std = []
        self._segment_video()
    
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
    
    def _segment_video(self):
        print('\tsegmenting video data...', end='', flush=True)
        for shot in self.shots:
            shot_sat_list = []
            brightness = []
            for i in range(shot.start_frame, shot.end_frame):
                frame = np.moveaxis(self.frames[i], 0, -1)
                hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                shot_sat_list.append(np.mean(hsv[:, :, 1]))
                brightness.append(np.mean(hsv[:, :, 2]))
            self.saturation_segments.append(np.mean(shot_sat_list))
            self.bri_std.append(np.std(brightness))
            # self.brightness.append(brightness)
                    
        print('done')
            
    
    def _test_play_audio(self, wf, data):
        import pyaudio
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),  # 2
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)
        stream.write(data[:48000*10])  # play for 3 seconds
        
        stream.stop_stream()
        stream.close()
        p.terminate()
            
    def build(self):
        for i, audio in enumerate(self.audio_segments):
            shot = self.shots[i]
            shot.features['entropy'] = self.build_spectral_entropy(audio)
            shot.features['duration'] = self.build_duration(shot)
            # shot.features['snr'] = self.build_snr(audio)
            # shot.features['edratio'] = shot.features['entropy'] / shot.features['duration']
            shot.features['sat'] = self.saturation_segments[i]
            shot.features['bri_std'] = self.bri_std[i]
            # shot.features['bright'] = sample_entropy(self.brightness[i])
    
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

            