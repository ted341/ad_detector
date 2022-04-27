import wave
import yaml
import numpy as np


class FeatureBuilder:
    def __init__(self, shots, input_video, input_audio):
        self.input_video = input_video
        self.input_audio = input_audio
        
        with open("format.yaml") as file:
            self.rate = yaml.safe_load(file)['audio_rate']
        
        self.shots = shots
        self.audio_segments = self._load_audio()
    
    def _load_audio(self):
        segments = []
        with wave.open(self.input_audio, 'rb') as wf:
            for shot in self.shots:
                start_position = int(shot.start_timestamp * self.rate)
                wf.setpos(start_position)
                
                total_audio_frames = int((shot.end_timestamp - shot.start_timestamp) * self.rate)
                raw_data = wf.readframes(total_audio_frames)
                segments.append(np.frombuffer(raw_data, dtype=np.int16))
            # self.__test_play_audio(wf, segments[6])
        return segments
    
    def __test_play_audio(self, wf, data):
        import pyaudio
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)
        stream.write(data[:48000*3])  # play for 3 seconds
        
        stream.stop_stream()
        stream.close()
        p.terminate()
            
    
    def build(self):
        pass