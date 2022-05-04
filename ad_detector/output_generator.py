from dataclasses import dataclass
from pprint import pprint

import yaml
import numpy as np
from tqdm import tqdm, trange
import wave


@dataclass
class OutputGroup:
    source: int = 'original'
    video_start_frame: int = None
    video_end_frame: int = None
    audio_start_frame: int = None
    audio_sample_count: int = 0
    
    @property
    def is_empty(self):
        return self.video_start_frame is None


class OutputGenerator:
    def __init__(self, shots, input_audio, frames, output_video, output_audio):
        self.shots = shots
        self.input_audio = input_audio
        self.frames = frames
        self.output_video = output_video
        self.output_audio = output_audio
        self.wf = wave.open(input_audio, 'rb')
        self.output_groups = []
        
        with open("format.yaml") as file:
            format = yaml.safe_load(file)
            self.video_width = format['video_width']
            self.video_height = format['video_height']
            self.video_fps = format['video_fps']
            self.audio_rate = format['audio_rate']
    
    def replace_logo(self, logos):
        og = OutputGroup()
        detected_logo = None
        for i, shot in enumerate(self.shots):
            if not shot.is_ad:
                og.video_start_frame = shot.start_frame if og.video_start_frame is None else og.video_start_frame
                og.video_end_frame = shot.end_frame
                og.audio_start_frame = int(shot.start_timestamp * self.audio_rate) if og.audio_start_frame is None else og.audio_start_frame
                og.audio_sample_count += int(shot.duration * self.audio_rate)
                
                # check if logo is detected within the shot
                for logo in logos:
                    if shot.start_timestamp <= logo['time'] <= shot.end_timestamp:
                        detected_logo = logo['logo']
                        break
            # First ad after scenes
            elif not og.is_empty:
                self.output_groups.append(og)
                if detected_logo:
                    self.output_groups.append(OutputGroup(source=detected_logo))
                    detected_logo = None
                og = OutputGroup()
                
        if not og.is_empty:
            self.output_groups.append(og)
        print("Output list:")
        pprint(self.output_groups)
    
    def output(self):
        self._output_video()
        self._output_audio()
        self.wf.close()
    
    def _output_video(self):
        with open(self.output_video, 'wb') as video_file:
            for og in self.output_groups:
                if og.source == 'original':
                    for frame_no in trange(og.video_start_frame, og.video_end_frame):
                        frame_tmp = np.moveaxis(self.frames[frame_no], -1, 0)
                        frame_tmp.tofile(video_file)
                else:
                    print(f'\tSaving data from {og.source}_Ad_15s.rgb')
                    with open(f'./dataset2/Ads/{og.source}_Ad_15s.rgb', 'rb') as ad_rgb_file:
                        while data := ad_rgb_file.read(1024):
                            video_file.write(data)
        
    def _output_audio(self):
        with wave.open(self.output_audio, 'wb') as audio_file:
            audio_file.setnchannels(1)
            audio_file.setsampwidth(2) # 2 * 8 = 16 bits
            audio_file.setframerate(self.audio_rate)
            
            for og in self.output_groups:
                if og.source == 'original':
                    self.wf.setpos(og.audio_start_frame)
                    data = self.wf.readframes(og.audio_sample_count)
                    audio_file.writeframesraw(data)
                else:
                    print(f'\tSaving data from {og.source}_Ad_15s.wav')
                    with open(f'./dataset2/Ads/{og.source}_Ad_15s.wav', 'rb') as ad_audio_file:
                        while data := ad_audio_file.read(1024):
                            audio_file.writeframesraw(data)
        
        
        
            
            
        
        