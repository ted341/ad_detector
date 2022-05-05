import math

import numpy as np
from matplotlib import pyplot as plt
import yaml
from tqdm import trange, tqdm
import wave

np.set_printoptions(precision=3)


class ShotClassifier:
    def __init__(self, shots):
        self.shots = shots
        
        with open("config.yaml") as file:
            config = yaml.safe_load(file)
            self.audio_rate = config['audio']['frame_rate']
    
    def classify(self):
        # Pass 1: 1 seconds <= duration <= 10 seconds
        for shot in self.shots:
            if shot.duration >= 10:
                shot.is_ad = False
            elif shot.duration <= 0.8:
                shot.is_ad = True
        
        # Pass 2: edration > 2
        for shot in self.shots:
            if shot.is_ad is None:
                # shot.is_ad = shot.features['edratio'] > 2
                shot.is_ad = shot.features['entropy'] < 13
            
        # Pass 3: Relabel ads with duration < 8 seconds to non-ads
        scenes = []
        current_is_ad = self.shots[0].is_ad
        duration = self.shots[0].duration
        start_seq, end_seq = shot.sequence, shot.sequence
        for shot in self.shots[1:]:
            if current_is_ad == shot.is_ad:
                duration += shot.duration
                end_seq = shot.sequence
            else:
                scenes.append({'is_ad': current_is_ad,
                               'start_seq': start_seq,
                               'end_seq': end_seq,
                               'duration': duration})
                current_is_ad = shot.is_ad
                duration = shot.duration
                start_seq, end_seq = shot.sequence, shot.sequence
        scenes.append({'is_ad': current_is_ad,
                       'start_seq': start_seq,
                       'end_seq': end_seq,
                       'duration': duration})
        for scene in scenes:
            if scene['duration'] < 14:
                for i in range(scene['start_seq'], scene['end_seq']+1):
                    self.shots[i].is_ad = not scene['is_ad']
        
        # pass 4: find similar saturation shots
        for i in range(1, len(self.shots)):
            previous_shot = self.shots[i-1]
            current_shot = self.shots[i]
            if previous_shot.is_ad != current_shot.is_ad:
                if abs(previous_shot.features['sat'] - current_shot.features['sat']) <= 5:
                    if max(previous_shot.features['sat'], current_shot.features['sat']) <= 100:
                        previous_shot.is_ad = False
                        current_shot.is_ad = False
                    else:
                        previous_shot.is_ad = True
                        current_shot.is_ad = True
        
        print('\t' + ''.join(['A' if shot.is_ad else 'N' for shot in self.shots]))
        print('\t' + ''.join(['A' if shot.test_is_ad else 'N' for shot in self.shots]))    
    
    def plot(self):
        is_ad = []
        not_ad = []
        for shot in self.shots:
            if shot.test_is_ad:
                is_ad.append(shot)
            else:
                not_ad.append(shot)
        print('ad')
        for shot in is_ad:
            print(shot.sequence, shot.features)
        print('not')
        for shot in not_ad:
            print(shot.sequence, shot.features)
        
        # self._plot2D('sat', 'duration')
        self._plot1D('bri_std')
        
    
    def _plot2D(self, xlabel, ylabel):
        is_ad = []
        not_ad = []
        for i, shot in enumerate(self.shots):
            if shot.test_is_ad:
                is_ad.append((shot.features[xlabel], shot.features[ylabel]))
            else:
                not_ad.append((shot.features[xlabel], shot.features[ylabel]))
        
        if is_ad:
            plt.plot(*np.array(is_ad).T, 'ro')
        if not_ad:
            plt.plot(*np.array(not_ad).T, 'bo')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
    
    def _plot1D(self, label):
        is_ad = []
        not_ad = []
        for i, shot in enumerate(self.shots):
            if shot.test_is_ad:
                is_ad.append(shot.features[label])
            else:
                not_ad.append(shot.features[label])
        
        print('ad', np.array(is_ad))
        print('no', np.array(not_ad))
        
        if is_ad:
            plt.plot(np.array(is_ad), [0] * len(is_ad), 'ro')
        if not_ad:
            plt.plot(np.array(not_ad),[0] * len(not_ad), 'bo')
        plt.xlabel(label)
        plt.show()