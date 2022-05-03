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
        
        with open("format.yaml") as file:
            format = yaml.safe_load(file)
            self.audio_rate = format['audio_rate']
    
    def classify(self):
        # Pass 1: 1 seconds <= duration <= 10 seconds
        for shot in self.shots:
            if shot.duration >= 10:
                shot.is_ad = False
            elif shot.duration <= 1:
                shot.is_ad = True
        
        # Pass 2: edration > 2
        for shot in self.shots:
            if shot.is_ad is None:
                shot.is_ad = shot.features['edratio'] > 2
        
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
            if scene['duration'] < 8:
                for i in range(scene['start_seq'], scene['end_seq']+1):
                    self.shots[i].is_ad = not scene['is_ad']
        
        print(''.join(['A' if shot.is_ad else 'N' for shot in self.shots]))
        print(''.join(['A' if shot.test_is_ad else 'N' for shot in self.shots]))    
    
    def plot(self):
        self._plot2D('edratio', 'duration')
        # self._plot1D('snr')
    
    def _plot2D(self, xlabel, ylabel):
        is_ad = []
        not_ad = []
        for i, shot in enumerate(self.shots):
            # only print out non-classified shots
            # if shot.is_ad is not None:
            #     if shot.is_ad != shot.test_is_ad:
            #         print('WRONG!!!', i, is_ad, shot)
            #     continue
            
            if shot.test_is_ad:
                is_ad.append((shot.features[xlabel], shot.features[ylabel]))
            else:
                not_ad.append((shot.features[xlabel], shot.features[ylabel]))
        
        print('ad', np.array(is_ad))
        print('no', np.array(not_ad))
        
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
            if shot.is_ad:
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