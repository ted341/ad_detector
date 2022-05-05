#!/usr/bin/env python

print('Loading packages...')

import click
import yaml
import time

from ad_detector.shot_detector import ShotDetector
from ad_detector.feature_builder import FeatureBuilder
from ad_detector.logo_detector import LogoDetector
from ad_detector.shot_classifier import ShotClassifier
from ad_detector.output_generator import OutputGenerator

@click.command()
@click.argument('input_video', type=click.Path(exists=True))
@click.argument('input_audio', type=click.Path(exists=True))
@click.argument('output_video', type=click.Path())
@click.argument('output_audio', type=click.Path())
@click.option('-d', '--dataset', 'dataset', type=int, default=None)
def main(input_video, input_audio, output_video, output_audio, dataset):
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    print('Logo detecting...')
    t1 = time.time()
    logo_detector = LogoDetector(input_video, output_video, config)
    frames = logo_detector.run()  # {'ae': [1362, 1951, 2124, 7470]}
    print(logo_detector.get_detected_framelist())
    print('done, time used:', time.time() - t1)
    
    # testing 
    # import numpy as np
    # raw_bytes = np.fromfile(input_video, np.dtype('B'))
    # n_frames = len(raw_bytes)//(270*480*3)
    # frames = raw_bytes.reshape((n_frames, 3, 270, 480)) # shape = (9000, 3, 270, 480)
    
    print('Shot detecting...')
    shot_detector = ShotDetector(input_video, frames)
    shots = shot_detector.detect(save_json=True)
    # shots = shot_detector.detect_from_json(dataset)
    
    print('Building features...')
    feature_builder = FeatureBuilder(shots, frames, input_audio)
    feature_builder.build()
    
    print('Classify shots...')
    shot_classifier = ShotClassifier(shots)
    shot_classifier.classify()
    # shot_classifier.plot()
    
    print('Generating output...')
    output_generator = OutputGenerator(shots, input_audio, frames, output_video, output_audio)
    
    output_generator.replace_logo(logo_detector.get_detected_framelist())
    output_generator.output()
    
    
if __name__ == '__main__':
    main()