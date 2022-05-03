#!/usr/bin/env python

print('Loading packages...')

import click
import numpy as np
import yaml

from ad_detector.shotdetector import ShotDetector
from ad_detector.featurebuilder import FeatureBuilder
from ad_detector.shotclassifier import ShotClassifier
from ad_detector.output_generator import OutputGenerator

@click.command()
@click.argument('input_video', type=click.Path(exists=True))
@click.argument('input_audio', type=click.Path(exists=True))
@click.argument('output_video', type=click.Path())
@click.argument('output_audio', type=click.Path())
@click.option('-d', '--dataset', 'dataset', type=int, default=None)
def main(input_video, input_audio, output_video, output_audio, dataset):
    # Get video frames
    with open("format.yaml") as file:
        format = yaml.safe_load(file)
        height = format['video_height']
        width = format['video_width']
        fps = format['video_fps']
    raw_bytes = np.fromfile(input_video, np.dtype('B'))
    n_frames = len(raw_bytes)//(height*width*3)
    frames = raw_bytes.reshape((n_frames, 3, height, width)) # shape = (9000, 3, 270, 480)
    frames = np.moveaxis(frames, 1, -1)  # pack rgb values per pixel to shape (9000, 270, 480, 3)
    
    print('Start detecting...')
    shot_detector = ShotDetector(input_video, frames)
    if dataset is None:
        shots = shot_detector.detect(save_json=True)
    else:
        shots = shot_detector.from_json(dataset)
    print('Detecting done')
    
    feature_builder = FeatureBuilder(shots, input_video, input_audio)
    feature_builder.build()
    
    shot_classifier = ShotClassifier(shots)
    shot_classifier.classify()
    # shot_classifier.plot()
    
    output_generator = OutputGenerator(shots, input_audio, frames, output_video, output_audio)
    output_generator.replace_logo([{"time": 99, "logo": "mcd"}, {"time": 100, "logo": "mcd"}, {"time": 220, "logo": "nfl"}])
    output_generator.output()
    
    

if __name__ == '__main__':
    main()