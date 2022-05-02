#!/usr/bin/env python

print('Loading packages...')

import click

from ad_detector.shotdetector import ShotDetector
from ad_detector.featurebuilder import FeatureBuilder
from ad_detector.shotclassifier import ShotClassifier

@click.command()
@click.argument('input_video', type=click.Path(exists=True))
@click.argument('input_audio', type=click.Path(exists=True))
@click.option('-d', '--dataset', 'dataset', type=int, default=None)
def main(input_video, input_audio, dataset):
    print('Start detecting...')
    shot_detector = ShotDetector(input_video)
    if dataset is None:
        shots = shot_detector.detect(save_json=True)
    else:
        shots = shot_detector.from_json(dataset)
    print('Detecting done')
    
    feature_builder = FeatureBuilder(shots, input_video, input_audio)
    feature_builder.build()
    
    shot_classifier = ShotClassifier(shots)
    shot_classifier.classify()
    shot_classifier.plot()
    
    

if __name__ == '__main__':
    main()