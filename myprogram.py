#!/usr/bin/env python

print('Loading packages...')

import click
import yaml

from ad_detector.shotdetector import ShotDetector
from ad_detector.featurebuilder import FeatureBuilder
from ad_detector.logo_detector import LogoDetector
from ad_detector.shotclassifier import ShotClassifier
from ad_detector.output_generator import OutputGenerator

@click.command()
@click.argument('input_video', type=click.Path(exists=True))
@click.argument('input_audio', type=click.Path(exists=True))
@click.argument('output_video', type=click.Path())
@click.argument('output_audio', type=click.Path())
@click.option('-d', '--dataset', 'dataset', type=int, default=None)
def main(input_video, input_audio, output_video, output_audio, dataset):
    print('Start detecting...')
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    logo_detector = LogoDetector(input_video, output_video, config)
    # return all frames after processing
    frames = logo_detector.run()
    
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
    # shot_classifier.plot()
    
    output_generator = OutputGenerator(shots, input_audio, frames, output_video, output_audio)
    output_generator.replace_logo([{"time": 99, "logo": "mcd"}, {"time": 100, "logo": "mcd"}, {"time": 220, "logo": "nfl"}])
    output_generator.output()
    
    
if __name__ == '__main__':
    main()