#!/usr/bin/env python
import click
import yaml

from ad_detector.shotdetector import ShotDetector
from ad_detector.featurebuilder import FeatureBuilder
from ad_detector.logo_detector import LogoDetector

@click.command()
@click.argument('input_video', type=click.Path(exists=True))
@click.argument('input_audio', type=click.Path(exists=True))
@click.argument('output_video', type=click.Path())
@click.argument('output_audio', type=click.Path())
def main(input_video, input_audio, output_video, output_audio):

    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    logo_detector = LogoDetector(input_video, output_video, config)
    logo_detector.run()

    '''
    shot_detector = ShotDetector(input_video, input_audio)
    # shots = shot_detector.detect()
    shots = shot_detector.debug()
    
    feature_builder = FeatureBuilder(shots, input_video, input_audio)
    feature_builder.build()'
    '''

if __name__ == '__main__':
    main()