#!/usr/bin/env python
import click

from ad_detector.shotdetector import ShotDetector
from ad_detector.featurebuilder import FeatureBuilder

@click.command()
@click.argument('input_video', type=click.Path(exists=True))
@click.argument('input_audio', type=click.Path(exists=True))
def main(input_video, input_audio):
    shot_detector = ShotDetector(input_video, input_audio)
    # shots = shot_detector.detect()
    shots = shot_detector.debug()
    
    feature_builder = FeatureBuilder(shots, input_video, input_audio)
    feature_builder.build()

if __name__ == '__main__':
    main()