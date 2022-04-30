#!/usr/bin/env python
import click
import yaml

from ad_detector.shotdetector import ShotDetector
from ad_detector.featurebuilder import FeatureBuilder
from ad_detector.logo_detector import LogoDetector, VideoFormat

@click.command()
@click.argument('input_video', type=click.Path(exists=True))
@click.argument('input_audio', type=click.Path(exists=True))
def main(input_video, input_audio):

    with open("format.yaml") as file:
        config = yaml.safe_load(file)

    input_logo = "dataset/Brand Images/starbucks_logo.bmp"
    video_format = VideoFormat(config["video_height"], config["video_width"])
    logo_detector = LogoDetector(input_video, input_logo, video_format)
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