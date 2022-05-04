#!/usr/bin/env python
import yaml
import click

from ad_detector.video_player import VideoPlayer

@click.command()
@click.argument('input_video', type=click.Path(exists=True))
@click.argument('input_audio', type=click.Path(exists=True))
def main(input_video, input_audio):
    with open("config.yaml") as file:
        config = yaml.safe_load(file)
    
    player = VideoPlayer(input_video, input_audio, config)
    player.play()


if __name__ == '__main__':
    main()