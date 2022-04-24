#!/usr/bin/env python
import yaml
import click

from ad_detector.videoplayer import VideoPlayer

@click.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.argument('audio_path', type=click.Path(exists=True))
def main(video_path, audio_path):
    with open("format.yaml") as file:
        player_format = yaml.safe_load(file)
    
    player = VideoPlayer(player_format)
    player.load(video_path, audio_path)
    player.play()


if __name__ == '__main__':
    main()