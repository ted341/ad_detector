#!/usr/bin/env python
import yaml
import click

from ad_detector.video_player import VideoPlayer

@click.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.argument('audio_path', type=click.Path(exists=True))
def main(video_path, audio_path):
    with open("config.yaml") as file:
        config = yaml.safe_load(file)
    
    player = VideoPlayer(config['video']['width'],
                         config['video']['height'],
                         config['video']['frame_rate'],
                         config['audio']['rate'])
    player.load(video_path, audio_path)
    player.play()


if __name__ == '__main__':
    main()