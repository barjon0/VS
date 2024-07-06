from ffmpeg import FFmpeg
import re
import os
import argparse

def convert_to_wav(source, dest):

    ffmpeg = FFmpeg()
    
    ffmpeg.input(source).output(
        dest,
        {"acodec": "pcm_s16le"},
    )

    ffmpeg.execute()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(prog='mp3-to-wav-Script')
    parser.add_argument('-s', '--source',required=True, help="Source folder path")  
    parser.add_argument('-d', '--destination',required=True, help="Destination folder path")
    args = parser.parse_args()

    SOURCE_FOLDER = args.source
    DEST_FOLDER = args.destination

    directory = os.fsencode(SOURCE_FOLDER)
    songs = os.listdir(directory)

    for song in songs:
        song = re.match(r"(.*).mp3",song.decode("utf-8"))[1]
        source = SOURCE_FOLDER+"/"+song+ ".mp3"
        dest = DEST_FOLDER+"/"+song+".wav"
        convert_to_wav(source, dest)