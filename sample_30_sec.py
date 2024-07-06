import soundfile as sf
from math import floor
import os

MUSIC_PATH="music/music/"
DEST_PATH="music/sampled/"

categories = os.listdir(MUSIC_PATH)

for category in categories:
    if not os.path.exists(DEST_PATH+category):
        os.makedirs(DEST_PATH+category)

    songs = os.listdir(MUSIC_PATH+category)

    for song in songs:
        track = sf.SoundFile(MUSIC_PATH + category + "/"+song)
        can_seek = track.seekable()
        if not can_seek:
            raise ValueError("Not compatible with seeking")

        sr = track.samplerate
        track_len= track.frames/sr

        start_frame = sr * floor(track_len/2)
        frames_to_read = sr * 30
        track.seek(start_frame)
        audio_section = track.read(frames_to_read)

        sf.write(f"{DEST_PATH}{category}/{song}", audio_section, sr)

