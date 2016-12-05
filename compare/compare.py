import sys
import os
import numpy as np
import audioop
from pydub import AudioSegment


def file_type_from_name(name):
    return name.split(".")[-1]


class CompareAudio(object):
    def __init__(self, original_song, comparison_song):
        self.original_song_name = original_song
        self.original_file_type = file_type_from_name(original_song)
        self.original_song_path = os.getcwd() + "/" + original_song

        self.comparison_song_name = comparison_song
        self.comparison_file_type = file_type_from_name(comparison_song)
        self.comparison_song_path = os.getcwd() + "/" + comparison_song

    def compare_quality(self, ):
        song1 = None
        if self.original_file_type == "wav":
            song1 = AudioSegment.from_wav(self.original_song_path)
        elif self.original_file_type == "mp3":
            song1 = AudioSegment.from_mp3(self.original_song_path)
        else:
            print("Error: first song - unknown file type")
            return

        song2 = None
        if self.original_file_type == "wav":
            song2 = AudioSegment.from_wav(self.comparison_song_path)
        elif self.original_file_type == "mp3":
            song2 = AudioSegment.from_mp3(self.comparison_song_path)
        else:
            print("Error: second song - unknown file type")
            return

        frame_length = 1 * 1000

        curr_frame = 0
        diff = 0

        while (curr_frame + frame_length < len(song1)):
            song1_frame = song1[curr_frame:curr_frame+frame_length]
            song2_frame = song2[curr_frame:curr_frame+frame_length]

            diff += abs(song1_frame.rms - song2_frame.rms)
            curr_frame+= frame_length

        print(diff)
        

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("************************************************")
        print("You must enter a path for the files you are comparing\n\n\n")
        print("Usage: python compare.py path/to/file1.mp3 path/to/file2.mp3")
        print("************************************************")
        exit(0)

    target_bitrate = None
    original_song = sys.argv[1]
    comparison_song = sys.argv[2]
    
    l = CompareAudio(original_song, comparison_song)
    print("Working...")
    l.compare_quality()
    print("Done...")