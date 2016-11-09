import sys
import os
from pydub import AudioSegment

def file_type_from_name(name):
	return name.split(".")[-1]

class LowerQuality(object):
	def __init__(self, song_name):
		self.song_name = song_name
		self.file_type = file_type_from_name(song_name)
		self.song_path = os.getcwd() + "/" + song_name
	def lower_quality(self, target_bitrate="12k"):
		song = None
		if self.file_type == "wav":
			song = AudioSegment.from_wav(self.song_path)
		elif self.file_type == "mp3":
			song = AudioSegment.from_mp3(self.song_path)
		else:
			print "Error: unknown file type"
			return
		song.export(target_bitrate+"_"+self.song_name, format=self.file_type, bitrate=target_bitrate)

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print "************************************************"
		print "Make sure to install the following:\n\n\n"
		print "     $ brew install ffmpeg --with-libvorbis --with-ffplay --with-theora\n\n\n"
		print "Usage: python lower.py path/to/file.mp3 (optional:) 48k"
		print "************************************************"
		exit(0)
	
	target_bitrate = None
	song_name = sys.argv[1]
	
	if (len(sys.argv) > 2):
		target_bitrate = sys.argv[2]
	
	l = LowerQuality(song_name)
	print "Working..."
	l.lower_quality(target_bitrate)
	print "Done..."