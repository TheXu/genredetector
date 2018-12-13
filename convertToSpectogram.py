from subprocess import Popen, PIPE, STDOUT
import os
from PIL import Image

#Define
currentPath = os.path.dirname(os.path.realpath(__file__))

def generateSpectogram():
    # Spectrogram resolution
    pixelPerSecond = 50
    basePath = "/Users/ananth/desktop/Audio"
    genres = ["Hip Hop"]
    for genre in genres:
        directory = "{}/{}/".format(basePath, genre)
        audioFiles = os.listdir(directory)
        audioFiles = [file for file in audioFiles]
        for file in audioFiles:
            if file.startswith('.'):
                continue
            command = "sox '{}' '/tmp/{}.mp3' remix 1,2".format(directory + file, file)
            p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, cwd=currentPath)
            output, errors = p.communicate()
            if errors:
                print()
            # Create spectrogram
            file.replace(".mp3", "")
            command = "sox '/tmp/{}.mp3' -n spectrogram -Y 200 -X {} -m -r -o '{}.png'".format(file, pixelPerSecond,
                                                                                               "{}spectograms/{}".format(directory, file))
            p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, cwd=currentPath)
            output, errors = p.communicate()
            if errors:
                print(errors)

            # Remove tmp mono track
            #os.remove("/tmp/{}.mp3".format(file))

def main():
    generateSpectogram()


if __name__ == "__main__":
    main()