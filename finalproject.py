from subprocess import Popen, PIPE, STDOUT
import os
import numpy as np
import keras
from PIL import Image
from random import shuffle
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.losses import categorical_crossentropy
import pickle

#Define
currentPath = os.path.dirname(os.path.realpath(__file__))
genres = ["Rap", "EDM", "Jazz", "Rock", "Classical"]
sliceSize = 128 #size of each slice of spectrogram
basePath = "/Users/ananth/desktop/Audio"
validationRatio = 0.2
testRatio = 0.1

def getProcessedImage(filename, size):
    image = Image.open(filename)
    image = image.resize((size,size), resample=Image.ANTIALIAS)
    imageData = np.asarray(image, dtype=np.uint8).reshape(size,size,1)
    imageData = imageData/255.
    return imageData

class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


def createModel():
    model = Sequential()
    input_shape = (sliceSize,sliceSize,1) #change this based on slice size
    model.add(Conv2D(32, kernel_size=(2, 2), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy'])


def trainModel(model):
    x_train, y_train, x_validate, y_validate, x_test, y_test = loadDataset()
    history = AccuracyHistory()
    model.fit(x_train, y_train,
              batch_size=128,
              epochs=5,
              verbose=1,
              validation_data=(x_validate, y_validate),
              callbacks=[history])
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

def saveModel(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("model_relu.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_relu.h5")
    print("Saved model to file")

def loadDataset():
    datasetName = "Alldata"
    datasetPath = "/Users/ananth/desktop/Audio/Data/"
    print("Loading Training, Validation, and Test Data Sets")
    train_X = try_to_load_as_pickled_object_or_None("{}train_X_{}.p".format(datasetPath, datasetName))
    train_y = try_to_load_as_pickled_object_or_None("{}train_y_{}.p".format(datasetPath, datasetName))
    validation_X = try_to_load_as_pickled_object_or_None("{}validation_X_{}.p".format(datasetPath, datasetName))
    validation_y = try_to_load_as_pickled_object_or_None("{}validation_y_{}.p".format(datasetPath, datasetName))
    test_X = try_to_load_as_pickled_object_or_None("{}test_X_{}.p".format(datasetPath, datasetName))
    test_y = try_to_load_as_pickled_object_or_None("{}test_y_{}.p".format(datasetPath, datasetName))
    print("All Datasets Loaded!")
    return train_X, train_y, validation_X, validation_y, test_X, test_y

def saveDataset(train_X, train_y, validation_X, validation_y, test_X, test_y):
    datasetPath = "/Users/ananth/desktop/Audio/Data/"
     #Create path for dataset if not existing
    if not os.path.exists(os.path.dirname(datasetPath)):
        try:
            os.makedirs(os.path.dirname(datasetPath))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    print("Saving Dataset To File...")
    datasetName = "Alldata"
    save_as_pickled_object(train_X, "{}train_X_{}.p".format(datasetPath,datasetName))
    save_as_pickled_object(train_y, "{}train_y_{}.p".format(datasetPath, datasetName))
    save_as_pickled_object(validation_X, "{}validation_X_{}.p".format(datasetPath, datasetName))
    save_as_pickled_object(validation_y, "{}validation_y_{}.p".format(datasetPath, datasetName))
    save_as_pickled_object(test_X, "{}test_X_{}.p".format(datasetPath, datasetName))
    save_as_pickled_object(test_y, "{}test_y_{}.p".format(datasetPath, datasetName))
    print("Dataset Saved!")

def prepareDataFromSlices():
    data = []
    for genre in genres:
        print(genre)
        slicesPath = "{}/{}/slices/".format(basePath, genre)
        sliceFiles = os.listdir(slicesPath)
        files = [file for file in sliceFiles]
        shuffle(files)
        for file in files:
            if file.startswith('.DS'):
                continue
            imageData = getProcessedImage(slicesPath+file, sliceSize)
            label = [1. if genre == g else 0. for g in genres]
            data.append((imageData, label))
    shuffle(data)
    X, y = zip(*data)

    #Split data
    validationNum = int(len(X)*validationRatio)
    testNum = int(len(X)*testRatio)
    trainNum = len(X)-(validationNum + testNum)
    # Prepare for Tflearn at the same time
    train_X = np.array(X[:trainNum]).reshape([-1, sliceSize, sliceSize, 1])
    train_y = np.array(y[:trainNum])
    validation_X = np.array(X[trainNum:trainNum + validationNum]).reshape([-1, sliceSize, sliceSize, 1])
    validation_y = np.array(y[trainNum:trainNum + validationNum])
    test_X = np.array(X[-testNum:]).reshape([-1, sliceSize, sliceSize, 1])
    test_y = np.array(y[-testNum:])
    print("Data sets created")
    # Save
    saveDataset(train_X, train_y, validation_X, validation_y, test_X, test_y)

    return train_X, train_y, validation_X, validation_y, test_X, test_y

class MacOSFile(object):

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size), end="", flush=True)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            print("writing bytes [%s, %s)... " % (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            print("done.", flush=True)
            idx += batch_size


def save_as_pickled_object(obj, file_path):
    with open(file_path, "wb") as f:
        return pickle.dump(obj, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)


def try_to_load_as_pickled_object_or_None(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(MacOSFile(f))


def divideSpectrogram():
    for genre in genres:
        directory = "{}/{}/spectrograms/".format(basePath, genre)
        slicesPath = "{}/{}/slices/".format(basePath, genre)
        imageFiles = os.listdir(directory)
        imageFiles = [file for file in imageFiles]
        for file in imageFiles:
            if file.startswith('.DS'):
                continue
            image = Image.open(directory + file)
            width, height = image.size
            numPiecesInFile = int(width/sliceSize)
            for i in range(numPiecesInFile):
                start = i * sliceSize
                tempImage = image.crop((start, 1, start + sliceSize, sliceSize + 1))
                tempImage.save(slicesPath + "{}_{}.png".format(file[:-4], i))


def generateSpectrogram():
    # Spectrogram resolution
    pixelPerSecond = 50
    for genre in genres:
        directory = "{}/{}/".format(basePath, genre)
        audioFiles = os.listdir(directory)
        audioFiles = [file for file in audioFiles]
        for file in audioFiles:
            if file.startswith('.') or file.startswith('spectrograms'):
                continue
            #creates temporary mono file to convert to spectrogram
            command = "sox '{}' '/tmp/{}.mp3' remix 1,2".format(directory + file, file)
            #print(command)
            p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, cwd=currentPath)
            output, errors = p.communicate()
            if errors:
                print(errors)
            # Create spectrogram
            file.replace(".mp3", "")
            command = "sox '/tmp/{}.mp3' -n spectrogram -Y 200 -X {} -m -r -o '{}.png'".format(file, pixelPerSecond,
                                                                                               "{}spectrograms/{}".format(directory, file))
            p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, cwd=currentPath)
            output, errors = p.communicate()
            if errors:
                print(errors)

            #Remove tmp mono track
            os.remove("/tmp/{}.mp3".format(file))

def main():
    createModel()


if __name__ == "__main__":
    main()