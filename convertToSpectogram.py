from subprocess import Popen, PIPE, STDOUT
import os
import numpy as np
from PIL import Image
from random import shuffle
from keras.models import Sequential
import pickle

#Define
currentPath = os.path.dirname(os.path.realpath(__file__))
genres = ["Rap", "EDM", "Jazz", "Rock", "Classical"]
sliceSize = 128
basePath = "/Users/ananth/desktop/Audio"

def getProcessedImage(filename, size):
    image = Image.open(filename)
    image = image.resize((size,size), resample=Image.ANTIALIAS)
    imageData = np.asarray(image, dtype=np.uint8).reshape(size,size,1)
    imageData = imageData/255.
    return imageData

def createModel():
    model = Sequential()
    input_shape = (128,128,1) #change this based on slice size
    model.add(Conv2D(32, kernel_size=(2, 2), strides=(1, 1),
                 activation='elu',
                 input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (2, 2), activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (2, 2), activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (2, 2), activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='elu'))
    model.add(Dense(6, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy'])
    x_train, y_train, x_validate, y_validate, x_test, y_test = loadDataset()
    model.fit(x_train, y_train,
              batch_size=128,
              epochs=5,
              verbose=1,
              validation_data=(x_validate, y_validate),
              callbacks=[history])
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def loadDataset():
    datasetName = "Alldata"
    datasetPath = "/Users/ananth/desktop/Audio/Data/"
    print("[+] Loading training and validation datasets... ")
    train_X = pickle.load(open("{}train_X_{}.p".format(datasetPath, datasetName), "rb"))
    train_y = pickle.load(open("{}train_y_{}.p".format(datasetPath, datasetName), "rb"))
    validation_X = pickle.load(open("{}validation_X_{}.p".format(datasetPath, datasetName), "rb"))
    validation_y = pickle.load(open("{}validation_y_{}.p".format(datasetPath, datasetName), "rb"))
    test_X = pickle.load(open("{}test_X_{}.p".format(datasetPath, datasetName), "rb"))
    test_y = pickle.load(open("{}test_y_{}.p".format(datasetPath, datasetName), "rb"))
    print("    Training and validation datasets loaded! ✅")
    return train_X, train_y, validation_X, validation_y, test_X, test_y

#Saves dataset
def saveDataset(train_X, train_y, validation_X, validation_y, test_X, test_y, nbPerGenre, genres, sliceSize):
    print("running save dataset")
    datasetPath = "/Users/ananth/desktop/Audio/Data/"
     #Create path for dataset if not existing
    if not os.path.exists(os.path.dirname(datasetPath)):
        try:
            os.makedirs(os.path.dirname(datasetPath))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    #SaveDataset
    print("[+] Saving dataset... ")
    datasetName = "Alldata"
    pickle.dump(train_X, open("{}train_X_{}.p".format(datasetPath,datasetName), "wb" ))
    pickle.dump(train_y, open("{}train_y_{}.p".format(datasetPath,datasetName), "wb" ))
    pickle.dump(validation_X, open("{}validation_X_{}.p".format(datasetPath,datasetName), "wb" ))
    pickle.dump(validation_y, open("{}validation_y_{}.p".format(datasetPath,datasetName), "wb" ))
    pickle.dump(test_X, open("{}test_X_{}.p".format(datasetPath,datasetName), "wb" ))
    pickle.dump(test_y, open("{}test_y_{}.p".format(datasetPath,datasetName), "wb" ))
    print("    Dataset saved! ✅💾")

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
    print("Loop Done")
    shuffle(data)
    X, y = zip(*data)
    validationRatio = 0.2
    testRatio = 0.1

    #Split data
    validationNum = int(len(X)*validationRatio)
    testNum = int(len(X)*testRatio)
    trainNum = len(X)-(validationNum + testNum)
    print("Split Data Done")
    # Prepare for Tflearn at the same time
    train_X = np.array(X[:trainNum]).reshape([-1, sliceSize, sliceSize, 1])
    train_y = np.array(y[:trainNum])
    validation_X = np.array(X[trainNum:trainNum + validationNum]).reshape([-1, sliceSize, sliceSize, 1])
    validation_y = np.array(y[trainNum:trainNum + validationNum])
    test_X = np.array(X[-testNum:]).reshape([-1, sliceSize, sliceSize, 1])
    test_y = np.array(y[-testNum:])
    print("Data sets created")
    # Save
    saveDataset(train_X, train_y, validation_X, validation_y, test_X, test_y, nbPerGenre, genres, sliceSize)

    return train_X, train_y, validation_X, validation_y, test_X, test_y


def divideSpectogram():
    for genre in genres:
        directory = "{}/{}/spectograms/".format(basePath, genre)
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





def generateSpectogram():
    # Spectrogram resolution
    pixelPerSecond = 50
    for genre in genres:
        directory = "{}/{}/".format(basePath, genre)
        audioFiles = os.listdir(directory)
        audioFiles = [file for file in audioFiles]
        for file in audioFiles:
            if file.startswith('.') or file.startswith('spectograms'):
                continue
            command = "sox '{}' '/tmp/{}.mp3' remix 1,2".format(directory + file, file)
            #print(command)
            p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, cwd=currentPath)
            output, errors = p.communicate()
            if errors:
                print(errors)
            # Create spectrogram
            file.replace(".mp3", "")
            command = "sox '/tmp/{}.mp3' -n spectrogram -Y 200 -X {} -m -r -o '{}.png'".format(file, pixelPerSecond,
                                                                                               "{}spectograms/{}".format(directory, file))
            p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, cwd=currentPath)
            output, errors = p.communicate()
            if errors:
                print(errors)

            #Remove tmp mono track
            os.remove("/tmp/{}.mp3".format(file))

def main():
    #generateSpectogram()
    #divideSpectogram()
    prepareDataFromSlices()
    createModel()


if __name__ == "__main__":
    main()