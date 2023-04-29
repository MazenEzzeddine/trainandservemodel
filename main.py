import requests
import tensorflow as tf
import random
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile


def loadImages():
    # import Fashion MNIST Dataset using Keras
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    # Data Normalization -> Between 0 and 1
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    print(X_train.shape)
    print(X_test.shape)
    #####preprocess
    # Reshape training data to be = (60000, 28, 28, 1) instead of (60000, 28,28)
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    # Do the same for the testing dataset
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)


    return X_train, y_train, X_test, y_test

def printFigures(X_train,y_train):
    W_grid = 4
    L_grid = 4

    fig, axes = plt.subplots(L_grid, W_grid, figsize=(15, 15))
    axes = axes.ravel()

    n_training = len(X_train)

    for i in np.arange(0, L_grid * W_grid):
        index = np.random.randint(0, n_training)  # pick a random number
        axes[i].imshow(X_train[index].reshape(28, 28))
        axes[i].set_title(y_train[index])
        axes[i].axis('off')

    plt.subplots_adjust(hspace=0.4)
    plt.show()

def buildModel():
    cnn = tf.keras.models.Sequential()

    cnn.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    cnn.add(tf.keras.layers.MaxPooling2D(2, 2))
    cnn.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

    cnn.add(tf.keras.layers.Flatten())

    cnn.add(tf.keras.layers.Dense(64, activation='relu'))

    cnn.add(tf.keras.layers.Dense(10, activation='softmax'))
    cnn.summary()
    return cnn

def trainModel(cnn, epochs):


    cnn.compile(optimizer=tf.keras.optimizers.Adam(),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    cnn.fit(X_train, y_train, epochs=epochs)
    test_loss, test_acc = cnn.evaluate(X_test, y_test)
    print('\nTest accuracy: {}'.format(test_acc))

def saveModel(model):
    # Let's obtain a temporary storage directory
    ##MODEL_DIR = tempfile.gettempdir()
    # Let's specify the model version, choose #1 for now
    version = 1

    ##export_path = os.path.join(MODEL_DIR, str(version))
    #print('export_path = {}\n'.format(export_path))
    # Let's save the model using simple_save
    # If the directory already exists, we will remove it using '!rm'
    # rm removes each file specified on the command line.

    tf.saved_model.save(model, "C:\\Users\\m.ezzeddine\\PycharmProjects\\fashiondataset\\mdl\\1")


def showFigure(id, title):
    plt.figure()
    plt.imshow(X_test[id].reshape(28, 28))
    plt.title('\n\n{}'.format(title), fontdict={'size': 16})
    plt.show()




if __name__ == '__main__':
    X_train, y_train, X_test, y_test = loadImages()
    # print(X_test.shape)
    # #printFigures(X_train,y_train)
    # buildModel()
    # model = buildModel()
    # trainModel(model, 1)
    # saveModel(model)
    rando = random.randint(0, len(X_test) - 1)
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    showFigure(rando, 'An Example Image: {}'.format(class_names[y_test[rando]]))

    # Let's create a JSON object and make 3 inference requests
    # data = json.dumps({"signature_name": "serving_default", "instances": X_test[0:3].tolist()})
    data = json.dumps({ "instances": X_test[0:3].tolist()})
    print('Data: {} ... {}'.format(data[:50], data[len(data) - 52:]))

    headers = {"content-type": "application/json"}
    json_response = requests.post('http://localhost:8501/v1/models/mdl:predict', data=data, headers=headers)
    predictions = json.loads(json_response.text)['predictions']


    for i in range(0, 3):
        showFigure(i, 'The model thought this was a {} (class {}), and it was actually a {} (class {})'.format(
            class_names[np.argmax(predictions[i])], y_test[i], class_names[np.argmax(predictions[i])], y_test[i]))










