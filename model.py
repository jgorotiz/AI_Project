from utils import prep_pixels, load_data, classification, classification_inverse
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.optimizers import SGD
from keras.models import load_model
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import random
random.seed(0)


def prototype_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(72, 128, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (5, 5), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (5, 5), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (5, 5), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, (5, 5), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (5, 5), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (5, 5), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dense(2, activation='softmax'))
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
# this is for normalization
def test_model(pathModel,folder_test):
    model = load_model(pathModel)
    X_test, Y_test = load_data(folder_test)
    X_test = prep_pixels(X_test)
    scoresTest = model.evaluate(X_test, Y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scoresTest[1]*100))

def model(folder_train, folder_validation, folder_test):
    X_train,Y_train = load_data(folder_train) #this not resize the input
    X_validation,Y_validation = load_data(folder_validation)
    X_test, Y_test = load_data(folder_test)
    X_train = prep_pixels(X_train)
    X_validation = prep_pixels(X_validation)
    X_test = prep_pixels(X_test)
    
    fig = plt.figure(figsize=(5,5))
    for i in range(6):
        #plt.show(X_train[i])
        #input()
        ax = fig.add_subplot(1, 6, i+1, xticks=[], yticks=[])
        ax.imshow(X_train[i])
        ax.set_title(str(Y_train[i]))
    plt.show()
    
    model = prototype_model()
    steps = int(X_train.shape[0] / 64)
    print  (model.summary())
    input()
    # steps = 1
    model.fit(x=X_train, y=Y_train, steps_per_epoch=steps, epochs=100, validation_data=(X_validation, Y_validation), validation_steps=20, verbose=1)
    """history = model.fit(x=X_train, y=Y_train, steps_per_epoch=steps, epochs=200, validation_data=(X_validation, Y_validation), validation_steps=20, verbose=1)
    plt.plot(history.history['loss'])
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.savefig('Loss.png')
    plt.clf()

    plt.plot(history.history['acc'])
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.savefig('Accuracy.png')
    plt.clf()

    plt.plot(history.history['val_loss'])
    plt.xlabel('Steps')
    plt.ylabel('Validation Loss')
    plt.savefig('Val_loss.png')
    plt.clf()

    plt.plot(history.history['val_acc'])
    plt.xlabel('Steps')
    plt.ylabel('Validation Accuracy')
    plt.savefig('Val_accuracy.png')
    plt.clf()

    acc, = plt.plot(history.history['acc'], label="Accuracy")
    loss, = plt.plot(history.history['loss'], label="Loss")
    val_loss, = plt.plot(history.history['val_loss'], label="Validation Loss")
    val_acc, = plt.plot(history.history['val_acc'], label="Validation Accuracy")
    plt.xlabel('Steps')
    plt.legend(handles=[acc, loss, val_loss,val_acc])
    plt.savefig('general.png')
    plt.clf()


"""
    scoresTraining = model.evaluate(X_train, Y_train)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scoresTraining[1]*100))

    scoresTest = model.evaluate(X_test, Y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scoresTest[1]*100))
    # guardar el model
    model.save('final_model.h5')




if __name__ == "__main__":
    print(len(sys.argv))
    if len(sys.argv) < 4:
        print("Faltan argumentos")
        print("Existen dos maneras de usar el modelo (1 para entrenar, 2 para cargar un modelo ya existente)")
        print("\t1. python model.py <NOMBRE DE LA CARPETA DE ENTRENAMIENTO> <NOMBRE DE LA CARPETA DE VALIDACION> <NOMBRE DE LA CARPETA DE PRUEBA>")
        print("\t2. python model.py test <NOMBRE DEL ARCHIVO .h5> <NOMBRE DE LA CARPETA DE PRUEBA>")
    else:
        if sys.argv[1].lower() == "test":
            print(sys.argv[2], sys.argv[3])
            input()
            test_model(sys.argv[2], sys.argv[3])
        else:
            folder_train = sys.argv[1]
            folder_validation = sys.argv[2]
            folder_test = sys.argv[3]
            model(folder_train, folder_validation, folder_test)
            #test_model('final_model.h5',folder_test )
    

