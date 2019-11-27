from keras.models import  Sequential, Model
from keras.layers import Dense, GlobalAveragePooling2D,Input, Activation, Dropout, Flatten
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras import backend as K

import os


n_categories=2
classes = ['me','notme']
n_epoch = 100
batch_size=32
train_dir='./train'
validation_dir='./validation'

dirname = os.path.basename(os.path.dirname(os.path.abspath(__file__)))


def plot_history(history):
    plt.plot(history.history['acc'],".-",label="accuracy")
    plt.plot(history.history['val_acc'],".-",label="val_acc")
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc="lower right")
    plt.savefig('accuracy'+dirname+'.png')
    plt.clf()

    plt.plot(history.history['loss'],".-",label="loss",)
    plt.plot(history.history['val_loss'],".-",label="val_loss")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='lower right')
    plt.savefig('loss'+dirname+'.png')


if __name__ == '__main__':
    vgg16_model=VGG16(weights='imagenet',include_top=False,input_tensor=Input(shape=(224,224,3)))

    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(2, activation='softmax'))

    model = Model(input=vgg16_model.input, output=top_model(vgg16_model.output))

    for layer in model.layers[:15]:
        layer.trainable=False
    
    model.compile(optimizer=SGD(lr=1e-4,momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    model.summary()

    train_datagen = ImageDataGenerator(rescale=1.0/255)
    validation_datagen = ImageDataGenerator(rescale=1.0/255)

    train_generator=train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    color_mode='rgb', 
    classes=classes,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
    )

    validation_generator=validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(224,224),
        color_mode='rgb', 
        classes=classes,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    hist=model.fit_generator(train_generator,
                        epochs=n_epoch,
                        verbose=1,
                        validation_data=validation_generator)

    model_json_str = model.to_json()
    open(dirname+'.json', 'w').write(model_json_str)


    plot_history(hist)
    model.save(dirname+'.h5')

    K.clear_session()
