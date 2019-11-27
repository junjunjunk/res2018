import os
import sys
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.preprocessing import image
from keras.models import model_from_json
from keras.optimizers import SGD
from keras.preprocessing.image import img_to_array, list_pictures, load_img
from keras import backend as K
import numpy as np

classes = ['me', 'notme']

if len(sys.argv) != 2:
    print("usage: python predict.py [filename]")
    sys.exit(1)

filename = sys.argv[1]
print('input:', filename)

dirname = os.path.basename(os.path.dirname(os.path.abspath(__file__)))

model = model_from_json(open(dirname+'.json').read())
model.load_weights(dirname+'.h5')

model.compile(optimizer=SGD(lr=1e-4, momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

img = image.load_img(filename, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x / 255.0

pred = model.predict(x)[0]

top_indices = pred.argsort()[-1:][::-1]
result = [(classes[i], pred[i]) for i in top_indices]

print("result:", result[0][0], sep='')
if result[0][0] == 'me':
    print("percent:", result[0][1]*100, sep='')
else:
    print("percent:", result[1][1]*100, sep='')

K.clear_session()
