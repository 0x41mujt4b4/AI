import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras import models, layers
import tensorflow as tf

# os.chdir('./Data')
# os.listdir('Train_Data')
os.getcwd()

train_dir = 'Train_Data'
test_dir = 'Test_Data'
validation_dir = 'Val_Data'

CLASSES_REQUIRED = os.listdir('Train_Data')
KERNEL_SIZE = (5, 5)
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
RESIZE_TO = (224, 224)
BATCH_SIZE_TRAIN = 20
BATCH_SIZE_VAL = 10
BATCH_SIZE_TEST = 25

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=RESIZE_TO,
    batch_size=BATCH_SIZE_TRAIN,
    classes=CLASSES_REQUIRED)#class_mode='categorical'

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=RESIZE_TO,
    batch_size=BATCH_SIZE_TEST,
    classes=CLASSES_REQUIRED)#class_mode='categorical'

validation_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=RESIZE_TO,
    batch_size=BATCH_SIZE_VAL,
    classes=CLASSES_REQUIRED)#class_mode='categorical'


# create a convolution layer
def conv(num_filters):
  return layers.Conv2D(num_filters, KERNEL_SIZE, activation='relu')
# craete a maxpooling layer
def maxp():
  return layers.MaxPooling2D((2, 2))


# create model
model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3),
                        activation='relu',
                        input_shape=(IMAGE_WIDTH,
                                     IMAGE_HEIGHT,
                                     3
                                     )))
model.add(conv(64))
model.add(maxp())

model.add(conv(64))
model.add(maxp())

model.add(conv(64))
model.add(maxp())

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(train_generator.num_classes,
                       activation='softmax'))
# model.summary()


# Complie the model
model.compile(tf.keras.optimizers.Adam(learning_rate=.00015), loss='categorical_crossentropy', metrics=['accuracy'])


#training the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples//BATCH_SIZE_TRAIN,
    epochs=5,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples//BATCH_SIZE_VAL
    )

# model.save('working_model.h5')

test_loss, test_acc = model.evaluate(test_generator)
print('Test accuracy:', test_acc)

X_test=[]
'''Converting Data Format according to the backend used by Keras
'''
# datagen=ImageDataGenerator(data_format=K.image_data_format())
def convert_to_image(X):
    '''Function to convert all Input Images to the STANDARD_SIZE and create Training Dataset
    '''
    for f in os.listdir('test_doc-external'):
        if os.path.isdir(f):
            continue
        img = image.load_img('test_doc-external/'+f, target_size=RESIZE_TO)
#         x = image.img_to_array(img)
#         x = np.expand_dims(x, axis=0)
        img=np.array(img)
        X.append(img)
    return X

X_test=np.array(convert_to_image(X_test))
# datagen.fit(X_test)


labels = train_generator.class_indices# == test_generator.class_indices == validation_generator.class_indices
# np.save('labels', labels)
# labels


predictions = model.predict(X_test)
# # predictions = (model.predict(X_test) > 0.5).astype("int32")
# classes_x=np.argmax(predictions,axis=1)
# # predictions
# classes_x
# for pred in classes_x:
#   print(pred.data.shape)
#   for key in labels:
#     print(key) if pred == labels[key] else None


x_classes = np.argmax(predictions, axis=1)
# x_classes


np.load.__defaults__=(None, True, True, 'ASCII')
class_indices = np.load('labels.npy').item()
classes = dict(enumerate(class_indices.keys()))
# classes


for index, result in enumerate(x_classes):
    print(str(predictions[index][result]), classes[result])
# predictions[0][result]

