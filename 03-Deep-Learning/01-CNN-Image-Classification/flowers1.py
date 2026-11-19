from configparser import Interpolation
import os 
import cv2
import numpy as np
from scipy import interpolate
os.chdir("C:\\YAPAY_ZEKA\\flowers")
path = os.getcwd()+'\\flowers\\'
folderNames = os.listdir(path)

xSample, ySample = [], []

for i, folderName in enumerate(folderNames): 
    for j, file in enumerate([f for f in os.listdir(path + folderName) if os.path.isfile(os.path.join(path + folderName, f))]):
        img = cv2.resize(cv2.imread(path + folderName + '/'+ file), (100, 100), interpolation=cv2.INTER_LINEAR)/255
        xSample.append(np.asarray(img))
        ySample.append(i)

xSample = np.array(xSample)
ySample = np.array(ySample)
# print(xSample.shape)
# print(ySample.shape)
# print(xSample)
# print(ySample)


from sklearn.model_selection import train_test_split
XTrain, XTest, yTrain, yTest= train_test_split(xSample, ySample, test_size=0.10, random_state=0)
XTrain, XVal, yTrain, yVal= train_test_split(XTrain, yTrain, test_size=0.111, random_state=0)

print(XTrain.shape)
print(XTest.shape)
print(XVal.shape)


import tensorflow as tf
from tensorflow.keras import datasets,layers, models

model=models.Sequential()
model.add(layers.Conv2D(100,(3,3),activation='relu',input_shape=(100,100,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(100,(3,3),activation='relu',input_shape=(100,100,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.BatchNormalization())
model.add(layers.Flatten())
model.add(layers.BatchNormalization())
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(5, activation="softmax"))
model.summary()

model.compile(optimizer='rmsprop',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(XTrain, yTrain, epochs=100,
                    validation_data=(XVal, yVal))

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Accuracy/Loss')
plt.ylabel('Accuracy/Loss')
plt.xlabel('Epoch')
plt.legend(['Train Acc','Val Acc', 'Train Loss', 'Val Loss'], loc='center right')
plt.show()


from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt 
import numpy as np

yTestPred=model.predict(XTest)
yTestPred=np.argmax(yTestPred, axis=1)

print(classification_report(yTest,yTestPred))
cm=confusion_matrix(yTest,yTestPred,normalize="true").round(2)
disp=ConfusionMatrixDisplay(confusion_matrix=cm)
disp=disp.plot(cmap=plt.cm.GnBu, values_format='g')
plt.title("Train Confusion Matrix", style="italic")
plt.show()