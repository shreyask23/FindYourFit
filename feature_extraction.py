import os
# os.environ['KERAS_BACKEND'] = "plaidml.keras.backend"
# import keras
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten

image_data = pd.read_csv("clothing-dataset/images.csv")

labels = set(list(image_data["label"]))
labels.remove("Not sure")

labels = list(labels)

ind_to_labels = dict([(i, labels[i]) for i in range(len(labels))])
labels_to_ind = dict([(labels[i], i) for i in range(len(labels))])

images = []
assoc_labels = []

for i in range(len(image_data)):
    if (i % 1000) == 0:
        print(i)

    if image_data.iloc[i]["label"] != "Not sure":
        try:
            images.append(np.array(Image.open("clothing-dataset/images/" + image_data.iloc[i]["image"] + ".jpg").resize((224, 224))))
            temp = [0] * len(labels)
            temp[labels_to_ind[image_data.iloc[i]["label"]]] = 1
            assoc_labels.append(temp)
        except:
            continue

images = np.array(images)[:4000]
assoc_labels = np.array(assoc_labels)[:4000]

print(images.shape)
print(assoc_labels.shape)

model = Sequential()
model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=len(labels), activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(images, assoc_labels, batch_size=128, epochs=15, validation_split=0.1, verbose=1)

model.save("out.h5")
