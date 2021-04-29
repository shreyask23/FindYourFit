import os
import sys
import random
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

model = tf.keras.models.load_model("out.h5")

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
img = int(sys.argv[1])

print("Running Inference")

out = ind_to_labels[np.argmax(model.predict(np.array([images[img]])))]

print("Indexing Other Images")
for i in range(int(sys.argv[2])):
	os.system("mv clothing-dataset/images/" + image_data[image_data["label"] == out].iloc[i]["image"] + ".jpg " + sys.argv[3])
