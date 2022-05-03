# %%
import tensorflow as tf 
import cv2 
import os
import matplotlib.pyplot as plt 
import numpy as np
import random

from tensorflow import keras
from tensorflow.keras import layers
from fer import FER 


# %%
img_array = cv2.imread('Raw/fer_2013/train/angry/Training_3908.jpg')
plt.imshow(img_array)


# %%
Data_dir = 'Raw/fer_2013/train'
# Classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'suprise']
Classes = ["0","1","2","3","4","5","6"]


# %%
img_size = 224 
new_array = cv2.resize(img_array, (img_size, img_size))
plt.imshow(cv2.cvtColor(new_array, cv2.COLOR_BGR2RGB))
plt.show()

# %%
training_Data = []
img_size = 224 

def create_training_Data():
  for category in Classes:    # For each Class of the Dataset

    path = os.path.join(Data_dir, category)
    class_num = Classes.index(category)
    
    for img in os.listdir(path)[0:400]:
      try:
        img_array = cv2.imread(os.path.join(path, img))
        new_array = cv2.resize(img_array, (img_size, img_size))
        training_Data.append([new_array, class_num])
      except Exception as e:
        pass

# %%
create_training_Data()

# %%
# Shuffle the Dataset, make it robust and dynamic
random.shuffle(training_Data)

# Seperate the features and labels

X = []
y = []

# Setting the features and labels

for features, label in training_Data:
  X.append(features)
  y.append(label)

X = np.array(X).reshape(-1, img_size, img_size, 3) 

# Normalize the Dataset

X = X / 255.0



# %%
# Call Model, we'll be using MobileNetV2

model = tf.keras.applications.MobileNetV2()

# Base Input
base_input = model.layers[0].input

base_output = model.layers[-2].output

final_output = layers.Dense(128)(base_output) 
final_output = layers.Activation('relu')(final_output) 
final_output = layers.Dense(64)(final_output)
final_output = layers.Activation('relu')(final_output)
final_output = layers.Dense(7, activation='softmax')(final_output)

# %%
X = np.array(X)
y = np.array(y)

# %%
new_model = keras.Model(inputs = base_input, outputs = final_output)

new_model.compile(loss="sparse_categorical_crossentropy", optimizer = 'adam', metrics = ['accuracy'])

# %%
new_model.fit(X, y, epochs = 25)


# %%
new_model.save('Emotional_Model.h5')

# %%
new_model.save_weights("weights.h5")

# %%
from keras.models import load_model

model = load_model('Emotional_Model.h5')

# %%
bounding_box = results[0]["box"]
emotions = results[0]["emotions"]

# Draw the Rectangles around the face

rectangle = cv2.rectangle(X_test,(bounding_box[0], bounding_box[1]),
(bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),(0, 155, 255), 2,)

plt.imshow(X_test)
plt.imshow(rectangle)

for idx, (emotion, score) in enumerate(emotions.items()):
    color = (211, 211, 211) if score < 0.01 else (255, 0, 0)
    emotion_score = "{}: {}".format(
          emotion, "{:.2f}".format(score) if score > 0.01 else ""
        )
    cv2.putText(X_test,emotion_score,
            (bounding_box[0], bounding_box[1] + bounding_box[3] + 30 + idx * 15),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1,cv2.LINE_AA,)
cv2.imwrite("emotion.jpg", X_test)
plt.figure(figsize = (15,6))
plt.imshow(X_test)
plt.imshow(rectangle)


