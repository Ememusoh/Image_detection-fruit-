#%%
#import libraries
import tensorflow as tf
import pandas as pd
import numpy as np
import PIL.Image as Image
import os
import matplotlib.pyplot as plt

# %%
images_data_path = 'data/raw/images'
img_list = os.listdir(images_data_path)
# remove .DS_Store file if it exists and sort the list
if '.DS_Store' in img_list:
    img_list.remove('.DS_Store')
# sort the list by name, convert string to int for sorting
img_list = sorted(img_list, key=lambda x: int(x.split('.')[0]))

# add the images to the path
images = [os.path.join(images_data_path, file) for file in img_list]

#%%
# open the images
image = [Image.open(img) for img in images]
# show the first image
image[1].show()


#convert to numpy array

# # %%
# # convert images to numpy arrays
# image_array = [np.array(img) for img in image]

# image_array = [img.resize((224, 224)) for img in image_array]
# image_array = np.array(image_array)

# #%%
# # convert the list to a numpy array

# %%
resized_images = [img.resize((224, 224)) for img in image]

# Then convert to numpy
image_array = np.stack([np.array(img) for img in resized_images])

#%%
#load my labels
labels = pd.read_csv('data/raw/labels.csv')
#convert the label_name column to 0 or 1 encoding
labels['label_name']= labels['label_name'].map({'apple':1, 'no-apple':0})
labels['bbox_x_norm'] = round(labels['bbox_x']/labels['image_width'], 4)
labels['bbox_y_norm'] = round(labels['bbox_y']/labels['image_height'], 4)
labels['bbox_width_norm'] = round(labels['bbox_width']/labels['image_width'], 4)
labels['bbox_height_norm'] = round(labels['bbox_height']/labels['image_height'], 4)

Y = labels[['label_name', 'bbox_x_norm', 'bbox_y_norm', 'bbox_width_norm', 'bbox_height_norm']]
X = image_array.copy()/255.0
# %%
#convert Y to numpy array
Y_num = Y.to_numpy()

# %%
#split the data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_num, test_size=0.2, random_state =42)

#%%

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

x = layers.Input(shape=(224, 224, 3))
x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding= 'same', activation='relu')(x)


#%% design custom convolutional neural network to predict the bounding box and label
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
model = Sequential([
    layers.Conv2D(32, (3, 3), strides=(2, 2), padding= 'same', activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(5, activation='sigmoid')  # 1 for label, 4 for bbox
])

model.compile(optimizer='adam', loss='binary_crossentropy')
model.summary()
# %%
#code to train the model
history = model.fit(X_train, Y_train, epochs=10, batch_size=16, validation_data=(X_test, Y_test))
# %%
#evaluate the model
model.evaluate(X_test, Y_test)
# %%
#save the model
model.save('apple_detector_model.h5')       
# %%
#plot the training and validation loss
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.show()  
# %%
#make predictions
predictions = model.predict(X_test)
# %%
#show the first prediction
predictions[0]
# %%
#function to plot the image with the bounding box
def plot_image_with_bbox(image, bbox, label):
    plt.imshow(image)
    ax = plt.gca()
    if label == 1:
        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, color='red')
        ax.add_patch(rect)
        plt.text(bbox[0], bbox[1], 'Apple', color='white', fontsize=12, backgroundcolor='red')
    else:
        plt.text(10, 10, 'No Apple', color='black', fontsize=12, backgroundcolor='white')
    plt.axis('off')
    plt.show()
# %%#plot the first test image with the predicted bounding box
iix=3
pred_bbox = predictions[iix][1:]
#denormalize the bounding box
img_width, img_height = X_test[iix].shape[1], X_test[iix].shape[0]
bbox = [pred_bbox[0]*img_width, pred_bbox[1]*img_height, pred_bbox[2]*img_width, pred_bbox[3]*img_height]   
plot_image_with_bbox(X_test[iix], bbox, round(predictions[iix][0]))
# %%
