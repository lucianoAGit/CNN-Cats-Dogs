# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import numpy as np
import random 
import itertools
import os
import shutil
import glob
import warnings
import itertools
import matplotlib.pyplot as plt
warnings.simplefilter(action='ignore', category=FutureWarning)

# Caminho das imagens selecionadas
train_path = '/content/drive/MyDrive/USP/CNN Cats&Dogs/train'
valid_path = '/content/drive/MyDrive/USP/CNN Cats&Dogs/valid'
test_path = '/content/drive/MyDrive/USP/CNN Cats&Dogs/test'

# Geração das imagens no formato da CNN
train_batches = ImageDataGenerator(preprocessing_function= tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path,target_size=(224,224), classes= ['cat','dog'], batch_size= 10)
    
valid_batches = ImageDataGenerator(preprocessing_function= tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_path,target_size=(224,224), classes= ['cat','dog'], batch_size= 10)
test_batches = ImageDataGenerator(preprocessing_function= tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path,target_size=(224,224), classes= ['cat','dog'], batch_size= 10, shuffle= False)

# Pega um unico lote de imagens e os respectivos rotulos
imgs,labels = next(train_batches)

def plotImages(images_arr):
  fig,axes = plt.subplots(1,10,figsize=(20,20))
  axes = axes.flatten()
  for img, ax in zip(images_arr, axes):
    ax.imshow(img)
    ax.axis('off')
  plt.tight_layout()
  plt.show()

plotImages(imgs)
print(labels)

# Modelagem do modelo
model = Sequential([
        Conv2D(filters=32, kernel_size=(3,3), activation='relu',padding='same', input_shape=(224,224,3)),
        MaxPool2D(pool_size=(2,2), strides=2),
        Conv2D(filters =64, kernel_size= (3,3), activation='relu', padding = 'same'),
        MaxPool2D(pool_size=(2,2), strides=2),
        Flatten(),
        # camada de saida com duas labels  gato/cachorro
        Dense(units=2, activation='softmax'),                 
    ])

model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(x=train_batches, validation_data=valid_batches, epochs=10, verbose=2)

# Visualização do dicionario gerado
history_dict = history.history
print(history_dict)

# Plotando Resultados
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
plt.plot(epochs, acc_values, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Predicao
predictions = model.predict(x = test_batches,verbose=0)
np.round(predictions)

# Matriz de confusao
cm = confusion_matrix(y_true= test_batches.classes, y_pred=np.argmax(predictions, axis=-1))

def plot_confusion_matrix(cm, classes, normalize=False, title = 'Confusion matrix', cmap=plt.cm.Blues):
  plt.imshow(cm,interpolation='nearest',cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks,classes,rotation=45)
  plt.yticks(tick_marks, classes)

  if normalize:
    cm= cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
    print("Normalized confusion matrix")
  else:
    print('confusion matrix, without normalization')
  print(cm)

  thresh= cm.max() / 2.
  for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
    plt.text(j,i,cm[i,j],horizontalalignment="center",color = "white" if cm[i,j] > thresh else "black")
  
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')

test_batches.class_indices

cm_plot_labels = ['Cat','Dog']
plot_confusion_matrix(cm=cm,classes= cm_plot_labels,title='Confusion Matrix')
