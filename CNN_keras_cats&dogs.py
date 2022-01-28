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

"""------------------------------------------------------------------------------
Para que os dados de entrada em cada camada sejam capazes de aprender
e executar tarefas mais complexas, a função de ativação realiza uma transformação não-linear nesses dados, isso faz com que possam aprender mais do que relações lineares. Existem diversas funções de ativação que podem ser usadas, algumas delas são:

• **Função de ativação sigmoid:** Frequentemente essa função é utilizada por
redes neurais com propagação positiva (Feedforward) que precisam ter como saída apenas números positivos, em redes neurais multicamadas e em outras redes com sinais contínuos;

• **Função de ativação tanh:** Possui uso muito comum em redes neurais cujas
saídas devem ser entre -1 e 1;

• **Função de ativação softmax:** Os elementos do vetor de saída estão no
intervalo (0, 1) e somam 1, é frequentemente usado como a ativação para a última camada de uma rede de classificação porque o resultado pode ser interpretado como uma distribuição de probabilidade.

• **Função de ativação ReLU:** A função de ativação linear retificada é uma
função linear por partes que produzirá a entrada diretamente se for positiva, caso contrário, ela produzirá zero. Ela se tornou a função de ativação padrão para muitos tipos de redes neurais porque um modelo que a usa é mais fácil de treinar e geralmente atinge um melhor desempenho.

------------------------------------------------------------------------------
**Max-pooling:** Reduzir agressivamente os mapas de características, tal como as convoluções de strided. O Max-pooling consiste em extrair janelas dos mapas de características de entrada e emitir o valor máximo de cada canal. É conceptualmente semelhante à convolução, excepto que, em vez de transformar remendos locais através de uma transformação linear aprendida (o núcleo de convolução), são transformados através de uma operação de tensão máxima hardcoded.
"""

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

"""

Otimizadores são a classe expandida, que inclui o método para treinar o
modelo de aprendizado profundo. Otimizadores corretos são necessários, pois
melhoram a velocidade e o desempenho do treinamento.

• **Adagrad:** adapta a taxa de aprendizado especificamente com recursos
individuais: isso significa que alguns dos pesos em seu conjunto de dados têm taxas de aprendizado diferentes de outros. Sempre funciona melhor em um conjunto de dados esparso onde muitas entradas estão faltando;

• **Adam:** significa estimativa de momento adaptável, que é outra maneira de
usar gradientes anteriores para calcular gradientes atuais, utiliza o conceito de momento adicionando frações de gradientes anteriores ao atual, é praticamente aceito em muitos projetos durante o treinamento de redes neurais;

• **SGD:** realiza uma atualização de parâmetro para cada exemplo de
treinamento, executa cálculos redundantes para conjuntos de dados maiores. Ele
realiza atualizações frequentes com uma alta variação que faz com que a função
objetivo oscile fortemente.

------------------------------------------------------------------------------

A função de perda é um dos componentes importantes das redes neurais. A
perda nada mais é do que um erro de previsão da rede neural. A perda é usada para calcular os gradientes e gradientes são usados para atualizar os pesos da rede neural, é assim que uma Rede Neural é treinada. Algumas das funções de perda que podem ser usadas são:

• **Erro Quadrático Médio (MSE)**: A função MSE é usada para tarefas de
regressão. Como o nome sugere, essa perda é calculada tomando a média das
diferenças quadradas entre os valores reais (alvo) e previstos;

• **Binary Crossentropy:** Essa função é usada para as tarefas de classificação binária. Caso seja usada a função de perda de binary crossentropy, é necessário um nó de saída para classificar os dados em duas classes. O valor de saída deve ser passado por uma função de ativação sigmóide e a faixa de saída é (0 - 1);

• **Categorical Crossentropy:** Quando se tem uma tarefa de classificação
multiclasse, uma das funções de perda que pode seguir é esta. Ao utilizar esta
função de perda, deve haver o mesmo número de nós de saída que as classes. E a
saída da camada final deve ser passada por uma ativação de softmax para que cada nó emita um valor de probabilidade entre (0–1).

"""

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(x=train_batches, validation_data=valid_batches, epochs=10, verbose=2)

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

"""

As métricas de avaliação têm como objetivo, como o próprio nome diz, avaliar
a qualidade do modelo escolhido e as predições realizadas pelo aprendizado
profundo.
Para entender melhor os métodos de avaliação, antes precisamos saber
sobre a matriz de confusão. A matriz de confusão é uma tabela que indica os
acertos e erros do modelo.

Tipos de métricas de avaliação:

• **Acurácia:** o método representa a divisão entre todos os acertos do modelo
pelo total, a acurácia não deve ser utilizada para avaliação de dados que sejam compostos em sua maioria por um único tipo de resultado.


• **Precisão: **o método de precisão tenta responder à pergunta – Qual a
proporção das predições que foram identificadas como positivo que realmente
estavam corretas?

• **Recall:** já o método de recall tenta responder à pergunta – “Qual proporção
de positivos reais foi identificada corretamente?”

• **F1 Score:** representa a média harmônica entre os métodos de Precisão e
Recall


"""