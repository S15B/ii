from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.utils import to_categorical 
np.random.seed(1671)

k = 0

# Сеть и обучение 
NB_EPOCH = 20
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10 
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2
DROPOUT = 0.3

# Данные
(X_train, y_train), (X_test, y_test) = mnist.load_data()
RESHAPED = 784

# Преобразуем 60000 * 24*24 -> 60000 * 784
X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Нормировка
X_train /= 255
X_test /= 255

# Преобразование векторов классов в бинарные матрицы классов
y_train = to_categorical(y_train, NB_CLASSES)
y_test = to_categorical(y_test, NB_CLASSES)

# Модели нейронных сетей
model_1 = Sequential()
model_1.add(Dense(N_HIDDEN, Activation('relu'), input_shape=(RESHAPED,)))
model_1.add(Dense(N_HIDDEN, Activation('relu')))
model_1.add(Dense(N_HIDDEN, Activation('relu')))
model_1.add(Dense(NB_CLASSES, Activation('softmax')))
model_1.summary()
model_1.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate = 0.01), metrics=['accuracy'])

model_2 = Sequential()
model_2.add(Dense(N_HIDDEN * 2, Activation('elu'), input_shape=(RESHAPED,)))
model_2.add(Dense(N_HIDDEN * 2, Activation('elu')))
model_2.add(Dense(NB_CLASSES, Activation('softmax')))
model_2.summary()
model_2.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate = 0.01), metrics=['accuracy'])

model_3 = Sequential()
model_3.add(Dense(N_HIDDEN, Activation('relu'), input_shape=(RESHAPED,)))
model_3.add(Dense(N_HIDDEN, Activation('relu'), Dropout(DROPOUT)))
model_3.add(Dense(N_HIDDEN, Activation('relu'), Dropout(DROPOUT)))
model_3.add(Dense(NB_CLASSES, Activation('softmax')))
model_3.summary()
model_3.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate = 0.01), metrics=['accuracy'])

model_4 = Sequential()
model_4.add(Dense(N_HIDDEN, Activation('relu'), input_shape=(RESHAPED,)))
model_4.add(Dense(N_HIDDEN, Activation('relu'), Dropout(DROPOUT)))
model_4.add(Dense(NB_CLASSES, Activation('softmax')))
model_4.summary()
model_4.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate = 0.01), metrics=['accuracy'])

# Обучение 
history_1 = model_1.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
history_2 = model_1.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
history_3 = model_1.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
history_4 = model_1.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

# Построение графика точности
for i in [history_1, history_2, history_3, history_4]:
    k += 1
    plt.subplot(2, 2, k)
    plt.plot(i.history['accuracy'], label='Точность на обучающем наборе')
    plt.plot(i.history['val_accuracy'], label='Точность на тестовом наборе')
    plt.title('Модель {0}'.format(k))
    plt.ylabel('Точность')
    plt.legend()

plt.show()

# Построение графика потерь
for i in [history_1, history_2, history_3, history_4]:
    k += 1
    plt.subplot(2, 2, k-4)
    plt.plot(i.history['loss'], label='Потери на обучающем наборе')
    plt.plot(i.history['val_loss'], label='Потери на тестовом наборе')
    plt.title('Модель {0}'.format(k-4))
    plt.ylabel('Потери')
    plt.legend()

plt.show()