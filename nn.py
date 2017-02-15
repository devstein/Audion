from keras.models import Sequential
from keras.layers import Dense, Activation
import pydub as pd
import numpy as np

# Training Data
audio_X = pd.AudioSegment.from_mp3("lq_spaceoddity.mp3")
audio_Y = pd.AudioSegment.from_mp3("hq_spaceoddity.mp3")

samples_X = np.array(audio_X.get_array_of_samples())
samples_Y = np.array(audio_Y.get_array_of_samples())

N = 1152
data_X = np.array([samples_X[n:n+N] for n in range(0, len(samples_X), N)])
data_Y = np.array([samples_Y[n:n+N] for n in range(0, len(samples_Y), N)])

data_X = data_X[:(len(data_X) / N) * N]
data_X = np.reshape(data_X, (len(data_X) / N, N))

data_Y = data_Y[:(len(data_Y) / N) * N]
data_Y = np.reshape(data_Y, (len(data_Y) / N, N))

split = int(np.round(data_X.size * .6))

train_X = data_X[0:split]
test_X = data_X[split:]
train_Y = data_Y[0:split]
test_Y = data_X[split:]

# Model
model = Sequential()

model.add(Dense(1152, input_dim=1152, init='normal', activation='relu'))
model.add(Dense(1152, init='normal', activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


# Fit
model.fit(train_X, train_Y, nb_epoch=5, batch_size=32)


# Test
loss_and_metrics = model.evaluate(test_X, test_Y, batch_size=32)
