import array
from pydub import AudioSegment
from pydub import utils
from pydub.playback import play
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#encoder matters!!
#load raw audio data
audio_X = AudioSegment.from_mp3("lq_spaceoddity.mp3")
audio_y = AudioSegment.from_mp3("hq_spaceoddity.mp3")

#get sample array turn into numpy array
X = np.array(audio_X.get_array_of_samples())
y = np.array(audio_y.get_array_of_samples())
#make 2d
X = np.reshape(X, (-1, 1))
y = np.reshape(y, (-1, 1))

#split the data 80/20
split = np.round(X.size * .8).astype(int)

# we need to split so that we know how to reassemble 
X_train = X[0:split]
X_test = X[split:]
y_train = y[0:split]
y_test = y[split:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

#predict and round
train_pred = np.round(regr.predict(X_train)).astype(int)
test_pred = np.round(regr.predict(X_test)).astype(int)

# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((test_pred - y_test)) ** 2)
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X_test, y_test))

# reassemble and convert to array.array  
array_type = utils.get_array_type(audio_X.sample_width * 8)
combined = np.concatenate((train_pred, test_pred), axis=0).flatten()
predicted = array.array(array_type,combined.tolist())

#create audio segment and export
upscaled = audio_X._spawn(predicted)
upscaled.export( "upscaled.mp3" , format="mp3", bitrate="320k")

# play(upscaled)

# # Plot outputs
# plt.scatter(X_test, y_test,  color='black')
# plt.plot(X_test, regr.predict(X_test), color='blue',
#          linewidth=3)

# plt.xticks(())
# plt.yticks(())

# plt.show()
