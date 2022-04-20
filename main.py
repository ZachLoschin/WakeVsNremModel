import h5py
import numpy as np
import torch

filepath = '/content/Rat08-20130711_017.h5'  # data file
f = h5py.File(filepath, 'r')  # read data with h5 format
fs = f.attrs['fs'][0]  # get sampling frequency of LFP signal (Hz)

states = []  # two states (NREM & WAKE) to be classified
# LFP recordings are store in two h5 groups for each state
# Under each h5 group, the LFP recordings are divided into several segments with different lengths.

for name, grp in f.items():
    states.append(name)

# Convert the recording in to numpy arrays
# Use a dictionary to store the LFP recordings of the two states
# each containing a list of numpy arrays of all segments
lfp = {key: [] for key in states}
for key in states:
    group = f[key]  # h5 group of a state
    n = len(group)  # number of segments
    for i in range(n):
        lfp[key].append(group[str(i+1)][()].astype(float))  # convert data to numpy array and from int type to float type

# Find minimum y-axis data length to construct pxx nfft later
data = []
for i in range(len(lfp['WAKE'])):
    x = lfp['WAKE'][i]
    data.append(len(x))
for i in range(len(lfp['NREM'])):
    x = lfp['NREM'][i]
    data.append(len(x))
minimum = min(data)

import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.signal as signal

def colorPeriodogram(input):
    x = lfp['WAKE'][input]  # accessing the 10-th LFP segment in NREM state
    y = lfp['NREM'][input]
    #t = np.arange(x.size)/fs  # time points

    # Periodogram calculations
    fwake, pxxwake = signal.periodogram(x, fs=fs, scaling="spectrum", nfft = 100000) # 100000
    fnrem, pxxnrem = signal.periodogram(y, fs=fs, scaling="spectrum", nfft = 100000)

    plt.figure()
    plt.plot(fnrem[80:320],pxxnrem[80:320], label = "Delta")
    plt.plot(fnrem[321:640], pxxnrem[321:640], "r", label = "Theta")
    plt.plot(fnrem[641:960], pxxnrem[641:960], "g", label = "Alpha")
    plt.plot(fnrem[961:3040], pxxnrem[961:3040], "m", label = "Beta")
    plt.xlabel('Frequencies (Hz)')
    plt.ylabel('PSD')
    plt.title('Periodogram of Delta Waves NREM')
    plt.xlim(0,50)
    #plt.ylim(0, max(pxx)/100)
    plt.ylim(0,4000)
    plt.show()

    # Plot bandwidths in color
    plt.figure()
    plt.plot(fwake[80:320],pxxwake[80:320], label = "Delta")
    plt.plot(fwake[321:640], pxxwake[321:640], "r", label = "Theta")
    plt.plot(fwake[641:960], pxxwake[641:960], "g", label = "Alpha")
    plt.plot(fwake[961:3040], pxxwake[961:3040], "m", label = "Beta")
    plt.xlabel('Frequencies (Hz)')
    plt.ylabel('PSD')
    plt.title('Periodogram of Delta Waves WAKE')
    plt.xlim(0,50)
    #plt.ylim(0, max(pxx)/100)
    plt.ylim(0,4000)
    plt.show()

    # Median Delta Wave NREM
    sortedDel = np.sort(pxxnrem[80:320])
    medianDel = (sortedDel[120] + sortedDel[121]) / 2
    #print("Median Delta Value NREM:", round(medianDel,2))

    # Median Delta Wave WAKE
    sortedDel = np.sort(pxxwake[80:320])
    medianDel = (sortedDel[120] + sortedDel[121]) / 2
    #print("Median Delta Value WAKE:", round(medianDel, 2))

    # Median Theta Wave NREM
    sortedThet = np.sort(pxxnrem[641:960])
    medianThet = (sortedThet[159] + sortedThet[160]) / 2
    print("Median Theta Value NREM:", round(medianThet,2))

    # Median Theta Wave WAKE
    sortedThet = np.sort(pxxwake[641:960])
    medianThet = (sortedThet[159] + sortedThet[160]) / 2
    #median_theta.append(medianThet)
    print("Median Theta Value WAKE:", round(medianThet,2))

    # Median Alpha Wave NREM
    sortedThet = np.sort(pxxnrem[641:960])
    medianThet = (sortedThet[120] + sortedThet[121]) / 2
    #print("Median Alpha Value NREM:", round(medianThet,2))

    # Median Alpha Wave WAKE
    sortedThet = np.sort(pxxwake[641:960])
    medianThet = (sortedThet[120] + sortedThet[121]) / 2
    #print("Median Alpha Value WAKE:", round(medianThet,2))


#for i in range(5):
 #   colorPeriodogram(i)


# Make input data for graphs, SVM classification, and model
median_delta = []
median_theta= []
median_delta_wake = []
median_theta_wake = []
median_delta_nrem = []
median_theta_nrem = []

mean_delta_wake = []
mean_theta_wake = []
mean_delta_nrem = []
mean_theta_nrem = []

input_data = []
input_test = []

mean_spike = []
mean_spike_nrem = []
mean_spike_wake = []

max_delta_wake = []
max_delta_nrem = []

max_spike_wake = []
max_spike_nrem = []
# 400-960

for input in range(len(lfp['WAKE'])):
    x = lfp['WAKE'][input]
    wake, pxxwake = signal.periodogram(x, fs=fs, scaling="spectrum", nfft = 100000)
    sortedDel = np.sort(pxxwake[80:320])
    medianDel = (sortedDel[120] + sortedDel[121]) / 2
    meanDel = sum(sortedDel) / len(sortedDel)

    max_delta_wake.append(max(sortedDel)/max(pxxwake[400:960]))

    #SPIKE FEATURE
    meanSpike = sum(pxxwake[400:960]) / len(pxxwake[400:960])
    mean_spike.append(meanSpike)
    max_spike_wake.append(max(pxxwake[400:960])/max(pxxwake[400:960]))

    median_delta.append(medianDel)
    median_delta_wake.append(medianDel)
    mean_delta_wake.append(meanDel)

    # Median Thteta Wave WAKE
    sortedThet = np.sort(pxxwake[641:960])
    medianThet = (sortedThet[159] + sortedThet[160]) / 2
    meanThet = sum(sortedThet) / len(sortedThet)

    median_theta.append(medianThet)
    median_theta_wake.append(medianThet)
    mean_theta_wake.append(meanThet)

    input_data.append([medianDel, medianThet])
    input_test.append([medianDel, meanSpike])
    mean_spike_wake.append(meanSpike)

for input in range(len(lfp['NREM'])):
    x = lfp['NREM'][input]
    nrem, pxxnrem = signal.periodogram(x, fs=fs, scaling="spectrum", nfft = 100000)
    sortedDel = np.sort(pxxnrem[80:320])
    medianDel = (sortedDel[120] + sortedDel[121]) / 2
    meanDel = sum(sortedDel) / len(sortedDel)
    max_delta_nrem.append(max(sortedDel)/max(pxxnrem[400:960]))

    median_delta.append(medianDel)
    median_delta_nrem.append(medianDel)
    mean_delta_nrem.append(meanDel)

    # Median Thteta Wave NREM
    sortedThet = np.sort(pxxnrem[641:960])
    medianThet = (sortedThet[159] + sortedThet[160]) / 2
    meanThet = sum(sortedThet) / len(sortedThet)

    median_theta.append(medianThet)
    median_theta_nrem.append(medianThet)
    mean_theta_nrem.append(meanThet)

    input_data.append([medianDel, medianThet])

    #SPIKE FEATURE
    meanSpike = sum(pxxnrem[400:960]) / len(pxxnrem[400:960])
    mean_spike.append(meanSpike)
    mean_spike_nrem.append(meanSpike)
    input_test.append([medianDel, meanSpike])
    max_spike_nrem.append(max(pxxnrem[400:960])/max(pxxnrem[400:960]))

# Visualize the medians
plt.figure()
plt.title("Theta vs Delta Median Waves in NREM and WAKE Mice")
plt.scatter(median_delta_nrem, median_theta_nrem, color = 'b', marker='o', label="NREM");
plt.scatter(median_delta_wake, median_theta_wake, color = 'r', marker='o', label="WAKE");
plt.xlabel("Delta Wave Median Power")
plt.ylabel("Theta Wave Median Power")
plt.legend()
plt.show()

# Visualize the means
plt.figure()
plt.title("Theta vs Delta Mean Waves in NREM and WAKE Mice")
plt.scatter(mean_delta_nrem, mean_theta_nrem, color = 'b', marker='o', label="NREM");
plt.scatter(mean_delta_wake, mean_theta_wake, color = 'r', marker='o', label="WAKE");
plt.xlabel("Delta Wave Mean Power")
plt.ylabel("Theta Wave Mean Power")
plt.legend()
plt.show()

# Spike data visualization
plt.figure()
plt.title("Theta vs Mean Spike(5-12Hz) in NREM and WAKE Mice")
plt.scatter(median_delta_nrem, mean_spike_nrem, color = 'b', marker='o', label="NREM");
plt.scatter(median_delta_wake, mean_spike_wake, color = 'r', marker='o', label="WAKE");
plt.xlabel("Delta Wave Median Power")
plt.ylabel("Theta Wave Median Power")
plt.legend()
plt.show()

# Maximum spike and max delta
plt.figure()
plt.title("Theta vs Mean Spike(5-12Hz) in NREM and WAKE Mice")
plt.scatter(max_delta_nrem, max_spike_nrem, color = 'b', marker='o', label="NREM");
plt.scatter(max_delta_wake, max_spike_wake, color = 'r', marker='o', label="WAKE");
plt.xlabel("Delta Wave Median Power")
plt.ylabel("Theta Wave Median Power")
plt.legend()
plt.show()

# Make output data for SVM Classification
# Create output data set with 1 for Wake 0 for NREM
output = []
for i in range(len(lfp['WAKE'])):
    output.append(1)
for i in range(len(lfp['NREM'])):
    output.append(0)
#print(output)

# Split data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(median_delta, output, test_size=0.3, random_state = 1)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)

# SVM Classify
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train.reshape(-1,1), y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test.reshape(-1,1))

# Model Accuracy: how often is the classifier correct?
from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))



# MY MODEL
import tensorflow as tf
from tensorflow import keras
from keras import layers

# The test data classification works well
x = np.array([[1, 1], [1, 2], [5,5], [5,6]])
y = np.array([0,0,1,1])

# Real data
#input = np.array(input_data)
input = np.array(input_test)
output = np.array(output)

# Define the model
model = keras.Sequential()

# Define layers
model.add(keras.Input(shape=(2,)))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(),
          loss='binary_crossentropy',metrics=["accuracy"])

epochs = 1000
history = model.fit(
    input,
    output,
    epochs = epochs,
    callbacks=[tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.001)]
    )
