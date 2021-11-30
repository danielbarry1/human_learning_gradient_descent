# this code performs the gradient descent in experiment 1
# the supporting data can be found in the 
# "input_data_for_linear_regression_and_gradient_descent" 
# folder in "methods" on figshare

# select GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
number_of_restarts=1
weights_filepath="weights.csv"
try:
    os.remove(weights_filepath)
except:
    pass
import numpy as np
# import input and target labels
input_labels=np.genfromtxt("data.csv",skip_header=1,delimiter=',',usecols=(0,1,2))
target_labels=np.genfromtxt("data.csv",skip_header=1,delimiter=',',usecols=(4))
import tensorflow
from tensorflow.keras import models
from tensorflow.keras import layers
import tensorflow.keras
from tensorflow.keras import initializers
q_model=models.Sequential()
# set all weights to zero
layer=layers.Dense(1,input_shape=(input_labels.shape[1],),kernel_initializer='zeros',bias_initializer='zeros')
q_model.add(layer)
from tensorflow.keras.utils import plot_model
plot_model(q_model, to_file='model.png')
# SGD parameters
epochs=4000
learning_rate = 0.01
from tensorflow.keras import optimizers
opt = optimizers.SGD(lr=learning_rate, 
                           momentum=False,
                           decay=decay,
                           nesterov=False)
q_model.compile(optimizer=opt, loss='mse', metrics=['mae'])
# callback to save the model weights at each epoch
class SaveWeights(tensorflow.keras.callbacks.Callback):
  def on_epoch_end(self,epoch,logs=None):
    weights=q_model.trainable_weights[0]
    weights_transposed=weights.numpy()
    weights_transposed=np.transpose(weights_transposed)
    weights_transposed=weights_transposed[0]
    def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
        with open(file_name, 'a+', newline='') as write_obj:
            from csv import writer
            # Create a writer object from csv module
            csv_writer = writer(write_obj)
            # Add contents of list as last row in the csv file
            csv_writer.writerow(list_of_elem)
    weights_filepath="/home/daniel/cue_experiment/tensorflow/weights.csv"
    append_list_as_row(weights_filepath,weights_transposed) 
    print('weights',weights_transposed)       
saveweightscallback=SaveWeights()
# fit weights
q_model.fit(input_labels,
            target_labels,
            epochs=epochs,
            batch_size=14,
            verbose=0,
            callbacks=[saveweightscallback])
cue_1=np.zeros((number_of_restarts,epochs))
cue_2=np.zeros((number_of_restarts,epochs))
cue_3=np.zeros((number_of_restarts,epochs))
for n in range(number_of_restarts):
    filename="/home/daniel/experiment/tensorflow/weights.csv"
    weights_values=np.genfromtxt(filename,skip_header=0,delimiter=',',usecols=(0,1,2))
    cue_1[n,:]=weights_values[:,0]
    cue_2[n,:]=weights_values[:,1]
    cue_3[n,:]=weights_values[:,2]
cue_1_and_2=cue_1+cue_2
cue_1=cue_1.squeeze()
cue_2=cue_2.squeeze()
cue_3=cue_3.squeeze()
cue_1_and_2=cue_1_and_2.squeeze()
import matplotlib.pyplot as plt
from scipy import stats
epochs = range(1, epochs + 1)
plt.figure()
plt.plot(epochs,cue_1_and_2,label='cue_1_&_2')
plt.plot(epochs,cue_3,label='cue_3')
plt.title('weights')
plt.legend()
plt.savefig('weights.png')
