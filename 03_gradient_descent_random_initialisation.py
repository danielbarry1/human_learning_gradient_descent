# select GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
number_of_restarts=102
import numpy as np
input_labels=np.genfromtxt("data.csv",skip_header=1,delimiter=',',usecols=(0,1,2))
target_labels=np.genfromtxt("data.csv",skip_header=1,delimiter=',',usecols=(4))
import tensorflow
from tensorflow.keras import models
from tensorflow.keras import layers
import tensorflow.keras

for iteration in range(number_of_restarts):
    from tensorflow.keras import initializers
    q_model=models.Sequential()
    layer=layers.Dense(1,input_shape=(input_labels.shape[1],))
    q_model.add(layer)
    from tensorflow.keras.utils import plot_model
    learning_rate = 0.01
    from tensorflow.keras import optimizers
    opt = optimizers.SGD(lr=learning_rate, 
                               momentum=False,
                               decay=decay,
                               nesterov=False)
    q_model.compile(optimizer=opt, loss='mse', metrics=['mae'])
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
        weights_filepath="/store20-2/daniel/cue_experiment/tensorflow_momentum_group/weights"+str(iteration)+".csv"
        append_list_as_row(weights_filepath,weights_transposed) 
        print('weights',weights_transposed)       
    saveweightscallback=SaveWeights()
    q_model.fit(input_labels,
                target_labels,
                epochs=4000,
                batch_size=14,
                verbose=0,
                callbacks=[saveweightscallback])
cue_1=np.zeros((number_of_restarts,4000))
cue_2=np.zeros((number_of_restarts,4000))
cue_3=np.zeros((number_of_restarts,4000))
for n in range(number_of_restarts):
    filename="/store20-2/daniel/cue_experiment/tensorflow_momentum_group/weights"+str(n)+".csv"
    weights_values=np.genfromtxt(filename,skip_header=0,delimiter=',',usecols=(0,1,2))
    cue_1[n,:]=weights_values[:,0]
    cue_2[n,:]=weights_values[:,1]
    cue_3[n,:]=weights_values[:,2]

cue_1_noise = np.random.normal(0, .1, cue_1.shape)
cue_2_noise = np.random.normal(0, .1, cue_2.shape)
cue_3_noise = np.random.normal(0, .1, cue_3.shape)
cue_1 = cue_1 + cue_1_noise
cue_2 = cue_2 + cue_2_noise
cue_3 = cue_3 + cue_3_noise

mean_cue_1=np.mean(cue_1,axis=0)
mean_cue_2=np.mean(cue_2,axis=0)
mean_cue_3=np.mean(cue_3,axis=0)
cue_1_and_2=cue_1+cue_2
mean_cue_1_and_2=np.mean(cue_1_and_2,axis=0)
import matplotlib.pyplot as plt
from scipy import stats
std_cue_1=stats.sem(cue_1)
std_cue_2=stats.sem(cue_2)
std_cue_3=stats.sem(cue_3)
std_cue_1_and_2=stats.sem(cue_1_and_2)
epochs = range(1, len(mean_cue_1) + 1)
plt.figure()
plt.errorbar(epochs,mean_cue_1_and_2,yerr=std_cue_1_and_2,ecolor='red',elinewidth=2,label='cue_1_&_2')
plt.errorbar(epochs,mean_cue_3,yerr=std_cue_3,ecolor='green',elinewidth=2,label='cue_3')
plt.title('cue weights group')
plt.legend()
plt.savefig('cue weights group.png')
np.savetxt("mean_cue_1_and_2.csv",mean_cue_1_and_2)
np.savetxt("mean_cue_3.csv",mean_cue_3)
np.savetxt("std_cue_1_and_2.csv",std_cue_1_and_2)
np.savetxt("std_cue_3.csv",std_cue_3)