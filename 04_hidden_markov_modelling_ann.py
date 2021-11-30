import numpy as np
from hmmlearn import hmm

decisions=np.zeros((102,4000))
filename="cue_1_and_2.csv"
filename2="cue_3.csv"
weights_1_and_2=np.genfromtxt(filename)
weights_3=np.genfromtxt(filename2)
for n in range(102):
    weak_weights=weights_1_and_2[n,:]
    strong_weights=weights_3[n,:]
    for m in range(4000):
        if strong_weights[m]>weak_weights[m]:
            decisions[n,m]=1   
import os 
decisions=decisions.astype("int")
np.savetxt("decisions.csv",decisions)    
 

X = np.atleast_2d(np.ndarray.flatten(decisions)).T


for q in range(10):
    print(q)
    model = hmm.MultinomialHMM(n_components=3,n_iter=1000,verbose=True)
    sequence_lengths=np.ones(len(decisions))*4000
    sequence_lengths=sequence_lengths.astype("int")
    model.fit(X,lengths=sequence_lengths)
    log_likelihood=model.score(X,lengths=sequence_lengths)
    if q==0:
        best_ll=log_likelihood
        best_model=model
    else:
        if log_likelihood>best_ll:
            best_ll=log_likelihood
            best_model=model

np.savetxt("emission_probs.csv",best_model.emissionprob_)
np.savetxt("start_prob.csv",best_model.startprob_)
np.savetxt("trans_mat.csv",best_model.transmat_)

import pickle
with open("best_model.pkl", "wb") as file:
    pickle.dump(best_model, file)
