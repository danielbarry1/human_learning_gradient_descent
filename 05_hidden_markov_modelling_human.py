# this code performs the hidden markov modelling in experiment 2
# the supporting data can be found in the 
# "input_data_for_hidden_markov_modelling" 
# folder in "methods" on figshare

import numpy as np
from hmmlearn import hmm

conflict_trials=np.genfromtxt("group_data.csv",skip_header=0,delimiter=' ',usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19))

# recode participant probability ratings into binary decisions
# 50% trials are recoded into most recent decision
decisions=np.zeros((len(conflict_trials),20))
for subject in range(len(decisions[:,0])):
    for trial in range(len(decisions[0,:])):
        # start with the instance of the first trial being 50%
        if trial==0 and conflict_trials[subject,trial]==50:
            if conflict_trials[subject,trial+1]<50:
                decisions[subject,trial]=0
            if conflict_trials[subject,trial+1]>50:
                decisions[subject,trial]=1
        if trial!=0 and conflict_trials[subject,trial]==50:    
            if conflict_trials[subject,trial-1]<50:
                decisions[subject,trial]=0
            if conflict_trials[subject,trial-1]>50:
                decisions[subject,trial]=1    
        if conflict_trials[subject,trial]<50:
           decisions[subject,trial]=0
        if conflict_trials[subject,trial]>50:
           decisions[subject,trial]=1          

                          

decisions=decisions.astype("int")
np.savetxt("decisions.csv",decisions)

X = np.atleast_2d(np.ndarray.flatten(decisions)).T

# fit hmm 1000 times and choose one with best log_likeihood
for q in range(1000):
    print(q)
    model = hmm.MultinomialHMM(n_components=3,n_iter=1000)
    sequence_lengths=np.ones(len(decisions_1))*20
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

# save model results
np.savetxt("emission_probs.csv",best_model.emissionprob_)
np.savetxt("start_prob.csv",best_model.startprob_)
np.savetxt("trans_mat.csv",best_model.transmat_)

# save model itself
import pickle
with open("best_model.pkl", "wb") as file:
    pickle.dump(best_model, file)