import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
import os
import joblib

def hmmmodel(pred_prob1, save_path, threshold):
    origin_label = [int(j > threshold) for j in pred_prob1]
    states = [0, 1]
    n_states = len(states)

    observations = [0, 1]
    n_observations = len(observations)

    model = hmm.MultinomialHMM(n_components=n_states, n_iter=10000, tol=0.0001)
    X = np.array([list(origin_label)])
    model.fit(X)
    joblib.dump(model,save_path)

    prediction_new = model.predict(X.T)
    prob = model.predict_proba(X.T)

    if model.emissionprob_[0, 0] + model.transmat_[0, 0] > 1.6:
        now_prediction = prediction_new
        now_prob = [i[1] for i in prob]
    else:
        now_prediction = 1 - prediction_new
        now_prob = [i[0] for i in prob]

    return (now_prediction, now_prob)

