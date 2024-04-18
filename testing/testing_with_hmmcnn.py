import numpy as np
import os
import joblib
from sklearn.metrics import average_precision_score
tool_list = ['Grasper', 'Bipolar', 'Hook', 'Scissors', 'Clipper', 'Irrigator', 'Specimen Bag']
def my_hmm_test(pred_prob1, model_path, threshold):
    origin_label = [int(j > threshold) for j in pred_prob1]
    model = joblib.load(model_path)
    X = np.array([list(origin_label)])
    prediction_new = model.predict(X.T)
    prob = model.predict_proba(X.T)
	# avoid the common transfer from different label
    if model.emissionprob_[0, 0] + model.transmat_[0, 0] > 1.6:
        now_prediction = prediction_new
        now_prob = [i[1] for i in prob]
    else:
        now_prediction = 1 - prediction_new
        now_prob = [i[0] for i in prob]
    return (now_prediction, now_prob)

def testing_toolnet_hmm(cnnresultsdir, modeldir):
    print('###Start Testing ToolNet_H###')
	# these parameters are estimated from the training set
    threshold = [0.95, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
    AP_list = []
    npy_file = np.load(cnnresultsdir)
    for number in range(7):
        a, b, c = npy_file[0][number], npy_file[1][number], npy_file[2][number]
        pred, prob = my_hmm_test(c, os.path.join(modeldir, str(number) + '.pkl'), threshold[number])
        AP = np.around(average_precision_score(b, prob), 3)
        AP_list.append(AP)
        print(tool_list[number], '|AP:' + str(AP))
    print('mAP:' + str(np.around(np.mean(AP_list), 3)))

def testing_endonet_hmm(cnnresultsdir, phaseresultsdir, modeldir):
    print('###Start Testing Endonet_H###')
	# these parameters are estimated from the training set
    threshold = [0.8, 0.5, 0.9, 0.5, 0.5, 0.5, 0.7]
    npy_file = np.load(cnnresultsdir, allow_pickle=True)
    phase = np.load(phaseresultsdir, allow_pickle=True)[1]
    AP_list = []
    for number in range(7):
        a, b, c = npy_file[0][number], npy_file[1][number], npy_file[2][number]
        index_phase = [[], [], [], [], [], [], []]
        a_phase, b_phase, c_phase = [[], [], [], [], [], [], []], \
                                    [[], [], [], [], [], [], []], [[], [], [], [], [], [], []]
        num = 0
        for a_, b_, c_, p_ in zip(a, b, c, phase):
            a_phase[p_].append(int(a_))
            b_phase[p_].append(int(b_))
            c_phase[p_].append(c_)
            index_phase[p_].append(num)
            num += 1
        pred_phase, prob_phase = [], []
        AP_phase = []
        for i in range(7):
            if sum(b_phase[i]) == 0:
                pred_phase.append(np.zeros(len(b_phase[i])))
                prob_phase.append(np.zeros(len(b_phase[i])))
                AP_phase.append(0)
                continue
            now_pred, now_prob = my_hmm_test(c_phase[i],
                                               os.path.join(modeldir, 't'+str(number)+'p' + str(i) + '.pkl'),
                                               threshold[number])
            pred_phase.append(now_pred)
            prob_phase.append(now_prob)
            AP_phase.append(average_precision_score(b_phase[i], now_prob))
        AP = np.around(np.sum([AP_phase[i] * sum(b_phase[i]) for i in range(7)]) / np.sum(b),3)
        AP_list.append(AP)
        print(tool_list[number], '|AP:' + str(AP))
    print('mAP:' + str(np.around(np.mean(AP_list), 3)))

def testing_swinnet_hmm(cnnresultsdir, modeldir):
    print('###Start Testing SwinNet_H###')
	# these parameters are estimated from the training set
    threshold = [0.9,0.9,0.9,0.9,0.9,0.9,0.9]
    AP_list = []
    npy_file = np.load(cnnresultsdir)
    for number in range(7):
        a, b, c = npy_file[0][number], npy_file[1][number], npy_file[2][number]
        pred, prob = my_hmm_test(c, os.path.join(modeldir, str(number) + '.pkl'), threshold[number])
        AP = np.around(average_precision_score(b, prob), 3)
        AP_list.append(AP)
        print(tool_list[number], '|AP:' + str(AP))
    print('mAP:' + str(np.around(np.mean(AP_list), 3)))

def testing_toolnetsd_hmm(cnnresultsdir, modeldir):
    print('###Start Testing ToolNet_H_sd###')
	# these parameters are estimated from the training set
    threshold = [0.8,0.99,0.8,0.8,0.95,0.9,0.995]
    AP_list = []
    for number in range(7):
        npy_file = np.load(os.path.join(cnnresultsdir,str(number)+'.npy'))
        a, b, c = npy_file[0], npy_file[1], npy_file[2]
        pred, prob = my_hmm_test(c, os.path.join(modeldir, str(number) + '.pkl'), threshold[number])
        AP = np.around(average_precision_score(b, prob), 3)
        AP_list.append(AP)
        print(tool_list[number], '|AP:' + str(AP))
    print('mAP:' + str(np.around(np.mean(AP_list), 3)))

def testing_endonetsd_hmm(cnnresultsdir, modeldir):
    print('###Start Testing EndoNet_H_sd###')
	# these parameters are estimated from the training set
    threshold = [0.9,0.9,0.9,0.9,0.9,0.995,0.9]
    AP_list = []
    for number in range(7):
        npy_file = np.load(os.path.join(cnnresultsdir,str(number)+'.npy'))
        a, b, c = npy_file[0], npy_file[1], npy_file[2]
        pred, prob = my_hmm_test(c, os.path.join(modeldir, str(number) + '.pkl'), threshold[number])
        AP = np.around(average_precision_score(b, prob), 3)
        AP_list.append(AP)
        print(tool_list[number], '|AP:' + str(AP))
    print('mAP:' + str(np.around(np.mean(AP_list), 3)))

