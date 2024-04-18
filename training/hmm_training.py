import numpy as np
from model.hmm import hmmmodel
import os
from sklearn.metrics import average_precision_score
tool_list = ['Grasper', 'Bipolar', 'Hook', 'Scissors', 'Clipper', 'Irrigator', 'Specimen Bag']
def singletool_hmm_training(cnnresultsdir, savedir):
    # these parameters are estimated by the CNN procedure from the training set
	threshold = [0.8, 0.5, 0.8, 0.5, 0.5, 0.5, 0.5]
    npy_file = np.load(cnnresultsdir, allow_pickle=True)
    for number in range(7):
        a, b, c = npy_file[0][number], npy_file[1][number], npy_file[2][number]
        pred, prob = hmmmodel(c, os.path.join(savedir,'toolnet'+str(number)+'.pkl'),threshold[number])
        print(tool_list[number],' AP:'+str(average_precision_score(b, prob)))

def toolandphase_hmm_training(cnnresultsdir,phaseresultsdir,savedir):
    # these parameters are estimated by the CNN procedure from the training set
	threshold = [0.8, 0.5, 0.9, 0.5, 0.5, 0.5, 0.7]
    npy_file = np.load(cnnresultsdir, allow_pickle=True)
    phase = np.load(phaseresultsdir,allow_pickle=True)[1]
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
            # print(sum(b_phase[i]))
            # now_pred, now_prob = my_hmm_t(a_phase[i],b_phase[i],c_phase[i],threshold_e[number])
            now_pred, now_prob = hmmmodel(c_phase[i],
                                          os.path.join(savedir, 'hmm_t'+str(number)+'p' + str(i) + '.pkl'),
                                          threshold[number])
            pred_phase.append(now_pred)
            prob_phase.append(now_prob)
            AP_phase.append(average_precision_score(b_phase[i], now_prob))
        AP = np.sum([AP_phase[i] * sum(b_phase[i]) for i in range(7)]) / np.sum(b)
        print(tool_list[number], ' AP:' + str(AP))




