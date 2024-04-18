from testing.testing_with_cnnresults import testing_toolnet, testing_endonet, testing_swinnet, \
    testing_toolnet_lstm, testing_swinnet_lstm, testing_endonet_lstm, \
    testing_toolnetsd, testing_endonetsd
from testing.testing_with_hmmcnn import testing_toolnet_hmm, testing_endonet_hmm, testing_swinnet_hmm, \
    testing_toolnetsd_hmm, testing_endonetsd_hmm
import os

### load CNN results
print('Load the CNN results of Cholec80')
root_dir = './results/results_paper/'
toolnet_results_path = os.path.join(root_dir, 'toolnet_results.npy')
endonet_results_path = os.path.join(root_dir, 'endonet_results.npy')
endonet_results_phase_path = os.path.join(root_dir, 'endonet_phase_results.npy')
swinnet_results_path = os.path.join(root_dir, 'swinnet_results.npy')

toolnetsd_results_path = os.path.join(root_dir, 'toolnetsd_results')
endonetsd_results_path = os.path.join(root_dir, 'endonetsd_results')

toolnet_lstm_results_path = os.path.join(root_dir, 'toolnet_lstm_results.npy')
endonet_lstm_results_path = os.path.join(root_dir, 'endonet_lstm_results.npy')
swinnet_lstm_results_path = os.path.join(root_dir, 'swinnet_lstm_results.npy')

### load HMM models
# cholec80
toolnet_hmm_model_dir = './pre-trained_model/hmm_toolnet'
endonet_hmm_model_dir = './pre-trained_model/hmm_endonet'
swinnet_hmm_model_dir = './pre-trained_model/hmm_swinnet'

toolnetsd_hmm_model_dir = './pre-trained_model/hmm_toolnetsd'
endonetsd_hmm_model_dir = './pre-trained_model/hmm_endonetsd'

### test plain-CNN
testing_toolnet(toolnet_results_path)
testing_toolnet_lstm(toolnet_lstm_results_path)

testing_endonet(endonet_results_path)
testing_endonet_lstm(endonet_lstm_results_path)

testing_swinnet(swinnet_results_path)
testing_swinnet_lstm(swinnet_lstm_results_path)

testing_toolnetsd(toolnetsd_results_path)
testing_endonetsd(endonetsd_results_path)

### test hmm-CNN
testing_toolnet_hmm(toolnet_results_path, toolnet_hmm_model_dir)
testing_endonet_hmm(endonet_results_path,endonet_results_phase_path,endonet_hmm_model_dir)
testing_swinnet_hmm(swinnet_results_path, swinnet_hmm_model_dir)

testing_toolnetsd_hmm(toolnetsd_results_path, toolnetsd_hmm_model_dir)
testing_endonetsd_hmm(endonetsd_results_path, toolnetsd_hmm_model_dir)