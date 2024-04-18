import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
import numpy as np
from utils.MyDataset import LSTMDataset
import os
import torch
from model.endonet_lstm import endonet_lstm

def testing(testtxtdir, modeldir, savedir):
    transform_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    BATCH_SIZE = 32
    my_model = endonet_lstm()
    my_model.load_state_dict(torch.load(modeldir))
    my_model = my_model.cuda()
    my_model.eval()
    txt_path_test = testtxtdir
    test_data = LSTMDataset(txt_path_test, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

    tool_pred, tool_true, tool_prob = [[], [], [], [], [], [], []], \
                                      [[], [], [], [], [], [], []], [[], [], [], [], [], [], []]
    phase_pred, phase_true, phase_prob = [], [], []
    for step_test, (b_x_test, b_y_test) in enumerate(test_loader):
        b_x_test = b_x_test.cuda()
        test_output = my_model(b_x_test)
        for i in range(7):
            now_output = test_output[i]
            tool_true[i] += [int(j) for j in b_y_test[i]]
            tool_pred[i] += torch.max(now_output, 1)[1].cuda().data.squeeze().tolist()
            tool_prob[i] += [j[1] for j in nn.Softmax(dim=1)(now_output).cuda().tolist()]
        phase_true += [int(j) for j in b_y_test[7]]
        phase_pred += torch.max(test_output[7], 1)[1].cuda().data.squeeze().tolist()
        phase_prob += nn.Softmax(dim=1)(test_output[7]).cuda().tolist()
        print('batch:'+str(step_test))
    np.save(os.path.join(savedir,'endonet_lstm_results.npy'),
            np.array([tool_pred, tool_true, tool_prob]))
    np.save(os.path.join(savedir,'endonet_lstm_phase_results.npy'),
            np.array([phase_pred, phase_true, phase_prob]),
            allow_pickle=True)

if __name__=='__main__':
    traintesttxt_dir = 'cholec80/traintest_txt'
    testtxtdir = os.path.join(traintesttxt_dir,'testendonet_lstm.txt')
    modeldir = os.path.join('./pre-trained_model/endonet_lstm/endonet_lstm.pth')
    savedir = os.path.join('./results')
    testing(testtxtdir,modeldir,savedir)