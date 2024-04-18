import numpy as np
import os
from sklearn.metrics import average_precision_score
tool_list = ['Grasper', 'Bipolar', 'Hook', 'Scissors', 'Clipper', 'Irrigator', 'Specimen Bag']
def testing_toolnet(results_path):
    print('###Start Testing ToolNet###')
    npyfile = np.load(results_path,allow_pickle=True)
    tool_pred,tool_true,tool_prob = npyfile[0],npyfile[1],npyfile[2]
    AP_list = []
    for i in range(7):
        now_AP = np.around(average_precision_score(tool_true[i], tool_prob[i]), 3)
        AP_list.append(now_AP)
        print(tool_list[i]+'|AP:'+str(now_AP))
    print('mAP:' + str(np.around(np.mean(AP_list), 3)))

def testing_endonet(results_path):
    print('###Start Testing EndoNet###')
    npyfile = np.load(results_path,allow_pickle=True)
    tool_pred,tool_true,tool_prob = npyfile[0],npyfile[1],npyfile[2]
    AP_list = []
    for i in range(7):
        now_AP = np.around(average_precision_score(tool_true[i], tool_prob[i]), 3)
        AP_list.append(now_AP)
        print(tool_list[i]+'|AP:'+str(now_AP))
    print('mAP:' + str(np.around(np.mean(AP_list), 3)))

def testing_swinnet(results_path):
    print('###Start Testing SwinNet###')
    npyfile = np.load(results_path,allow_pickle=True)
    tool_pred,tool_true,tool_prob = npyfile[0],npyfile[1],npyfile[2]
    AP_list = []
    for i in range(7):
        now_AP = np.around(average_precision_score(tool_true[i], tool_prob[i]), 3)
        AP_list.append(now_AP)
        print(tool_list[i]+'|AP:'+str(now_AP))
    print('mAP:' + str(np.around(np.mean(AP_list), 3)))

def testing_toolnet_lstm(results_path):
    print('###Start Testing ToolNet_L###')
    npyfile = np.load(results_path,allow_pickle=True)
    tool_pred,tool_true,tool_prob = npyfile[0],npyfile[1],npyfile[2]
    AP_list = []
    for i in range(7):
        now_AP = np.around(average_precision_score(tool_true[i], tool_prob[i]), 3)
        AP_list.append(now_AP)
        print(tool_list[i]+'|AP:'+str(now_AP))
    print('mAP:' + str(np.around(np.mean(AP_list), 3)))

def testing_endonet_lstm(results_path):
    print('###Start Testing Endonet_L###')
    npyfile = np.load(results_path,allow_pickle=True)
    tool_pred,tool_true,tool_prob = npyfile[0],npyfile[1],npyfile[2]
    AP_list = []
    for i in range(7):
        now_AP = np.around(average_precision_score(tool_true[i], tool_prob[i]), 3)
        AP_list.append(now_AP)
        print(tool_list[i]+'|AP:'+str(now_AP))
    print('mAP:' + str(np.around(np.mean(AP_list), 3)))

def testing_swinnet_lstm(results_path):
    print('###Start Testing SwinNet_L###')
    npyfile = np.load(results_path,allow_pickle=True)
    tool_pred,tool_true,tool_prob = npyfile[0],npyfile[1],npyfile[2]
    AP_list = []
    for i in range(7):
        now_AP = np.around(average_precision_score(tool_true[i], tool_prob[i]), 3)
        AP_list.append(now_AP)
        print(tool_list[i]+'|AP:'+str(now_AP))
    print('mAP:' + str(np.around(np.mean(AP_list), 3)))

def testing_toolnetsd(results_path):
    print('###Start Testing ToolNet_sd###')
    AP_list = []
    for i in range(7):
        npyfile = np.load(os.path.join(results_path, str(i)+'.npy'))
        tool_pred,tool_true,tool_prob = npyfile[0], npyfile[1], npyfile[2]
        now_AP = np.around(average_precision_score(tool_true, tool_prob), 3)
        AP_list.append(now_AP)
        print(tool_list[i]+'|AP:'+str(now_AP))
    print('mAP:'+str(np.around(np.mean(AP_list), 3)))

def testing_endonetsd(results_path):
    print('###Start Testing EndoNet_sd###')
    AP_list = []
    for i in range(7):
        npyfile = np.load(os.path.join(results_path, str(i)+'.npy'))
        tool_pred,tool_true,tool_prob = npyfile[0], npyfile[1], npyfile[2]
        now_AP = np.around(average_precision_score(tool_true, tool_prob), 3)
        AP_list.append(now_AP)
        print(tool_list[i]+'|AP:'+str(now_AP))
    print('mAP:'+str(np.around(np.mean(AP_list), 3)))