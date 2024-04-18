import os
import numpy as np
import random
def construct_tinytoolnet(root_dir):
    number_list = []
    for i in range(8):
        for j in range(10):
            number_list.append(str(i) + str(j))
    number_list = number_list[1:]
    number_list.append('80')

    train_label = []
    for i in number_list[:40]:
        txt_file = os.path.join(root_dir, 'tool_annotations', 'video' + i + '-tool.txt')
        with open(txt_file, 'r') as f:
            rows = f.readlines()
            for row in rows[1:]:
                train_label.append(row.strip().split('\t')[1:])

    train_img = []
    from glob import glob
    for i in number_list[:40]:
        img_dir = os.path.join(root_dir, 'frames', 'video' + i)
        img = glob(img_dir + '\\*')
        train_img += img

    test_label = []
    for i in number_list[40:]:
        txt_file = os.path.join(root_dir, 'tool_annotations', 'video' + i + '-tool.txt')
        with open(txt_file, 'r') as f:
            rows = f.readlines()
            for row in rows[1:]:
                test_label.append(row.strip().split('\t')[1:])

    test_img = []
    from glob import glob
    for i in number_list[40:]:
        img_dir = os.path.join(root_dir, 'frames', 'video' + i)
        img = glob(img_dir + '\\*.png')
        test_img += img

    try:
        os.mkdir(os.path.join(root_dir,'traintest_txt'))
    except:
        pass
    txt_dir = os.path.join(root_dir,'traintest_txt')

    for i in range(7):
        train_txt = os.path.join(txt_dir, 'traintoolnetsd_' + str(i) + '.txt')
        train_label_s = np.array(train_label)[:,i]
        train_pos, train_neg = [], []
        for j,k in zip(train_label_s,train_img):
            if(j == '1'):
                train_pos.append(k)
            else:
                train_neg.append(k)
        # print(i, len(train_pos), len(train_neg))
        train_img_pos_now = random.sample(train_pos,1000)
        train_img_neg_now = random.sample(train_neg,1000)
        with open(train_txt, 'w') as f:
            for j in train_img_pos_now:
                f.write(j+','+'1'+'\n')
            for k in train_img_neg_now:
                f.write(k+','+'0'+'\n')

    for i in range(7):
        test_txt = os.path.join(txt_dir, 'testtoolnetsd_' + str(i) + '.txt')
        with open(test_txt, 'w') as f:
            for j,k in zip(test_img, test_label):
                f.write(j+','+k[i]+'\n')