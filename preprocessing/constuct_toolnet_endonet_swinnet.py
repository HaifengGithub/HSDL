import os
def construct_toolnet_endonet_swinnet(root_dir):
    number_list = []
    for i in range(8):
        for j in range(10):
            number_list.append(str(i)+str(j))
    number_list = number_list[1:]
    number_list.append('80')

    train_label = []
    for i in number_list[:40]:
        txt_file = os.path.join(root_dir,'tool_annotations','video'+i+'-tool.txt')
        with open(txt_file, 'r') as f:
            rows = f.readlines()
            for row in rows[1:]:
                train_label.append(row.strip().split('\t')[1:])

    train_phase = []
    for i in number_list[:40]:
        txt_file = os.path.join(root_dir,'phase_annotations','video'+i+'-phase.txt')
        with open(txt_file, 'r') as f:
            rows = f.readlines()
            for row in rows[1:-24]:
                if int(row.strip().split('\t')[0]) % 25 == 0:
                    train_phase.append(row.strip().split('\t')[1:])

    train_img = []
    from glob import glob
    for i in number_list[:40]:
        img_dir = os.path.join(root_dir,'frames','video'+i)
        img = glob(img_dir+'/*')
        img = sorted(img, key=lambda x: int(x.split('_')[-1][:-4]))
        train_img += img

    test_label = []
    for i in number_list[40:]:
        txt_file = os.path.join(root_dir,'tool_annotations','video'+i+'-tool.txt')
        with open(txt_file, 'r') as f:
            rows = f.readlines()
            for row in rows[1:]:
                test_label.append(row.strip().split('\t')[1:])

    test_phase = []
    for i in number_list[40:]:
        txt_file = os.path.join(root_dir,'phase_annotations','video'+i+'-phase.txt')
        with open(txt_file, 'r') as f:
            rows = f.readlines()
            for row in rows[1:-24]:
                if int(row.strip().split('\t')[0]) % 25 == 0:
                    test_phase.append(row.strip().split('\t')[1:])

    test_img = []
    from glob import glob
    for i in number_list[40:]:
        img_dir = os.path.join(root_dir,'frames','video'+i)
        img = glob(img_dir+'/*')
        img = sorted(img, key=lambda x: int(x.split('_')[-1][:-4]))
        test_img += img

    phase_list = ['Preparation','CalotTriangleDissection','ClippingCutting','GallbladderDissection',
                  'GallbladderPackaging','CleaningCoagulation','GallbladderRetraction']

    try:
        os.mkdir(os.path.join(root_dir,'traintest_txt'))
    except:
        pass
    txt_dir = os.path.join(root_dir,'traintest_txt')
    trainall_txt = os.path.join(txt_dir,'traintoolnet.txt')
    testall_txt = os.path.join(txt_dir,'testtoolnet.txt')
    trainswin_txt = os.path.join(txt_dir,'trainswinnet.txt')
    testswin_txt = os.path.join(txt_dir,'testswinnet.txt')
    with open(trainall_txt, 'w') as f:
        for i,j in zip(train_img, train_label):
            f.write(i+','+','.join(j)+'\n')
    with open(testall_txt, 'w') as f:
        for i,j in zip(test_img, test_label):
            f.write(i+','+','.join(j)+'\n')

    with open(trainswin_txt, 'w') as f:
        for i,j in zip(train_img, train_label):
            f.write(i+','+','.join(j)+'\n')
    with open(testswin_txt, 'w') as f:
        for i,j in zip(test_img, test_label):
            f.write(i+','+','.join(j)+'\n')

    txt_dir = os.path.join(root_dir, 'traintest_txt')
    trainphase_txt = os.path.join(txt_dir,'trainendonet.txt')
    testphase_txt = os.path.join(txt_dir,'testendonet.txt')
    with open(trainphase_txt, 'w') as f:
        for i,j,k in zip(train_img, train_label, train_phase):
            f.write(i+','+','.join(j)+','+str(phase_list.index(k[0]))+'\n')
    with open(testphase_txt, 'w') as f:
        for i,j,k in zip(test_img, test_label, test_phase):
            f.write(i+','+','.join(j)+','+str(phase_list.index(k[0]))+'\n')
