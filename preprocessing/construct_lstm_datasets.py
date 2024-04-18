import os
def construct_lstm_all(rootdir):
    construct_lstm_single(rootdir, 'traintoolnet.txt', 'testtoolnet.txt')
    construct_lstm_single(rootdir, 'trainendonet.txt', 'testendonet.txt')
    construct_lstm_single(rootdir, 'trainswinnet.txt', 'testswinnet.txt')

def construct_lstm_single(rootdir, modelnametrain, modelnametest, LSTMLEN = 5):
    txt_dir = os.path.join(rootdir, 'traintest_txt')
    txt_path_traino = os.path.join(txt_dir, modelnametrain)
    txt_path_testo = os.path.join(txt_dir, modelnametest)
    txt_path_train = os.path.join(txt_dir, modelnametrain[:-4] + '_lstm.txt')
    txt_path_test = os.path.join(txt_dir, modelnametest[:-4] + '_lstm.txt')

    with open(txt_path_train, 'w') as f1:
        with open(txt_path_traino, 'r') as f2:
            rows = f2.readlines()
            for i in range(len(rows)-LSTMLEN):
                label = ','.join(rows[i + LSTMLEN - 1].split(',')[1:])
                img = ''
                for j in range(LSTMLEN):
                    img += rows[i + j].split(',')[0] + ','
                f1.write(img[:-1] + ';' + label)

    with open(txt_path_test, 'w') as f1:
        with open(txt_path_testo, 'r') as f2:
            rows = f2.readlines()
            for i in range(len(rows)-LSTMLEN):
                label = ','.join(rows[i + LSTMLEN - 1].split(',')[1:])
                img = ''
                for j in range(LSTMLEN):
                    img += rows[i + j].split(',')[0] + ','
                f1.write(img[:-1] + ';' + label)