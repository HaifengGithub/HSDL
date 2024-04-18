################### construct toolnet & endonet txt ####################
from preprocessing.constuct_toolnet_endonet_swinnet import construct_toolnet_endonet_swinnet
from preprocessing.construct_lstm_datasets import construct_lstm_all
data_root_dir = 'E:/Project_Dataset/cholec80'
construct_toolnet_endonet_swinnet(data_root_dir)
construct_lstm_all(data_root_dir)

# from preprocessing.construct_tinytoolnet import construct_toolnetsd
# construct_toolnetsd('E:/Project_Dataset/cholec80')
