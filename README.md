# HMM-Stabilized Deep Learning
This is a PyTorch implementation for paper "Efficient Surgical Tool Recognition via HMM-Stabilized Deep Learning".
## Reproduction
If you want to reproduce the mAP of our methods in the paper.
* You don't need to train models and download datasets, because we provide the prediction results of CNN model and pre-trained model of HMM.
* Simply run `python example.py`, and the results of all the methods of Cholec80 Tool Presence Detection in Table 3 are achieved.
## Datasets
* The published benchmark datasets cholec80 is available in the link below:
https://s3.unistra.fr/camma_public/datasets/cholec80/cholec80.tar.gz
## Training and Testing
* If you want to training the CNN model "ToolNet, ToolNetL, EndoNet, EndoNetL, SwinNet, SwinNetL", change the rootdir of datasets in the file `preprocessing/preprocessing.py`and run it.
* We provide the pre-trained model of the above six deep learning model, which is available in the link below:
https://cloud.tsinghua.edu.cn/f/9aa2ab6cfffa44058e36/
* The training and testing code is available in the folder `training` and `testing`, you only need to set the rootdir of training and testing txt which is produced in the preprocessing step and run it. 
* For example, if you want to training and testing SwinNetL on cholec80, run `python swinnet_lstm_training.py` and `python swinnet_lstm_testing.py`
* Early stopping is no needing in training, because we tune the specific parameter only according to the training set. In this way, we don't need addtional data from training set for eval in order to get the full size training set.
* If you want to use our pre-trained model and testing on cholec80, simply replace the folder `pre-trained_model` by the folder you have downloaded.
* If you have any question about the code, please contact `519837980@qq.com`.
## Online Materials
Please refer to `Online Materials.pdf`.

