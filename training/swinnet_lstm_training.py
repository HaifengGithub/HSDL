import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from utils.MyDataset import LSTMDataset
import os
import torch
from model.swinnet_lstm import swinnet_lstm

EPOCH = 5
BATCH_SIZE = 32
LR = 5e-5
CUDA= True

def training(traintxtdir, savedir):
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.75),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    my_model = swinnet_lstm()
    my_model = my_model.cuda()
    train_data = LSTMDataset(traintxtdir, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(my_model.parameters(), lr=LR)
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):
            # break
            b_x = b_x.cuda()
            output = my_model(b_x)
            b_y_0 = torch.LongTensor([int(j) for j in b_y[0]])
            b_y_0 = b_y_0.cuda()
            loss = loss_func(output[0], b_y_0)
            for i in range(1, 7):
                b_y_i = torch.LongTensor([int(j) for j in b_y[i]])
                b_y_i = b_y_i.cuda()
                loss += loss_func(output[i], b_y_i)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 20 == 0:
                print('epoch:'+str(epoch)+'|batch:'+str(step)+'|loss:'
                      +str(loss.item()))
    torch.save(my_model.state_dict(), savedir)

if __name__=='__main__':
    traintesttxt_dir = 'cholec80/traintest_txt'
    traintxt = os.path.join(traintesttxt_dir,'trainswinnet_lstm.txt')
    savedir = os.path.join('pre-trained_model/swinnet_lstm.pth')
    training(traintxt, savedir)