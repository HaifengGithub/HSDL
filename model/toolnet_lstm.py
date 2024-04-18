import torch
import torch.utils.data
import torchvision.models as models
from torch import nn
from torch.nn import Sequential, Linear, ReLU, Dropout

class toolnet_lstm(nn.Module):
    def __init__(self):
        super(toolnet_lstm, self).__init__()
        resnet_pretrained = models.resnet18(pretrained=True)
        self.feature = torch.nn.Sequential(*list(resnet_pretrained.children())[:-1])
        self.lstm = torch.nn.Sequential(
            nn.LSTM(input_size=512, hidden_size=512, num_layers=2, batch_first=False, bidirectional=False)
        )
        self.fc1 = Sequential(
            nn.Linear(512, 512, bias=True),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(512, 2, bias=True),
        )
        self.fc2 = Sequential(
            nn.Linear(512, 512, bias=True),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(512, 2, bias=True),
        )
        self.fc3 = Sequential(
            nn.Linear(512, 512, bias=True),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(512, 2, bias=True),
        )
        self.fc4 = Sequential(
            nn.Linear(512, 512, bias=True),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(512, 2, bias=True),
        )
        self.fc5 = Sequential(
            nn.Linear(512, 512, bias=True),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(512, 2, bias=True),
        )
        self.fc6 = Sequential(
            nn.Linear(512, 512, bias=True),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(512, 2, bias=True),
        )
        self.fc7 = Sequential(
            nn.Linear(512, 512, bias=True),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(512, 2, bias=True),
        )

    def forward(self, x):
        x = x.transpose(0, 1)
        h1 = self.feature(x[0]).reshape([-1,512]).unsqueeze(0)
        # print(h1.shape)
        h2 = self.feature(x[1]).reshape([-1,512]).unsqueeze(0)
        h3 = self.feature(x[2]).reshape([-1,512]).unsqueeze(0)
        h4 = self.feature(x[3]).reshape([-1,512]).unsqueeze(0)
        h5 = self.feature(x[4]).reshape([-1,512]).unsqueeze(0)
        h = torch.cat((h1, h2, h3, h4, h5),0)
        # print(h.shape)
        x_, _ = self.lstm(h)
        # print(x_.shape)
        # x = torch.cat((x_[0],x_[-1]), 0)
        # x = x.reshape((-1, 512))
        x = x_[-1]
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x)
        x4 = self.fc4(x)
        x5 = self.fc5(x)
        x6 = self.fc6(x)
        x7 = self.fc7(x)
        # print(x1.shape)
        return (x1, x2, x3, x4, x5, x6, x7)