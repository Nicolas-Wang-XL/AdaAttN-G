import torch
import torch.nn as nn
import itertools
from .base_model import BaseModel
from . import networks
from d2l import torch as d2l
import os
import torchvision


class ArtVGGModel(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.net_artvgg = torchvision.models.vgg16(pretrained=True).to(self.device)
        self.net_artvgg.classifier._modules['6'] = nn.Linear(4096, 27).to(self.device)
        # nn.Sequential(
        #     nn.Conv2d(3, 3, (1, 1)),
        #     nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     nn.ReLU(inplace=True),  # relu1-1
        #     nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     nn.ReLU(inplace=True),  # relu1-2
        #     nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        #     nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     nn.ReLU(inplace=True),  # relu2-1
        #     nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     nn.ReLU(inplace=True),  # relu2-2
        #     nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        #     nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     nn.ReLU(inplace=True),  # relu3-1
        #     nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     nn.ReLU(inplace=True),  # relu3-2
        #     nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     nn.ReLU(inplace=True),  # relu3-3
        #     nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     nn.ReLU(inplace=True),  # relu3-4
        #     nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        #     nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     nn.ReLU(inplace=True),  # relu4-1, this is the last layer used
        #     nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     nn.ReLU(inplace=True),  # relu4-2
        #     nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     nn.ReLU(inplace=True),  # relu4-3
        #     nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     nn.ReLU(inplace=True),  # relu4-4
        #     nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        #     nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     nn.ReLU(inplace=True),  # relu5-1
        #     nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     nn.ReLU(inplace=True),  # relu5-2
        #     nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     nn.ReLU(inplace=True),  # relu5-3
        #     nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     nn.ReLU(inplace=True),  # relu5-4
        #     nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        #
        #     nn.AdaptiveAvgPool2d(output_size=(7, 7)),
        #     nn.Flatten(),
        #     nn.Linear(in_features=25088, out_features=4096, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.5, inplace=False),
        #     nn.Linear(in_features=4096, out_features=4096, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.5, inplace=False),
        #     nn.Linear(in_features=4096, out_features=4, bias=True)
        # )

        # parameters = []
        # self.net = networks.init_net(net, opt.init_type, opt.init_gain, opt.gpu_ids)
        # parameters.append(self.net.parameters())
        self.x = None
        self.y = None
        self.y_hat = None
        self.loss_total = None
        self.optimizers = []
        self.visual_names = ['x']
        self.model_names = ['artvgg']

        if self.isTrain:
            self.loss_names = ['total']
            self.loss = nn.CrossEntropyLoss().to(self.device)
            self.optimizer_g = torch.optim.SGD(self.net_artvgg.parameters(), lr=opt.lr)#torch.optim.Adam(itertools.chain(self.net.parameters()), lr=opt.lr)
            self.optimizers.append(self.optimizer_g)

    def set_input(self, input_dict):
        self.x = input_dict['x'].to(self.device)
        self.y = input_dict['y'].to(self.device)
        # print(self.y, self.x.shape)
        # self.image_paths = input_dict['name']

    def forward(self):
        self.y_hat = self.net_artvgg(self.x)
        # print(self.y_hat)

    def compute_losses(self):
        self.loss_total = self.loss(self.y_hat, self.y)
        # print(self.loss_total)

    def optimize_parameters(self):
        self.seed = int(torch.randint(10000000, (1,))[0])
        # os.system('pause')
        self.forward()
        self.optimizer_g.zero_grad()
        self.compute_losses()
        self.loss_total.backward()
        self.optimizer_g.step()



