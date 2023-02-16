import torch
import torch.nn as nn
import itertools
from .base_model import BaseModel
from . import networks
from d2l import torch as d2l
import torchvision

class artvggAdaAttNModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--style_encoder_path', required=True, help='path to pretrained image encoder')
        parser.add_argument('--skip_connection_3', action='store_true',
                            help='if specified, add skip connection on ReLU-3')
        parser.add_argument('--shallow_layer', action='store_true',
                            help='if specified, also use features of shallow layers')
        parser.add_argument('--data_norm',action='store_true',
                            help='img normazation to imagenet')
        if is_train:
            parser.add_argument('--lambda_content', type=float, default=0., help='weight for L2 content loss')
            parser.add_argument('--lambda_global', type=float, default=10., help='weight for L2 style loss')
            parser.add_argument('--lambda_local', type=float, default=3.,
                                help='weight for attention weighted style loss')
            parser.add_argument('--lambda_edge', type=float, default=0.002,
                                help='weight for content edge loss')
            parser.add_argument('--lambda_c', type=float, default=50,
                                help='weight for content edge loss')
            parser.add_argument('--lambda_s', type=float, default=50,
                                help='weight for content edge loss')
            parser.add_argument('--lambda_d', type=float, default=50,
                                help='weight for content edge loss')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (1): ReLU(inplace=True) # relu1-1
        # (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (3): ReLU(inplace=True) # relu1-2
        # (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        # (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (6): ReLU(inplace=True) # relu2-1
        # (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (8): ReLU(inplace=True) # relu2-2
        # (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        # (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (11): ReLU(inplace=True) # relu3-1
        # (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (13): ReLU(inplace=True) # relu3-2
        # (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (15): ReLU(inplace=True) # relu3-3
        # (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        # (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (18): ReLU(inplace=True) # relu4-1
        # (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (20): ReLU(inplace=True) # relu4-2
        # (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (22): ReLU(inplace=True) # relu4-3
        # (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        # (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (25): ReLU(inplace=True) # relu5-1
        # (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (27): ReLU(inplace=True) # relu5-2
        # (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (29): ReLU(inplace=True) # relu5-3
        # (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        # content_image encoder
        content_image_encoder = torchvision.models.vgg19(pretrained=True).features
        print(content_image_encoder)
        # content_image_encoder.load_state_dict(torch.load(opt.image_encoder_path))
        enc_layers = list(content_image_encoder.children())
        enc_1 = nn.DataParallel(nn.Sequential(*enc_layers[:2]).to(opt.gpu_ids[0]), opt.gpu_ids)
        enc_2 = nn.DataParallel(nn.Sequential(*enc_layers[2:7]).to(opt.gpu_ids[0]), opt.gpu_ids)
        enc_3 = nn.DataParallel(nn.Sequential(*enc_layers[7:12]).to(opt.gpu_ids[0]), opt.gpu_ids)
        enc_4 = nn.DataParallel(nn.Sequential(*enc_layers[12:21]).to(opt.gpu_ids[0]), opt.gpu_ids)
        enc_5 = nn.DataParallel(nn.Sequential(*enc_layers[21:30]).to(opt.gpu_ids[0]), opt.gpu_ids)
        # enc_1 = nn.DataParallel(nn.Sequential(*enc_layers[:2]).to(opt.gpu_ids[0]), opt.gpu_ids)
        # enc_2 = nn.DataParallel(nn.Sequential(*enc_layers[2:7]).to(opt.gpu_ids[0]), opt.gpu_ids)
        # enc_3 = nn.DataParallel(nn.Sequential(*enc_layers[7:12]).to(opt.gpu_ids[0]), opt.gpu_ids)
        # enc_4 = nn.DataParallel(nn.Sequential(*enc_layers[12:19]).to(opt.gpu_ids[0]), opt.gpu_ids)
        # enc_5 = nn.DataParallel(nn.Sequential(*enc_layers[19:26]).to(opt.gpu_ids[0]), opt.gpu_ids)
        self.content_encoder_layers = [enc_1, enc_2, enc_3, enc_4, enc_5]
        for layer in self.content_encoder_layers:
            for param in layer.parameters():
                param.requires_grad = False

        # style image encoder
        style_image_encoder = torchvision.models.vgg19(pretrained=True)
        style_image_encoder.classifier._modules['6'] = nn.Linear(4096, 24)
        style_image_encoder.load_state_dict(torch.load(opt.style_encoder_path))
        style_image_encoder = style_image_encoder.features
        enc_layers = list(style_image_encoder.children())
        enc_1 = nn.DataParallel(nn.Sequential(*enc_layers[:2]).to(opt.gpu_ids[0]), opt.gpu_ids)
        enc_2 = nn.DataParallel(nn.Sequential(*enc_layers[2:7]).to(opt.gpu_ids[0]), opt.gpu_ids)
        enc_3 = nn.DataParallel(nn.Sequential(*enc_layers[7:12]).to(opt.gpu_ids[0]), opt.gpu_ids)
        enc_4 = nn.DataParallel(nn.Sequential(*enc_layers[12:21]).to(opt.gpu_ids[0]), opt.gpu_ids)
        enc_5 = nn.DataParallel(nn.Sequential(*enc_layers[21:30]).to(opt.gpu_ids[0]), opt.gpu_ids)
        self.style_encoder_layers = [enc_1, enc_2, enc_3, enc_4, enc_5]
        for layer in self.style_encoder_layers:
            for param in layer.parameters():
                param.requires_grad = False

        self.visual_names = ['c', 'cs', 's', 'c_org', 'cs_org', 's_org']#, 'edge_c', 'edge_cs', 'edge_s']
        self.model_names = ['decoder', 'transformer', 'SD', 'CD']
        parameters = []
        self.max_sample = 64 * 64
        if opt.skip_connection_3:
            adaattn_3 = networks.AdaAttN(in_planes=256, key_planes=256 + 128 + 64 if opt.shallow_layer else 256,
                                              max_sample=self.max_sample)
            self.net_adaattn_3 = networks.init_net(adaattn_3, opt.init_type, opt.init_gain, opt.gpu_ids)
            self.model_names.append('adaattn_3')
            parameters.append(self.net_adaattn_3.parameters())
        if opt.shallow_layer:
            channels = 512 + 256 + 128 + 64
        else:
            channels = 512
        transformer = networks.Transformer(
            in_planes=512, key_planes=channels, shallow_layer=opt.shallow_layer)
        decoder = networks.Decoder(opt.skip_connection_3)

        SD = torchvision.models.resnet18(pretrained=True)
        SD.fc = nn.Linear(512, 1)
        SD.sigmoid = nn.Sigmoid()

        CD = torchvision.models.resnet18(pretrained=True)
        CD.fc = nn.Linear(512, 1)
        CD.sigmoid = nn.Sigmoid()

        self.net_decoder = networks.init_net(decoder, opt.init_type, opt.init_gain, opt.gpu_ids)
        self.net_transformer = networks.init_net(transformer, opt.init_type, opt.init_gain, opt.gpu_ids)
        self.net_SD = networks.init_net(SD, opt.init_type, opt.init_gain, opt.gpu_ids)
        self.net_CD = networks.init_net(CD, opt.init_type, opt.init_gain, opt.gpu_ids)

        parameters.append(self.net_decoder.parameters())
        parameters.append(self.net_transformer.parameters())
        self.c = None
        self.cs = None
        self.s = None
        self.rgb_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.rgb_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
        self.c_org = None
        self.cs_org = None
        self.s_org = None
        self.edge_c = None
        self.edge_cs = None
        self.edge_s = None
        self.s_feats = None
        self.c_feats = None
        self.seed = 6666
        if self.isTrain:
            self.loss_names = ['content', 'global', 'local', 'SD', 'CD', 'D', 'G']#, 'edge']
            self.criterionMSE = torch.nn.MSELoss().to(self.device)
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.optimizer_g = torch.optim.Adam(itertools.chain(*parameters), lr=opt.lr)
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.net_SD.parameters(), self.net_CD.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_g)
            self.loss_global = torch.tensor(0., device=self.device)
            self.loss_local = torch.tensor(0., device=self.device)
            self.loss_content = torch.tensor(0., device=self.device)
            self.loss_edge = torch.tensor(0., device=self.device)
            self.loss_SD = torch.tensor(0., device=self.device)
            self.loss_CD = torch.tensor(0., device=self.device)
            self.loss_D = torch.tensor(0., device=self.device)
            self.loss_G = torch.tensor(0., device=self.device)

            self.conv_op_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False, device=self.device).requires_grad_(False)
            self.conv_op_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False, device=self.device).requires_grad_(False)
            sobel_x = torch.Tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]).to(self.device)
            sobel_y = torch.Tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]]).to(self.device)
            print(self.conv_op_y.weight.data.shape, sobel_x.shape)
            self.conv_op_x.weight.data = sobel_x
            self.conv_op_y.weight.data = sobel_y

    def set_input(self, input_dict):
        self.c = input_dict['c'].to(self.device)
        self.s = input_dict['s'].to(self.device)
        self.image_paths = input_dict['name']

    # def weight_norm(self, ec):
    #     # encoder normlazation
    #     for layer in ec:

    def postprocess(self, x):
        return torch.clamp(x.permute(0, 2, 3, 1) * self.rgb_std + self.rgb_mean, 0, 1).permute(0, 3, 1, 2)

    def encode_with_intermediate(self, input_img, type):
        results = [input_img]
        for i in range(5):
            if type == 'c':
                func = self.content_encoder_layers[i]
                results.append(func(results[-1]))
            elif type == 's':
                func = self.style_encoder_layers[i]
                results.append(func(results[-1]))
        return results[1:]

    @staticmethod
    def get_key(feats, last_layer_idx, need_shallow=True):
        if need_shallow and last_layer_idx > 0:
            results = []
            _, _, h, w = feats[last_layer_idx].shape
            for i in range(last_layer_idx):
                results.append(networks.mean_variance_norm(nn.functional.interpolate(feats[i], (h, w))))
            results.append(networks.mean_variance_norm(feats[last_layer_idx]))
            return torch.cat(results, dim=1)
        else:
            return networks.mean_variance_norm(feats[last_layer_idx])

    def forward(self):
        self.c_feats = self.encode_with_intermediate(self.c, 'c')
        self.s_feats = self.encode_with_intermediate(self.s, 's')
        if self.opt.skip_connection_3:
            c_adain_feat_3 = self.net_adaattn_3(self.c_feats[2], self.s_feats[2], self.get_key(self.c_feats, 2, self.opt.shallow_layer),
                                                   self.get_key(self.s_feats, 2, self.opt.shallow_layer), self.seed)
        else:
            c_adain_feat_3 = None
        cs = self.net_transformer(self.c_feats[3], self.s_feats[3], self.c_feats[4], self.s_feats[4],
                                  self.get_key(self.c_feats, 3, self.opt.shallow_layer),
                                  self.get_key(self.s_feats, 3, self.opt.shallow_layer),
                                  self.get_key(self.c_feats, 4, self.opt.shallow_layer),
                                  self.get_key(self.s_feats, 4, self.opt.shallow_layer), self.seed)
        self.cs = self.net_decoder(cs, c_adain_feat_3)
        self.c_org = self.postprocess(self.c)
        self.cs_org= self.postprocess(self.cs)
        self.s_org = self.postprocess(self.s)

    def compute_content_loss(self, stylized_feats):
        self.loss_content = torch.tensor(0., device=self.device)
        if self.opt.lambda_content > 0:
            for i in range(1, 5):
                self.loss_content += self.criterionMSE(networks.mean_variance_norm(stylized_feats[i]),
                                                       networks.mean_variance_norm(self.c_feats[i]))

    def compute_style_loss(self, stylized_feats):
        self.loss_global = torch.tensor(0., device=self.device)
        if self.opt.lambda_global > 0:
            for i in range(1, 5):
                s_feats_mean, s_feats_std = networks.calc_mean_std(self.s_feats[i])
                stylized_feats_mean, stylized_feats_std = networks.calc_mean_std(stylized_feats[i])
                self.loss_global += self.criterionMSE(
                    stylized_feats_mean, s_feats_mean) + self.criterionMSE(stylized_feats_std, s_feats_std)
        self.loss_local = torch.tensor(0., device=self.device)
        if self.opt.lambda_local > 0:
            for i in range(1, 5):
                c_key = self.get_key(self.c_feats, i, self.opt.shallow_layer)
                s_key = self.get_key(self.s_feats, i, self.opt.shallow_layer)
                s_value = self.s_feats[i]
                b, _, h_s, w_s = s_key.size()
                s_key = s_key.view(b, -1, h_s * w_s).contiguous()
                if h_s * w_s > self.max_sample:
                    torch.manual_seed(self.seed)
                    index = torch.randperm(h_s * w_s).to(self.device)[:self.max_sample]
                    s_key = s_key[:, :, index]
                    style_flat = s_value.view(b, -1, h_s * w_s)[:, :, index].transpose(1, 2).contiguous()
                else:
                    style_flat = s_value.view(b, -1, h_s * w_s).transpose(1, 2).contiguous()
                b, _, h_c, w_c = c_key.size()
                c_key = c_key.view(b, -1, h_c * w_c).permute(0, 2, 1).contiguous()
                attn = torch.bmm(c_key, s_key)
                # S: b, n_c, n_s
                attn = torch.softmax(attn, dim=-1)
                # mean: b, n_c, c
                mean = torch.bmm(attn, style_flat)
                # std: b, n_c, c
                std = torch.sqrt(torch.relu(torch.bmm(attn, style_flat ** 2) - mean ** 2))
                # mean, std: b, c, h, w
                mean = mean.view(b, h_c, w_c, -1).permute(0, 3, 1, 2).contiguous()
                std = std.view(b, h_c, w_c, -1).permute(0, 3, 1, 2).contiguous()
                self.loss_local += self.criterionMSE(stylized_feats[i], std * networks.mean_variance_norm(self.c_feats[i]) + mean)

    def rgb2gray(self, rgb_img):

        gray = 0.299*rgb_img[:,0,:,:]+0.587*rgb_img[:,1,:,:]+0.114*rgb_img[:,2,:,:]
        return gray.unsqueeze(1)

    def edge_detection(self, gray_img):
        # gx = self.conv_op_x(gray_img)
        # gy = self.conv_op_y(gray_img)
        return torch.abs(self.conv_op_x(gray_img))+torch.abs(self.conv_op_y(gray_img))

    def edge_losses(self):
        self.loss_edge = torch.tensor(0., device=self.device)
        # zhuanhundu
        self.edge_c = self.rgb2gray(self.c)
        self.edge_s = self.rgb2gray(self.s)
        self.edge_cs = self.rgb2gray(self.cs)

        # panbian
        self.edge_c = self.edge_detection(self.edge_c)
        self.edge_s = self.edge_detection(self.edge_s)
        self.edge_cs = self.edge_detection(self.edge_cs)

        # erzhihua
        a = torch.ones_like(self.c)
        b = torch.zeros_like(self.c)

        bin_c  = torch.where(self.edge_c >0.5, a, b)
        bin_cs = torch.where(self.edge_cs>0.3, a, b)

        self.loss_edge = self.criterionMSE(self.edge_c, self.edge_cs)



    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        self.loss_D = (loss_D_real + loss_D_fake) * 0.5 * self.opt.lambda_d
        self.loss_D.backward()
        return self.loss_D

    def backward_D(self):
        """Calculate the loss for generators G_A and G_B"""
        # train SD
        self.backward_D_basic(self.net_SD, self.s, self.cs)
        # train CD
        # self.backward_D_basic(self.net_CD, self.c, self.cs)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_s = self.opt.lambda_s
        lambda_c = self.opt.lambda_c

        # GAN loss SD(G(A))
        self.loss_SD = self.criterionGAN(self.net_SD(self.cs), True)
        # GAN loss CD(G(A))
        # self.loss_CD = self.criterionGAN(self.net_CD(self.cs), True)
        # combined loss and calculate gradients
        self.loss_G = self.loss_SD*lambda_s# + self.loss_CD*lambda_c
        # self.loss_G.backward()

    def optimize_G(self):
        # forward
        # G_A and G_B
        self.set_requires_grad([self.net_SD, self.net_CD], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_g.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.compute_losses()
        loss = self.loss_content + self.loss_global + self.loss_local + self.loss_G# + self.loss_edge
        loss.backward()
        self.optimizer_g.step()

    def optimize_D(self):
        # forward
        # D_A and D_B
        self.set_requires_grad([self.net_SD, self.net_CD], True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D()  # calculate gradients for D_A
        self.optimizer_D.step()  # update D_A and D_B's weights

    def compute_losses(self):
        content_stylized_feats = self.encode_with_intermediate(self.cs, 'c')
        style_stylized_feats = self.encode_with_intermediate(self.cs, 's')
        self.compute_content_loss(content_stylized_feats)
        self.compute_style_loss(style_stylized_feats)
        # self.edge_losses()
        self.loss_content = self.loss_content * self.opt.lambda_content
        self.loss_local = self.loss_local * self.opt.lambda_local
        self.loss_global = self.loss_global * self.opt.lambda_global
        # self.loss_edge = self.loss_edge * self.opt.lambda_edge

    def optimize_parameters(self):
        self.seed = int(torch.randint(10000000, (1,))[0])
        self.forward()
        self.optimize_G()
        # self.optimizer_g.zero_grad()
        self.optimize_D()


