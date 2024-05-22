import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import sys
import pdb
import monai
from monai.networks.nets import ViT
from torchvision.ops.misc import MLP, Permute

import vn_layers
from networks import SwinClassifier


class EncoderBlock(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, kernel_size=3, conv_act='leaky_relu', dropout=0, num_conv=2):
        super(EncoderBlock, self).__init__()
        if conv_act == 'relu':
            conv_act_layer = nn.ReLU(inplace=True)
        elif conv_act == 'leaky_relu':
            conv_act_layer = nn.LeakyReLU(0.2, inplace=True)
            #conv_act_layer = vn_layers.VNLeakyReLU(out_num_ch, share_nonlinearity=True, negative_slope=0.2)
        else:
            raise ValueError('No implementation of ', conv_act)

        if num_conv == 1:
            self.conv = nn.Sequential(
                            nn.Conv3d(in_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
                            nn.BatchNorm3d(out_num_ch),
                            conv_act_layer,
                            nn.Dropout3d(dropout),
                            nn.MaxPool3d(2))
        elif num_conv == 2:
            self.conv = nn.Sequential(
                            nn.Conv3d(in_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
                            nn.BatchNorm3d(out_num_ch),
                            conv_act_layer,
                            nn.Dropout3d(dropout),
                            nn.Conv3d(out_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
                            nn.BatchNorm3d(out_num_ch),
                            conv_act_layer,
                            nn.Dropout3d(dropout),
                            nn.MaxPool3d(2))
        # abstract model size to num_conv
        elif num_conv == 3:
            self.conv = nn.Sequential(
                            nn.Conv3d(in_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
                            nn.BatchNorm3d(out_num_ch),
                            conv_act_layer,

                            nn.Dropout3d(dropout),
                            nn.Conv3d(out_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
                            nn.BatchNorm3d(out_num_ch),
                            conv_act_layer,

                            nn.Dropout3d(dropout),
                            nn.Conv3d(out_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
                            nn.BatchNorm3d(out_num_ch),
                            conv_act_layer,
                            
                            nn.Dropout3d(dropout),
                            nn.MaxPool3d(2))
        elif num_conv == 4:
            self.conv = nn.Sequential(
                            nn.Conv3d(in_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
                            nn.BatchNorm3d(out_num_ch),
                            conv_act_layer,

                            nn.Dropout3d(dropout),
                            nn.Conv3d(out_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
                            nn.BatchNorm3d(out_num_ch),
                            conv_act_layer,

                            nn.Dropout3d(dropout),
                            nn.Conv3d(out_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
                            nn.BatchNorm3d(out_num_ch),
                            conv_act_layer,

                            nn.Dropout3d(dropout),
                            nn.Conv3d(out_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
                            nn.BatchNorm3d(out_num_ch),
                            conv_act_layer,
                            
                            nn.Dropout3d(dropout),
                            nn.MaxPool3d(2))
        elif num_conv == 6:
            self.conv = nn.Sequential(
                            nn.Conv3d(in_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
                            nn.BatchNorm3d(out_num_ch),
                            conv_act_layer,

                            nn.Dropout3d(dropout),
                            nn.Conv3d(out_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
                            nn.BatchNorm3d(out_num_ch),
                            conv_act_layer,

                            nn.Dropout3d(dropout),
                            nn.Conv3d(out_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
                            nn.BatchNorm3d(out_num_ch),
                            conv_act_layer,

                            nn.Dropout3d(dropout),
                            nn.Conv3d(out_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
                            nn.BatchNorm3d(out_num_ch),
                            conv_act_layer,

                            nn.Dropout3d(dropout),
                            nn.Conv3d(out_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
                            nn.BatchNorm3d(out_num_ch),
                            conv_act_layer,

                            nn.Dropout3d(dropout),
                            nn.Conv3d(out_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
                            nn.BatchNorm3d(out_num_ch),
                            conv_act_layer,
                            
                            nn.Dropout3d(dropout),
                            nn.MaxPool3d(2))

        else:
            raise ValueError('Number of conv can only be 1 or 2')

        self.init_model()

    def init_model(self):
        for layer in self.conv.children():
            if isinstance(layer, nn.Conv3d):
                for name, weight in layer.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(weight)
                    if 'bias' in name:
                        nn.init.constant_(weight, 0.0)

    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, kernel_size=3, conv_act='leaky_relu', dropout=0, num_conv=2):
        super(DecoderBlock, self).__init__()
        if conv_act == 'relu':
            conv_act_layer = nn.ReLU(inplace=True)
        elif conv_act == 'leaky_relu':
            conv_act_layer = vn_layers.VNLeakyReLU(out_num_ch, share_nonlinearity=True, negative_slope=0.2)
        else:
            raise ValueError('No implementation of ', conv_act)

        if num_conv == 1:
            self.conv = nn.Sequential(
                            nn.Conv3d(in_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
                            nn.BatchNorm3d(out_num_ch),
                            conv_act_layer,
                            nn.Dropout3d(dropout),
                            nn.Upsample(scale_factor=(2,2,2), mode='trilinear', align_corners=True))
        elif num_conv == 2:
            self.conv = nn.Sequential(
                            nn.Conv3d(in_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
                            nn.BatchNorm3d(out_num_ch),
                            conv_act_layer,
                            nn.Dropout3d(dropout),
                            nn.Conv3d(out_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
                            nn.BatchNorm3d(out_num_ch),
                            conv_act_layer,
                            nn.Dropout3d(dropout),
                            nn.Upsample(scale_factor=(2,2,2), mode='trilinear', align_corners=True))
        else:
            raise ValueError('Number of conv can only be 1 or 2')

        self.init_model()

    def init_model(self):
        for layer in self.conv.children():
            if isinstance(layer, nn.Conv3d):
                for name, weight in layer.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(weight)
                    if 'bias' in name:
                        nn.init.constant_(weight, 0.0)

    def forward(self, x):
        return self.conv(x)


class Encoder_var(nn.Module):
    def __init__(self, in_num_ch=1, img_size=(64,64,64), inter_num_ch=16, kernel_size=3, conv_act='leaky_relu', num_conv=2):
        super(Encoder_var, self).__init__()

        self.conv1 = EncoderBlock(in_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
        self.conv2 = EncoderBlock(inter_num_ch, 2*inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
        self.conv3 = EncoderBlock(2*inter_num_ch, 4*inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
        self.conv4 = EncoderBlock(4*inter_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
        self.mu = nn.Conv3d(inter_num_ch, inter_num_ch, kernel_size=kernel_size, padding=1)
        self.log_var = nn.Conv3d(inter_num_ch, inter_num_ch, kernel_size=kernel_size, padding=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        mu = self.mu(conv4)
        log_var = self.log_var(conv4)
        # (16,4,4,4)
        return mu.view(x.shape[0], -1), log_var.view(x.shape[0], -1)

class EncoderVN(nn.Module):
    def __init__(self, in_num_ch=1, img_size=(64,64,64), inter_num_ch=16, kernel_size=3, conv_act='leaky_relu', num_conv=2, dropout=False, normalize_output=False):
        super(EncoderVN, self).__init__()
        self.normalize_output = normalize_output

        if dropout:
            self.conv1 = EncoderBlock(in_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
            self.conv2 = EncoderBlock(inter_num_ch, 2*inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0.1, num_conv=num_conv)
            self.conv3 = EncoderBlock(2*inter_num_ch, 4*inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0.2, num_conv=num_conv)
            self.conv4 = EncoderBlock(4*inter_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
        else:
            self.conv1 = EncoderBlock(in_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
            self.conv2 = EncoderBlock(inter_num_ch, 2*inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
            self.conv3 = EncoderBlock(2*inter_num_ch, 4*inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
            self.conv4 = EncoderBlock(4*inter_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        # normalize the output so that scaling does not affect the loss
        if self.normalize_output:
            conv4 = conv4 / torch.norm(conv4, p='fro') # normalize such that L2 norm of each matrix is 1
        return conv4

class Encoder_Var(nn.Module):
    def __init__(self, in_num_ch=1, img_size=(64,64,64), inter_num_ch=16, kernel_size=3, conv_act='leaky_relu', num_conv=2):
        super(Encoder_Var, self).__init__()

        self.conv1 = EncoderBlock(in_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
        self.conv2 = EncoderBlock(inter_num_ch, 2*inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
        self.conv3 = EncoderBlock(2*inter_num_ch, 4*inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
        self.conv4_mean = EncoderBlock(4*inter_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
        self.conv4_logvar = EncoderBlock(4*inter_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        mean = self.conv4_mean(conv3)
        logvar = self.conv4_logvar(conv3)
        # (16,4,4,4)
        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, out_num_ch=1, img_size=(64,64,64), inter_num_ch=16, kernel_size=3, conv_act='leaky_relu', num_conv=2):
        super(Decoder, self).__init__()

        self.conv4 = DecoderBlock(inter_num_ch, 4*inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
        self.conv3 = DecoderBlock(4*inter_num_ch, 2*inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
        self.conv2 = DecoderBlock(2*inter_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
        self.conv1 = DecoderBlock(inter_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act, dropout=0, num_conv=num_conv)
        self.conv0 = nn.Conv3d(inter_num_ch, out_num_ch, kernel_size=3, padding=1)

    def forward(self, x):
        x_reshaped = x.view(x.shape[0], 16, 4, 4, 4)
        conv4 = self.conv4(x_reshaped)
        conv3 = self.conv3(conv4)
        conv2 = self.conv2(conv3)
        conv1 = self.conv1(conv2)
        output = self.conv0(conv1)
        return output

# replaced with vnn layers!
class Classifier(nn.Module):
    def __init__(self, latent_size=1024, inter_num_ch=64):
        super(Classifier, self).__init__()
        if latent_size == '2048':  # z+delta_z
            self.fc = nn.Sequential(
                            vn_layers.VNBatchNorm(latent_size),
                            #nn.BatchNorm1d(latent_size),

                            nn.Dropout(0.5),

                            #nn.Linear(latent_size, inter_num_ch),
                            vn_layers.VNLinear(latent_size, inter_num_ch),
                            #vn_layers.VNLeakyReLU(inter_num_ch, negative_slope=0.2),
                            nn.LeakyReLU(0.2),
                            #vn_layers.VNLinearAndLeakyReLU(latent_size, inter_num_ch, negative_slope=0.2),

                            #nn.Linear(inter_num_ch, 1)
                            vn_layers.VNLinear(inter_num_ch, 1)
                        )
        else:           # z
            self.fc = nn.Sequential(
                        nn.Dropout(0.2),

                        #nn.Linear(latent_size, inter_num_ch),
                        vn_layers.VNLinear(latent_size, inter_num_ch),
                        #vn_layers.VNLeakyReLU(inter_num_ch, negative_slope=0.2),
                        nn.LeakyReLU(0.2),
                        #vn_layers.VNLinearAndLeakyReLU(latent_size, inter_num_ch, negative_slope=0.2),

                        #nn.Linear(inter_num_ch, 1)
                        vn_layers.VNLinear(inter_num_ch, 1)
                        )
        self._init()

    def _init(self):
        for layer in self.fc.children():
            if isinstance(layer, nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        return self.fc(x)

class VN_Module(nn.Module):
    def __init__(self, latent_size=1024, inter_num_ch=64):
        super(VN_Module, self).__init__()
        self.fc = nn.Sequential(
                    vn_layers.VNLinear(1, 3),
                    #vn_layers.VNLeakyReLU(inter_num_ch, negative_slope=0.2),
                    #vn_layers.VNBatchNorm(3),
                )
        self._init()

    def _init(self):
        for layer in self.fc.children():
            if isinstance(layer, nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        return self.fc(x)

class CLS(nn.Module):
    def __init__(self, latent_size=1024, use_feature=['z', 'delta_z'], dropout=False, gpu=None, num_conv=1, encoder='base', ViT_dims=(1024, 6, 6, 16), feature_size=24, kernel_size=3):
        super(CLS, self).__init__()
        self.gpu = gpu
        self.use_feature = use_feature
        if encoder == 'base':
            self.encoder = EncoderVN(in_num_ch=1, inter_num_ch=16, num_conv=num_conv, dropout=dropout, normalize_output=False, kernel_size=kernel_size)
        elif encoder == 'ViT':
            self.encoder = ViT(
                in_channels=1,
                img_size = (64, 64, 64),
                patch_size = (ViT_dims[3], ViT_dims[3], ViT_dims[3]),
                mlp_dim = ViT_dims[0],
                num_layers = ViT_dims[1],
                num_heads = ViT_dims[2],
                hidden_size = latent_size,
                proj_type='conv',
                pos_embed_type='sincos', 
                classification=False
            )
        elif encoder == 'SWIN':
            # https://developer.nvidia.com/blog/novel-transformer-model-achieves-state-of-the-art-benchmarks-in-3d-medical-image-analysis/
            # https://docs.monai.io/en/stable/_modules/monai/networks/nets/swin_unetr.html
            self.encoder = SwinClassifier(
                #img_size = (64, 64, 64),
                in_channels = 1,
                out_channels = 1,
                spatial_dims=3,
                feature_size=feature_size
            )
        self.encoder_name = encoder
        self.classifier = Classifier(latent_size=len(use_feature)*latent_size, inter_num_ch=64)

    def forward(self, img1, img2, interval):
        bs = img1.shape[0]
        zs = self.encoder(torch.cat([img1, img2], 0))
        zs_flatten = zs.view(bs*2, -1)
        z1, z2 = zs_flatten[:bs], zs_flatten[bs:]
        delta_z = (z2 - z1) / interval.unsqueeze(1)      # [bs, ls]

        if len(self.use_feature) == 2:
            input = torch.cat([z1, delta_z], 1)
        elif 'z' in self.use_feature:
            input = z1
        else:
            input = delta_z
        pred = self.classifier(input)
        return pred

    def forward_single(self, img1):
        z1 = self.encoder(img1)
        
        if self.encoder_name == "ViT":
            z1 = z1[0][:, 0]
        z1 = z1.view(img1.shape[0], -1)
        pred = self.classifier(z1)
        return pred, z1

    def compute_classification_loss(self, pred, label, pos_weight=torch.tensor([2.]), dataset_name='ADNI', postfix='NC_AD', task='classification'):
        if task == 'age':
            loss = nn.MSELoss()(pred.squeeze(1), label)
            return loss, pred
        else:
            # pdb.set_trace()
            if dataset_name == 'ADNI':
                if  'NC_AD' in postfix:
                    label = label / 2
                    #label[label != 0] = 1 # anything not NC is 1
                elif 'pMCI_sMCI' in postfix:
                    label = label - 3
            elif dataset_name == 'LAB':
                if 'C_E_HE' in postfix:
                    label = (label > 0).double()
            elif dataset_name == 'NCANDA':
                label = (label > 0).double()
            elif dataset_name == 'RADFUSION':
                label = label
            else:
                raise ValueError('Not supported!')
            loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.gpu, dtype=torch.float))(pred.squeeze(1), label)
            return loss, F.sigmoid(pred)
        
    def compute_classification_invariance_loss(self, pred, label, z1, z2, pos_weight=torch.tensor([2.]), dataset_name='ADNI', postfix='NC_AD', task='classification', alpha=0.5):
        if task == 'age':
            loss = nn.MSELoss()(pred.squeeze(1), label)
            return loss, pred
        else:
            # pdb.set_trace()
            if dataset_name == 'ADNI':
                if  'NC_AD' in postfix:
                    label = label / 2
                elif 'pMCI_sMCI' in postfix:
                    label = label - 3
            elif dataset_name == 'LAB':
                if 'C_E_HE' in postfix:
                    label = (label > 0).double()
            elif dataset_name == 'NCANDA':
                label = (label > 0).double()
            elif dataset_name == 'RADFUSION':
                label = label
            else:
                raise ValueError('Not support!')
            
            # calculate the weight based on class imbalance
            # based on pytroch recommended weight computation: pos_weight to be a ratio between the negative counts and the positive counts for each classpos_weight to be a ratio between the negative counts and the positive counts for each class
            num_samples = len(label)
            positive_samples = sum(label)
            positive_samples = 1 if positive_samples == 0 else positive_samples
            negative_samples = num_samples - positive_samples
            weight = torch.tensor([negative_samples / positive_samples])

            loss = nn.BCEWithLogitsLoss(pos_weight=weight.to(self.gpu, dtype=torch.float))(pred.squeeze(1), label)
            invariance_loss = alpha * nn.MSELoss()(z1, z2)
            loss += invariance_loss
            return loss, invariance_loss, F.sigmoid(pred)
        

class VN_Net(nn.Module):
    def __init__(self, latent_size=1024, use_feature=['z', 'delta_z'], num_conv=1, encoder='base', vn_module = False, dropout=False, gpu=None, normalize_output=False, ViT_dims=(1024, 6, 6, 16), feature_size=24):
        # ViT_dims is the mlp dimension and number of layers for the ViT model
        super(VN_Net, self).__init__()
        self.gpu = gpu
        self.use_feature = use_feature
        self.vn_module = None
        self.normalize_output = normalize_output

        self.encoder_name = encoder

        #if encoder == 'base':
        #    self.encoder = EncoderBase(in_num_ch=1, inter_num_ch=16, num_conv=num_conv, dropout=dropout)
        #elif encoder == 'vn':
        if encoder == 'base':
            self.encoder = EncoderVN(in_num_ch=1, inter_num_ch=16, num_conv=num_conv, dropout=dropout, normalize_output=normalize_output)
        elif encoder == 'ViT':
            self.encoder = ViT(
                in_channels=1,
                img_size = (64, 64, 64),
                patch_size = (ViT_dims[3], ViT_dims[3], ViT_dims[3]),
                mlp_dim = ViT_dims[0],
                num_layers = ViT_dims[1],
                num_heads = ViT_dims[2],
                hidden_size = latent_size,
                proj_type='conv',
                pos_embed_type='sincos', 
                classification=False # to get true hidden state output
            )
        elif encoder == 'SWIN':
            # https://developer.nvidia.com/blog/novel-transformer-model-achieves-state-of-the-art-benchmarks-in-3d-medical-image-analysis/
            # https://docs.monai.io/en/stable/_modules/monai/networks/nets/swin_unetr.html
            self.encoder = SwinClassifier(
                #img_size = (64, 64, 64),
                in_channels = 1,
                out_channels = 2,
                spatial_dims=3,
                feature_size=feature_size
            )

        if vn_module:
            self.vn_module = VN_Module(latent_size=latent_size, inter_num_ch=64)

        self.classifier = Classifier(latent_size=len(use_feature)*latent_size, inter_num_ch=64)

    def forward(self, img1, img2, interval):
        bs = img1.shape[0]
        zs = self.encoder(torch.cat([img1, img2], 0))
        zs_flatten = zs.view(bs*2, -1)
        z1, z2 = zs_flatten[:bs], zs_flatten[bs:]
        delta_z = (z2 - z1) / interval.unsqueeze(1)      # [bs, ls]

        if len(self.use_feature) == 2:
            input = torch.cat([z1, delta_z], 1)
        elif 'z' in self.use_feature:
            input = z1
        else:
            input = delta_z
        pred = self.classifier(input)
        return pred

    def forward_single(self, img1):
        z1 = self.encoder(img1)
        if self.encoder_name == "ViT":
            z1 = z1[0][:, 0]
        z1 = z1.view(img1.shape[0], -1)
        pred = self.classifier(z1)
        return pred, z1
    
    def forward_encoder(self, img1):
        return self.encoder(img1)

    def forward_single_fs(self, img1):
        #print(f'Input shape: {img1.shape}')
        z1 = self.encoder(img1) # [bs,16,4,4,4], for ViT [bs, 768]
        #print(f'Encoder output shape: {z1.shape}')

        if self.encoder_name == "base":
            z1_reshaped = z1.view(-1, 16, 16, 4) # [bs,16,16,4]
            #print(f'Encoder output shape after reshape: {z1_reshaped.shape}')
            if self.vn_module is None:
                return z1_reshaped
        elif self.encoder_name == "ViT":
            z1 = z1[0][:, 0]

        # don't need this, need this to flatten in order to pass into a FC Network, but could keep for consistency
        z1_flattened = z1.view(img1.shape[0], -1, 1) # [bs,n,1]
        # n dimensional 1D vector, as in paper
        # would be input to classifier/decoder for classification result
        #print(f'Encoder output shape as n dimensional 1D vector: {z1_flattened.shape}')

        # reshape to 2D tensor for VN module
        #z1_flattened_2d = z1_flattened.view(-1, 1) # [bs * n, 1]
        z1_flattened_2d = z1_flattened.reshape(-1, 1) # [bs * n, 1]
        
        # use the VN module to get the nx3 output
        z1_out_2d = self.vn_module(z1_flattened_2d) # [bs * n, 3]
        #print(f'VN module output shape: {z1_out_2d.shape}')

        # Reshape back to 3D tensor
        z1_out = z1_out_2d.view(img1.shape[0], -1, 3) # [bs, n, 3]
        #print(f'VN module output shape as 3D tensor: {z1_out.shape}')

        return z1_out

    def compute_classification_invariance_loss(self, pred, label, z1, z2, pos_weight=torch.tensor([2.]), dataset_name='ADNI', postfix='NC_AD', task='classification', alpha=0.5):
        if task == 'age':
            loss = nn.MSELoss()(pred.squeeze(1), label)
            return loss, pred
        else:
            # pdb.set_trace()
            if dataset_name == 'ADNI':
                if  'NC_AD' in postfix:
                    label = label / 2
                elif 'pMCI_sMCI' in postfix:
                    label = label - 3
            elif dataset_name == 'LAB':
                if 'C_E_HE' in postfix:
                    label = (label > 0).double()
            elif dataset_name == 'NCANDA':
                label = (label > 0).double()
            elif dataset_name == 'RADFUSION':
                label = label
            else:
                raise ValueError('Not support!')
            
            # calculate the weight based on class imbalance
            # based on pytroch recommended weight computation: pos_weight to be a ratio between the negative counts and the positive counts for each classpos_weight to be a ratio between the negative counts and the positive counts for each class
            num_samples = len(label)
            positive_samples = sum(label)
            positive_samples = 1 if positive_samples == 0 else positive_samples
            negative_samples = num_samples - positive_samples
            weight = torch.tensor([negative_samples / positive_samples])

            loss = nn.BCEWithLogitsLoss(pos_weight=weight.to(self.gpu, dtype=torch.float))(pred.squeeze(1), label)
            invariance_loss = alpha * nn.MSELoss()(z1, z2)
            loss += invariance_loss
            return loss, invariance_loss, F.sigmoid(pred)
        
    def compute_classification_loss(self, pred, label, pos_weight=torch.tensor([2.]), dataset_name='ADNI', postfix='NC_AD', task='classification'):
        if task == 'age':
            loss = nn.MSELoss()(pred.squeeze(1), label)
            return loss, pred
        else:
            # pdb.set_trace()
            if dataset_name == 'ADNI':
                if  'NC_AD' in postfix:
                    label = label / 2
                elif 'pMCI_sMCI' in postfix:
                    label = label - 3
            elif dataset_name == 'LAB':
                if 'C_E_HE' in postfix:
                    label = (label > 0).double()
            elif dataset_name == 'NCANDA':
                label = (label > 0).double()
            elif dataset_name == 'RADFUSION':
                label = label
            else:
                raise ValueError('Not support!')
            
            # calculate the weight based on class imbalance
            # based on pytroch recommended weight computation: pos_weight to be a ratio between the negative counts and the positive counts for each classpos_weight to be a ratio between the negative counts and the positive counts for each class
            num_samples = len(label)
            positive_samples = sum(label)
            positive_samples = 1 if positive_samples == 0 else positive_samples
            negative_samples = num_samples - positive_samples
            weight = torch.tensor([negative_samples / positive_samples])

            loss = nn.BCEWithLogitsLoss(pos_weight=weight.to(self.gpu, dtype=torch.float))(pred.squeeze(1), label)

            return loss, F.sigmoid(pred)

    def compute_distance_loss(self, fs1, post_rot_fs1, fs2, post_rot_fs2, bs, alpha=1):      
        # Calculate the Frobenius norm (L2 norm) of the difference between the matrices
        diff_matrix_1 = fs2 - post_rot_fs1
        diff_matrix_2 = fs1 - post_rot_fs2
        # updated loss version
        distance_loss = torch.norm(diff_matrix_1, p='fro', dim=(1, 2)) + torch.norm(diff_matrix_2, p='fro', dim=(1, 2))
        #return torch.mean(distance_loss)        
        #return distance_loss / bs

        # updated loss extra version, week 19, avoid embedding to same point and also avoid invariance
        # also consider maximizing the distance between the matrices fs1 and fs2 to avoid them being embedded to the same thing
        diff_matrix_3 = fs1 - fs2
        diff_matrix_4 = post_rot_fs1 - post_rot_fs2
        maximize_loss = 1 / torch.norm(diff_matrix_3, p='fro', dim=(1, 2))  # maximize the distance

        return torch.mean(distance_loss) + alpha * torch.mean(maximize_loss), [torch.mean(distance_loss), torch.mean(maximize_loss)]
        
        # so, what I want to enforce is that the distance between fs1 and fs2 is larger than the distance between fs1 and post_rot_fs2
        # and the distance between fs2 and fs1 is larger than the distance between fs2 and post_rot_fs1
        # this is what has to be added to the loss:
        # diff_matrix_4 = fs1 - post_rot_fs2
        # diff_matrix_5 = fs2 - post_rot_fs1
        # maximize_loss = -torch.norm(diff_matrix_3, p='fro', dim=(1, 2)) + torch.norm(diff_matrix_4, p='fro', dim=(1, 2)) - torch.norm(diff_matrix_5, p='fro', dim=(1, 2))

        #return torch.mean(distance_loss) + 0.1 * torch.mean(maximize_loss)
