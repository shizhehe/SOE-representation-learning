import vn_layers2D as vn_layers

import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F



class get_model(nn.Module):

    def __init__(self, num_class=10, normal_channel=False):
        super(get_model, self).__init__()
        self.n_knn = 5 #args.n_knn
        
        self.conv1 = vn_layers.VNLinearLeakyReLU(1, 64//3)
        self.conv2 = vn_layers.VNLinearLeakyReLU(64//3, 64//3) 
        self.conv3 = vn_layers.VNLinearLeakyReLU(64//3, 128//3)
        self.conv4 = vn_layers.VNLinearLeakyReLU(128//3, 256//3)
        self.conv5 = vn_layers.VNLinearLeakyReLU(256//3+128//3+64//3+64//3, 1024//3, dim=4, share_nonlinearity=True)
        
        self.std_feature = vn_layers.VNStdFeature(1024//3*2, dim=3, normalize_frame=False)
        self.linear1 = nn.Linear((1024//3)*4, 512)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, num_class)
        

        self.pool1 = vn_layers.VNMaxPool(64//3)
        self.pool2 = vn_layers.VNMaxPool(64//3)
        self.pool3 = vn_layers.VNMaxPool(128//3)
        self.pool4 = vn_layers.VNMaxPool(256//3)


    def forward(self, x):
        batch_size = x.size(0)
        #x = x.unsqueeze(1)
        x = self.conv1(x)
        x1 = self.pool1(x)
        
        x = self.conv2(x)
        x2 = self.pool2(x)
        
        x = self.conv3(x)
        x3 = self.pool3(x)
        
        x = self.conv4(x)
        x4 = self.pool4(x)
        
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)
        
        num_points = x.size(-1)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x, trans = self.std_feature(x)
        #swap x with trans?
        #x = x.view(batch_size, -1, num_points)
        
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        x = F.leaky_relu(self.bn1(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn2(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        
        trans_feat = None
        return x, trans_feat



class get_loss(torch.nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()


    def forward(self, pred, target, trans_feat, smoothing=True):
        ''' Calculate cross entropy loss, apply label smoothing if needed. '''

        target = target.contiguous().view(-1)

        if smoothing:
            eps = 0.2
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(pred, target, reduction='mean')
            
        return loss

class get_model_complex(nn.Module):

    def __init__(self, num_class=10, normal_channel=False):
        super(get_model_complex, self).__init__()
        self.n_knn = 5 #args.n_knn
        
        self.conv1 = vn_layers.VNLinearLeakyReLU(1, 64//3)
        self.conv2 = vn_layers.VNLinearLeakyReLU(64//3, 64//3) 
        self.conv3 = vn_layers.VNLinearLeakyReLU(64//3, 128//3)
        self.conv4 = vn_layers.VNLinearLeakyReLU(128//3, 256//3)
        self.conv5 = vn_layers.VNLinearLeakyReLU(256//3+128//3+64//3+64//3, 1024//3, dim=4, share_nonlinearity=True)
        
        self.std_feature = vn_layers.VNStdFeature(1024//3*2, dim=3, normalize_frame=False)
        self.linear1 = nn.Linear((1024//3)*4, 512)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, num_class)
        

        self.pool1 = vn_layers.VNMaxPool(64//3)
        self.pool2 = vn_layers.VNMaxPool(64//3)
        self.pool3 = vn_layers.VNMaxPool(128//3)
        self.pool4 = vn_layers.VNMaxPool(256//3)


    def forward(self, x):
        batch_size = x.size(0)
        #x = x.unsqueeze(1)
        x = self.conv1(x)
        x1 = self.pool1(x)
        
        x = self.conv2(x)
        x2 = self.pool2(x)
        
        x = self.conv3(x)
        x3 = self.pool3(x)
        
        x = self.conv4(x)
        x4 = self.pool4(x)
        
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)
        
        num_points = x.size(-1)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x, trans = self.std_feature(x)
        #swap x with trans?
        #x = x.view(batch_size, -1, num_points)
        
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        x = F.leaky_relu(self.bn1(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn2(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        
        trans_feat = None
        return x, trans_feat



class get_loss(torch.nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()


    def forward(self, pred, target, trans_feat, smoothing=True):
        ''' Calculate cross entropy loss, apply label smoothing if needed. '''

        target = target.contiguous().view(-1)

        if smoothing:
            eps = 0.2
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(pred, target, reduction='mean')
            
        return loss
