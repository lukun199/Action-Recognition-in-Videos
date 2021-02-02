import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

class Res3d(nn.Module):
    def __init__(self):
        super(Res3d, self).__init__()

        resnet = models.video.r2plus1d_18(pretrained=True) # r3d_18  r2plus1d_18
        self.resnet = nn.Sequential(*list(resnet.children())[:-3])
        # resnet(torch.from_numpy(np.random.rand(1,3,16,160,160)).float())
        self.transforms = transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def preprocess(self, images):
        # images: [16, 160, 160, 3] RGB
        images = images.astype('float32') / 255.0
        images = torch.from_numpy(images.transpose(0, 3, 1, 2)).float()  # to tensor.
        images = torch.stack(list(map(lambda x: self.transforms(x), images)))
        # images: [16, 3, 160, 160]
        images = images.permute(1,0,2,3).unsqueeze(0)
        return images  # [1, 3, 16, 160, 160]


    def forward(self, x_in):
        # ResNet CNN
        with torch.no_grad():
            x = self.resnet(x_in)  # ResNet
            x = x.squeeze()  # [256, 4, 20, 20]
        return x

# 2D CNN encoder using ResNet-101 pretrained
class ResCNNEncoder(nn.Module):
    def __init__(self, resnet101_file, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300, att_size = 5):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResCNNEncoder, self).__init__()

        self.resnet101_file = resnet101_file
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        self.att_size = att_size
        self.transforms = transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        resnet = models.resnet101()
        ckpt = torch.load(self.resnet101_file, map_location=lambda s, l: s)
        resnet.load_state_dict(ckpt)

        modules = list(resnet.children())[:-2]  # come to the last fc layer.
        self.resnet = nn.Sequential(*modules)

        self.fc1 = nn.Linear(resnet.fc.in_features, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim)


    def preprocess(self, images):
        if len(images.shape) == 2:
            images = images[:, :, :, np.newaxis]
            images = np.concatenate((images, images, images), axis=3)

        images = images.astype('float32') / 255.0
        images = torch.from_numpy(images.transpose(0, 3, 1, 2))  # to tensor.
        images = torch.stack(list(map(lambda x: self.transforms(x), images)))

        return images


    def forward(self, x_in):

        # ResNet CNN
        with torch.no_grad():
            x = self.resnet(x_in)  # ResNet
            # x = x.view(x.size(0), -1)  # flatten output of conv

        # after the conv, adapool used

        #conv_feats = F.adaptive_avg_pool2d(x, [self.att_size, self.att_size]).permute(0, 2, 3, 1)
        fc_feats = x.mean(3).mean(2)

        """
        # FC layers
        x = self.bn1(self.fc1(x))
        x = F.relu(x)
        x = self.bn2(self.fc2(x))
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc3(x)
        
        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq
        """

        return fc_feats


# 2D CNN encoder using ResNet-152 pretrained
class ResCNNEncoder_101(nn.Module):
    def __init__(self, resnet101_file, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300, att_size=5):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResCNNEncoder_101, self).__init__()

        self.resnet101_file = resnet101_file
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        self.att_size = att_size
        self.transforms = transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        resnet = models.resnet101()
        ckpt = torch.load(self.resnet101_file, map_location=lambda s, l: s)
        resnet.load_state_dict(ckpt)

        modules = list(resnet.children())[:-1]  # come to the last fc layer.
        self.resnet = nn.Sequential(*modules)

        self.fc1 = nn.Linear(resnet.fc.in_features, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim)

    def preprocess(self, images):
        #images = images.astype('float32') / 255.0
        #images = torch.from_numpy(images.transpose(0, 3, 1, 2))  # to tensor.
        #images = torch.stack(list(map(lambda x: self.transforms(x), images)))
        images = self.transforms(images)

        return images

    def forward(self, x_in):
        # ResNet CNN
        with torch.no_grad():
            x = self.resnet(x_in.unsqueeze(0))  # ResNet
            x = torch.flatten(x, 1)  # flatten output of conv

        # after the conv, adapool used

        # conv_feats = F.adaptive_avg_pool2d(x, [self.att_size, self.att_size]).permute(0, 2, 3, 1)
        # fc_feats = x.mean(3).mean(2)

        """
        # FC layers
        x = self.bn1(self.fc1(x))
        x = F.relu(x)
        x = self.bn2(self.fc2(x))
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc3(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq
        """

        return x


class CNN_fc_EmbedEncoder(nn.Module):

    def __init__(self):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(CNN_fc_EmbedEncoder, self).__init__()
        self.embed_conv1 = nn.Conv2d(256,512,3,stride=2,bias=False)
        self.maxpool = nn.MaxPool2d(2,2)
        self.adapool = nn.AdaptiveAvgPool2d((1,1))
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    def forward(self, x_in, x_bbox):
        # x_in: [bs, 256, 4, 20, 20]
        # x_bbox: [bs, 4]
        tempelate = torch.ones_like(x_in).to(self.device)
        for i in range(x_in.size()[0]):
            tempelate[i,:,:,x_bbox[i,1]//8:x_bbox[i,3]//8,x_bbox[i,0]//8:x_bbox[i,2]//8] = 1.5
        x_in *= tempelate

        x_in = x_in.reshape(x_in.size()[0]*4, -1, 20, 20).float()
        x = self.maxpool(x_in) # [bs*4, 256, 10, 10]
        x = self.embed_conv1(x)  # [bs*4, 512, 4, 4]
        x = self.adapool(x).squeeze()  # [bs*4,512]

        return x.view(-1, 4, 512)

if __name__ == '__main__':
    xx = Res3d()