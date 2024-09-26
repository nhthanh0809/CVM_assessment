from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision.models as models
from torchsummaryX import summary

class CephaLandmark_v2(nn.Module):
    def __init__(self, points_num=17):
        super(CephaLandmark_v2, self).__init__()

        self.features = models.densenet121(pretrained=True).features
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.w1_conv11_0 = nn.Sequential(OrderedDict([
            ('conv11_0', nn.Conv2d(3, 32, kernel_size=1)),
            ('norm11_0', nn.BatchNorm2d(32)),
            ('relu11_0', nn.ReLU(inplace=True)),
        ]))
        self.w1_conv33_01 = nn.Sequential(OrderedDict([
            ('conv11_1', nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)),
            ('norm11_1', nn.BatchNorm2d(512)),
            ('relu11_1', nn.ReLU(inplace=True)),
        ]))

        self.mid_conv11_1 = nn.Sequential(OrderedDict([
            ('conv11_4', nn.Conv2d(512, 256, kernel_size=1)),
            ('norm11_4', nn.BatchNorm2d(256)),
            ('relu11_4', nn.ReLU(inplace=True)),
        ]))

        self.mid_conv33_01 = nn.Sequential(OrderedDict([
            ('conv11_1', nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)),
            ('norm11_1', nn.BatchNorm2d(512)),
            ('relu11_1', nn.ReLU(inplace=True)),
        ]))
        self.mid_conv11_2 = nn.Sequential(OrderedDict([
            ('conv11_4', nn.Conv2d(512, 256, kernel_size=1)),
            ('norm11_4', nn.BatchNorm2d(256)),
            ('relu11_4', nn.ReLU(inplace=True)),
        ]))
        self.mid_conv11_3 = nn.Sequential(OrderedDict([
            ('conv11_4', nn.Conv2d(320, 128, kernel_size=1)),
            ('norm11_4', nn.BatchNorm2d(128)),
            ('relu11_4', nn.ReLU(inplace=True)),
        ]))

        self.w1_conv11_1 = nn.Sequential(OrderedDict([
            ('conv11_1', nn.Conv2d(512, 256, kernel_size=1)),
            ('norm11_1', nn.BatchNorm2d(256)),
            ('relu11_1', nn.ReLU(inplace=True)),
        ]))

        self.w1_conv11_2 = nn.Sequential(OrderedDict([
            ('conv11_2', nn.Conv2d(1280, 256, kernel_size=1)),
            ('norm11_2', nn.BatchNorm2d(256)),
            ('relu11_2', nn.ReLU(inplace=True)),
        ]))
        self.w1_conv11_3 = nn.Sequential(OrderedDict([
            ('conv11_3', nn.Conv2d(768, 256, kernel_size=1)),
            ('norm11_3', nn.BatchNorm2d(256)),
            ('relu11_3', nn.ReLU(inplace=True)),
        ]))

        self.w2_conv11_5 = nn.Sequential(OrderedDict([
            ('conv11_4', nn.Conv2d(128, 64, kernel_size=1)),
            ('norm11_4', nn.BatchNorm2d(64)),
            ('relu11_4', nn.ReLU(inplace=True)),
        ]))
        self.conv_33_refine1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        last_conv_size = 64 + 32 + points_num
        self.conv_33_refine2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv_11_refine = nn.Conv2d(64, points_num, kernel_size=1)
        self.conv_33_last1 = nn.Conv2d(last_conv_size, last_conv_size, kernel_size=3, stride=1, padding=1)
        self.conv_33_last2 = nn.Conv2d(last_conv_size, last_conv_size, kernel_size=3, stride=1, padding=1)
        self.conv_11_last = nn.Conv2d(last_conv_size, points_num, kernel_size=1)

    def forward(self, x):
        w1_f0 = self.w1_conv11_0(x)
        x = self.features[0](x)
        w1_f1 = x

        for i in range(1, 5):
            x = self.features[i](x)
        w1_f2 = x

        for i in range(5, 7):
            x = self.features[i](x)
        w1_f3 = x
        for i in range(7, 9):
            x = self.features[i](x)
        w1_f4 = x

        for i in range(9, 12):
            x = self.features[i](x)
        x = self.w1_conv33_01(x)
        x = self.w1_conv11_1(x)
        x = self.upsample2(x)
        x = torch.cat((x, w1_f4), 1)
        x = self.w1_conv11_2(x)
        x = self.upsample2(x)
        x = torch.cat((x, w1_f3), 1)
        x = self.w1_conv11_3(x)
        x = self.upsample2(x)
        x = torch.cat((x, w1_f2), 1)
        x = self.mid_conv11_1(x)
        x = self.mid_conv33_01(x)
        x = self.mid_conv11_2(x)
        x = self.upsample2(x)
        x = torch.cat((x, w1_f1), 1)
        x = self.mid_conv11_3(x)
        x = self.w2_conv11_5(x)
        refine_hp = self.conv_33_refine1(x)
        refine_hp = self.conv_11_refine(refine_hp)
        x = self.upsample2(x)
        refine1_up = self.upsample2(refine_hp)
        x = torch.cat((x, w1_f0, refine1_up), 1)
        # output
        hp = self.conv_33_last1(x)
        hp = self.conv_11_last(hp)
        return hp, refine_hp
#
# if __name__ == '__main__':
#     model = CephaLandmark_v2(points_num=17)
#     img = torch.rand([1, 3, 512, 480])
#     arch = summary(model, torch.rand([1, 3, 512, 480]))
#     outputs, outputs_refine = model(img)














