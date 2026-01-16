import torch.nn.functional as F
from vgg16 import VGG16BN
import torch
from torch import nn
import math

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class local_att(nn.Module):
    def __init__(self, channel, reduction=16):
        super(local_att, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out


class MIFM(nn.Module):  # 多尺度有效融合模块
    def __init__(self, c1, c2):
        super().__init__()
        kernel_size = int(abs((math.log(c1, 2) +1) / 2))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(c1, c2, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(c2, c2, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(c2, c2, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(c2)
        self.sigomid = nn.Sigmoid()
        self.group_num = 16
        self.eps = 1e-10
        self.gamma = nn.Parameter(torch.randn(c2, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c2, 1, 1))
        self.gate_genator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(c2, c2, 1, 1),
            nn.ReLU(True),
            nn.Softmax(dim=1),
        )
        self.dwconv = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1, groups=c2)
        self.conv3 = nn.Conv2d(c2, c2, kernel_size=1, stride=1)
        self.Apt = nn.AdaptiveAvgPool2d(1)
        self.one = c2
        self.two = c2
        self.conv4_gobal = nn.Conv2d(c2, 1, kernel_size=1, stride=1)
        for group_id in range(0, 4):
            self.interact = nn.Conv2d(c2 // 4, c2 // 4, 1, 1, )

    def forward(self, x1, x2):

        global_conv1 = self.conv1(x1)
        bn_x = self.bn(global_conv1)
        weight_1 = self.sigomid(bn_x)
        global_conv2 = self.conv2(x2)
        bn_x2 = self.bn(global_conv2)
        weight_2 = self.sigomid(bn_x2)
        X_GOBAL = global_conv1 + global_conv2

        x_conv4 = self.conv4_gobal(X_GOBAL)
        X_4_sigmoid = self.sigomid(x_conv4)
        X_ = X_4_sigmoid * X_GOBAL
        X_ = X_.chunk(4, dim=1)
        out = []
        for group_id in range(0, 4):
            out_1 = self.interact(X_[group_id])
            N, C, H, W = out_1.size()
            x_1_map = out_1.reshape(N, 1, -1)
            mean_1 = x_1_map.mean(dim=2, keepdim=True)
            x_1_av = x_1_map / mean_1
            x_2_2 = F.softmax(x_1_av, dim=1)
            x1 = x_2_2.reshape(N, C, H, W)
            x1 = X_[group_id] * x1
            out.append(x1)
        out = torch.cat([out[0], out[1], out[2], out[3]], dim=1)
        N, C, H, W = out.size()
        x_add_1 = out.reshape(N, self.group_num, -1)
        N, C, H, W = X_GOBAL.size()
        x_shape_1 = X_GOBAL.reshape(N, self.group_num, -1)
        mean_1 = x_shape_1.mean(dim=2, keepdim=True)
        std_1 = x_shape_1.std(dim=2, keepdim=True)
        x_guiyi = (x_add_1 - mean_1) / (std_1 + self.eps)
        x_guiyi_1 = x_guiyi.reshape(N, C, H, W)
        x_gui = (x_guiyi_1 * self.gamma + self.beta)

        y = self.avg_pool(X_GOBAL)
        y = y.squeeze(-1)
        y = y.transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        y = X_GOBAL * y.expand_as(X_GOBAL)

        ACGF = x_gui + y

        weight_x3 = self.Apt(X_GOBAL)
        reweights = self.sigomid(weight_x3)
        x_up_1 = reweights >= weight_1
        x_low_1 = reweights < weight_1
        x_up_2 = reweights >= weight_2
        x_low_2 = reweights < weight_2
        x_up = x_up_1 * X_GOBAL + x_up_2 * X_GOBAL
        x_low = x_low_1 * X_GOBAL + x_low_2 * X_GOBAL
        x11_up_dwc = self.dwconv(x_low)
        x11_up_dwc = self.conv3(x11_up_dwc)
        x_so = self.gate_genator(x_low)
        x11_up_dwc = x11_up_dwc * x_so
        x22_low_pw = self.conv4(x_up)
        RFRB = x11_up_dwc + x22_low_pw


        out = RFRB + ACGF


        return out

class MP_Block(nn.Module):
    """Multi-level Interaction Perception Block (MP-Block)."""

    def __init__(self, c1, c2):
        super().__init__()
        # 正确：把模块实例赋值给 self.xxx
        self.mifm = MIFM(c1, c2)      # 注意：这里用类名/构造器
        self.pam = local_att(c2)  # 这里假设 LocalAtt 是类；若是函数也要返回 nn.Module

    def forward(self, x1, x2):
        x = self.mifm(x1, x2)      # 正确：调用 self.mifm
        out = self.pam(x)
        return out


class CFFNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        vgg16_bn = VGG16BN(pretrained=pretrained)
        self.inc = vgg16_bn.features[:5]  # 64
        self.down1 = vgg16_bn.features[5:12]  # 128
        self.down2 = vgg16_bn.features[12:22]  # 256
        self.down3 = vgg16_bn.features[22:32]  # 512
        self.down4 = vgg16_bn.features[32:42]  # 512

        self.conv_reduce_1 = BasicConv2d(128*2, 128, 3, 1, 1)
        self.conv_reduce_2 = BasicConv2d(256*2, 256, 3, 1, 1)
        self.conv_reduce_3 = BasicConv2d(512*2, 512, 3, 1, 1)
        self.conv_reduce_4 = BasicConv2d(512*2, 512, 3, 1, 1)

        self.decoder_final = nn.Sequential(
            BasicConv2d(128, 64, 3, 1, 1),
            nn.Conv2d(64, 1, 1))


        self.MP_B3 = MP_Block(512, 512)
        self.MP_B2 = MP_Block(512, 256)
        self.MP_B1 = MP_Block(256, 128)
    def forward(self, x1, x2):
        size = x1.size()[2:]
        layer1_pre = self.inc(x1)
        layer1_A = self.down1(layer1_pre)
        layer2_A = self.down2(layer1_A)
        layer3_A = self.down3(layer2_A)
        layer4_A = self.down4(layer3_A)

        layer1_pre = self.inc(x2)
        layer1_B = self.down1(layer1_pre)
        layer2_B = self.down2(layer1_B)
        layer3_B = self.down3(layer2_B)
        layer4_B = self.down4(layer3_B)

        layer1 = torch.cat((layer1_B,layer1_A), dim=1)
        layer2 = torch.cat((layer2_B,layer2_A), dim=1)
        layer3 = torch.cat((layer3_B,layer3_A), dim=1)
        layer4 = torch.cat((layer4_B,layer4_A), dim=1)

        layer1 = self.conv_reduce_1(layer1)
        layer2 = self.conv_reduce_2(layer2)
        layer3 = self.conv_reduce_3(layer3)
        layer4 = self.conv_reduce_4(layer4)

        layer4_1 = F.interpolate(layer4, layer3.size()[2:], mode='bilinear', align_corners=True)
        layer3 = self.MP_B3(layer4_1,layer3)

        layer3_1 = F.interpolate(layer3, layer2.size()[2:], mode='bilinear', align_corners=True)
        layer2 = self.MP_B2(layer3_1, layer2)

        layer2_1 = F.interpolate(layer2, layer1.size()[2:], mode='bilinear', align_corners=True)
        layer1 = self.MP_B1(layer2_1, layer1)

        final_map = self.decoder_final(layer1)
        final_map = F.interpolate(final_map, size, mode='bilinear', align_corners=True)

        return final_map




if __name__ == '__main__':
    from thop import profile, clever_format  # 根据实际模型定义路径导入
    # 创建输入张量
    inp1 = torch.rand(1, 3, 256, 256)
    inp2 = torch.rand(1, 3, 256, 256)

    # 创建模型实例
    model = CFFNet()

    # 计算输出（根据模型的前向传播）
    out = model(inp1, inp2)

    # 使用 THOP 计算 FLOPS 和参数量
    flops, param = profile(model, inputs=(inp1, inp2))

    # 格式化 FLOPS 和参数量
    flops, param = clever_format([flops, param], "%.2f")

    # 显示结果
    print(f"FLOPS: {flops}")
    print(f"Parameters: {param}")

