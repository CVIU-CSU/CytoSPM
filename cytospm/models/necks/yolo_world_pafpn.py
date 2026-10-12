# Copyright (c) Tencent Inc. All rights reserved.
# RepVGG portions Copyright (c) 2020 DingXiaoH (MIT).
# Modified by the CytoSPM Authors, 2026.
import torch
import torch.nn as nn

from mmdet.registry import MODELS
from torch.nn.parameter import Parameter
from torch.nn.modules.batchnorm import _BatchNorm


activation_table = {'relu':nn.ReLU(),
                    'silu':nn.SiLU(),
                    'hardswish':nn.Hardswish()
                    }


class ConvModule_torch(nn.Module):
    '''A combination of Conv + BN + Activation'''
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation_type, padding=None, groups=1, bias=False):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        if activation_type is not None:
            self.act = activation_table.get(activation_type)
        self.activation_type = activation_type

    def forward(self, x):
        if self.activation_type is None:
            return self.bn(self.conv(x))
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        if self.activation_type is None:
            return self.conv(x)
        return self.act(self.conv(x))


class ConvBNReLU(nn.Module):
    '''Conv and BN with ReLU activation'''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1, bias=False):
        super().__init__()
        self.block = ConvModule_torch(in_channels, out_channels, kernel_size, stride, 'relu', padding, groups, bias)

    def forward(self, x):
        return self.block(x)


class ConvBNSiLU(nn.Module):
    '''Conv and BN with SiLU activation'''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1, bias=False):
        super().__init__()
        self.block = ConvModule_torch(in_channels, out_channels, kernel_size, stride, 'silu', padding, groups, bias)

    def forward(self, x):
        return self.block(x)


class Transpose(nn.Module):
    '''Normal Transpose, default for upsampling'''
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.upsample_transpose = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=True
        )

    def forward(self, x):
        return self.upsample_transpose(x)


class RepVGGBlock(nn.Module):
    '''RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        """ Initialization of the class.
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of
                the input. Default: 1
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            padding_mode (string, optional): Default: 'zeros'
            deploy: Whether to be deploy status or training status. Default: False
            use_se: Whether to use se. Default: False
        """
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = ConvModule_torch(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, activation_type=None, padding=padding, groups=groups)
            self.rbr_1x1 = ConvModule_torch(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, activation_type=None, padding=padding_11, groups=groups)

    def forward(self, inputs):
        '''Forward process'''
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _avg_to_3x3_tensor(self, avgp):
        channels = self.in_channels
        groups = self.groups
        kernel_size = avgp.kernel_size
        input_dim = channels // groups
        k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
        k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2
        return k

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, ConvModule_torch):
            kernel = branch.conv.weight
            bias = branch.conv.bias
            return kernel, bias
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


#   A CSLA block is a LinearAddBlock with is_csla=True


class RepBlock(nn.Module):
    '''
        RepBlock is a stage block with rep-style basic block
    '''
    def __init__(self, in_channels, out_channels, n=1, block=RepVGGBlock, basic_block=RepVGGBlock):
        super().__init__()

        self.conv1 = block(in_channels, out_channels)
        self.block = nn.Sequential(*(block(out_channels, out_channels) for _ in range(n - 1))) if n > 1 else None
        if block == BottleRep:
            self.conv1 = BottleRep(in_channels, out_channels, basic_block=basic_block, weight=True)
            n = n // 2
            self.block = nn.Sequential(*(BottleRep(out_channels, out_channels, basic_block=basic_block, weight=True) for _ in range(n - 1))) if n > 1 else None

    def forward(self, x):
        x = self.conv1(x)
        if self.block is not None:
            x = self.block(x)
        return x


class BottleRep(nn.Module):

    def __init__(self, in_channels, out_channels, basic_block=RepVGGBlock, weight=False):
        super().__init__()
        self.conv1 = basic_block(in_channels, out_channels)
        self.conv2 = basic_block(out_channels, out_channels)
        if in_channels != out_channels:
            self.shortcut = False
        else:
            self.shortcut = True
        if weight:
            self.alpha = Parameter(torch.ones(1))
        else:
            self.alpha = 1.0

    def forward(self, x):
        outputs = self.conv1(x)
        outputs = self.conv2(outputs)
        return outputs + self.alpha * x if self.shortcut else outputs


class BottleRep3(nn.Module):

    def __init__(self, in_channels, out_channels, basic_block=RepVGGBlock, weight=False):
        super().__init__()
        self.conv1 = basic_block(in_channels, out_channels)
        self.conv2 = basic_block(out_channels, out_channels)
        self.conv3 = basic_block(out_channels, out_channels)
        if in_channels != out_channels:
            self.shortcut = False
        else:
            self.shortcut = True
        if weight:
            self.alpha = Parameter(torch.ones(1))
        else:
            self.alpha = 1.0

    def forward(self, x):
        outputs = self.conv1(x)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs + self.alpha * x if self.shortcut else outputs


class BepC3(nn.Module):
    '''CSPStackRep Block'''
    def __init__(self, in_channels, out_channels, n=1, e=0.5, block=RepVGGBlock):
        super().__init__()
        c_ = int(out_channels * e)  # hidden channels
        self.cv1 = ConvBNReLU(in_channels, c_, 1, 1)
        self.cv2 = ConvBNReLU(in_channels, c_, 1, 1)
        self.cv3 = ConvBNReLU(2 * c_, out_channels, 1, 1)
        if block == ConvBNSiLU:
            self.cv1 = ConvBNSiLU(in_channels, c_, 1, 1)
            self.cv2 = ConvBNSiLU(in_channels, c_, 1, 1)
            self.cv3 = ConvBNSiLU(2 * c_, out_channels, 1, 1)

        self.m = RepBlock(in_channels=c_, out_channels=c_, n=n, block=BottleRep, basic_block=block)

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class MBLABlock(nn.Module):
    ''' Multi Branch Layer Aggregation Block'''
    def __init__(self, in_channels, out_channels, n=1, e=0.5, block=RepVGGBlock):
        super().__init__()
        n = n // 2
        if n <= 0:
            n = 1

        # max add one branch
        if n == 1:
            n_list = [0, 1]
        else:
            extra_branch_steps = 1
            while extra_branch_steps * 2 < n:
                extra_branch_steps *= 2
            n_list = [0, extra_branch_steps, n]
        branch_num = len(n_list)

        c_ = int(out_channels * e)  # hidden channels
        self.c = c_
        self.cv1 = ConvModule_torch(in_channels, branch_num * self.c, 1, 1, 'relu', bias=False)
        self.cv2 = ConvModule_torch((sum(n_list) + branch_num) * self.c, out_channels, 1, 1,'relu', bias=False)

        if block == ConvBNSiLU:
            self.cv1 = ConvModule_torch(in_channels, branch_num * self.c, 1, 1, 'silu', bias=False)
            self.cv2 = ConvModule_torch((sum(n_list) + branch_num) * self.c, out_channels, 1, 1,'silu', bias=False)

        self.m = nn.ModuleList()
        for n_list_i in n_list[1:]:
            self.m.append(nn.Sequential(*(BottleRep3(self.c, self.c, basic_block=block, weight=True) for _ in range(n_list_i))))

        self.split_num = tuple([self.c]*branch_num)

    def forward(self, x):
        y = list(self.cv1(x).split(self.split_num, 1))
        all_y = [y[0]]
        for m_idx, m_i in enumerate(self.m):
            all_y.append(y[m_idx+1])
            all_y.extend(m(all_y[-1]) for m in m_i)
        return self.cv2(torch.cat(all_y, 1))


class BiFusion(nn.Module):
    '''BiFusion Block in PAN'''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cv1 = ConvBNReLU(in_channels[0], out_channels, 1, 1)
        self.cv2 = ConvBNReLU(in_channels[1], out_channels, 1, 1)
        self.cv3 = ConvBNReLU(out_channels * 3, out_channels, 1, 1)

        self.upsample = Transpose(
            in_channels=out_channels,
            out_channels=out_channels,
        )
        self.downsample = ConvBNReLU(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2
        )

    def forward(self, x):
        x0 = self.upsample(x[0])
        x1 = self.cv1(x[1])
        x2 = self.downsample(self.cv2(x[2]))
        return self.cv3(torch.cat((x0, x1, x2), dim=1))


@MODELS.register_module()
class CSPRepBiFPANNeck(nn.Module):
    """
    CSPRepBiFPANNeck module.
    """

    def __init__(self,
                 scale_factor=0.75,
                 model_size='base',
                 freeze_all: bool = False) -> None:
        super().__init__()

        channels_list=[64, 128, 256, 512, 1024, 256, 128, 128, 256, 256, 512]

        if model_size == 'small' or model_size == 'base' or model_size == 'large':
            num_repeats=[1, 6, 12, 18, 6, 12, 12, 12, 12] # small base large
        if model_size == 'tiny':
            num_repeats=[1, 6, 12, 18, 6, 6, 6, 6, 6] # tiny

        block=ConvBNSiLU
        csp_e=float(1)/2
        stage_block_type = "BepC3"

        assert channels_list is not None
        assert num_repeats is not None

        if stage_block_type == "BepC3":
            stage_block = BepC3
        elif stage_block_type == "MBLABlock":
            stage_block = MBLABlock
        else:
            raise NotImplementedError
        
        scale_factor = scale_factor

        self.reduce_layer0 = ConvBNReLU(
            in_channels=int(channels_list[4] * scale_factor), # 1024
            out_channels=int(channels_list[5] * scale_factor), # 256
            kernel_size=1,
            stride=1
        )

        self.Bifusion0 = BiFusion(
            in_channels=[int(channels_list[3] * scale_factor), int(channels_list[2] * scale_factor)], # 512, 256
            out_channels=int(channels_list[5] * scale_factor), # 256
        )

        self.Rep_p4 = stage_block(
            in_channels=int(channels_list[5] * scale_factor), # 256
            out_channels=int(channels_list[5] * scale_factor), # 256
            n=num_repeats[5],
            e=csp_e,
            block=block
        )

        self.reduce_layer1 = ConvBNReLU(
            in_channels=int(channels_list[5] * scale_factor), # 256
            out_channels=int(channels_list[6] * scale_factor), # 128
            kernel_size=1,
            stride=1
        )

        self.Bifusion1 = BiFusion(
            in_channels=[int(channels_list[2] * scale_factor), int(channels_list[1] * scale_factor)], # 256, 128
            out_channels=int(channels_list[6] * scale_factor), # 128
        )

        self.Rep_p3 = stage_block(
            in_channels=int(channels_list[6] * scale_factor), # 128
            out_channels=int(channels_list[6] * scale_factor), # 128
            n=num_repeats[6],
            e=csp_e,
            block=block
        )

        self.downsample2 = ConvBNReLU(
            in_channels=int(channels_list[6] * scale_factor), # 128
            out_channels=int(channels_list[7] * scale_factor), # 128
            kernel_size=3,
            stride=2
        )

        self.Rep_n3 = stage_block(
            in_channels=int(channels_list[6] * scale_factor) + int(channels_list[7] * scale_factor), # 128 + 128
            out_channels=int(channels_list[8] * scale_factor), # 256
            n=num_repeats[7],
            e=csp_e,
            block=block
        )

        self.downsample1 = ConvBNReLU(
            in_channels=int(channels_list[8] * scale_factor), # 256
            out_channels=int(channels_list[9] * scale_factor), # 256
            kernel_size=3,
            stride=2
        )


        self.Rep_n4 = stage_block(
            in_channels=int(channels_list[5] * scale_factor) + int(channels_list[9] * scale_factor), # 256 + 256
            out_channels=int(channels_list[10] * scale_factor), # 512
            n=num_repeats[8],
            e=csp_e,
            block=block
        )

        self.freeze_all = freeze_all
        if self.freeze_all:
            self._freeze_all()


    def train(self, mode=True):
        """Convert the model into training mode while keep the normalization
        layer freezed."""
        super().train(mode)
        if self.freeze_all:
            self._freeze_all()
            
    
    def _freeze_all(self):
        """Freeze the model."""
        for m in self.modules():
            if isinstance(m, _BatchNorm):
                m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, input):

        (x3, x2, x1, x0) = input

        fpn_out0 = self.reduce_layer0(x0)
        f_concat_layer0 = self.Bifusion0([fpn_out0, x1, x2])
        f_out0 = self.Rep_p4(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        f_concat_layer1 = self.Bifusion1([fpn_out1, x2, x3])
        pan_out2 = self.Rep_p3(f_concat_layer1)

        down_feat1 = self.downsample2(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n3(p_concat_layer1)

        down_feat0 = self.downsample1(pan_out1)
        p_concat_layer2 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n4(p_concat_layer2)

        return [pan_out2, pan_out1, pan_out0]


