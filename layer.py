import torch
import math
# import tensorflow as tf
from torch import nn


class AddCoordinates(object):

    r"""Coordinate Adder Module as defined in 'An Intriguing Failing of
    Convolutional Neural Networks and the CoordConv Solution'
    (https://arxiv.org/pdf/1807.03247.pdf).
    This module concatenates coordinate information (`x`, `y`, and `r`) with
    given input tensor.
    `x` and `y` coordinates are scaled to `[-1, 1]` range where origin is the
    center. `r` is the Euclidean distance from the center and is scaled to
    `[0, 1]`.
    Args:
        with_r (bool, optional): If `True`, adds radius (`r`) coordinate
            information to input image. Default: `False`
    Shape:
        - Input: `(N, C_{in}, H_{in}, W_{in})`
        - Output: `(N, (C_{in} + 2) or (C_{in} + 3), H_{in}, W_{in})`
    Examples:
        >>> coord_adder = AddCoordinates(True)
        >>> input = torch.randn(8, 3, 64, 64)
        >>> output = coord_adder(input)
        >>> coord_adder = AddCoordinates(True)
        >>> input = torch.randn(8, 3, 64, 64).cuda()
        >>> output = coord_adder(input)
        >>> device = torch.device("cuda:0")
        >>> coord_adder = AddCoordinates(True)
        >>> input = torch.randn(8, 3, 64, 64).to(device)
        >>> output = coord_adder(input)
    """

    def __init__(self, with_r=False):
        self.with_r = with_r

    def __call__(self, image):
        batch_size, _, image_height, image_width = image.size()

        y_coords = 2.0 * torch.arange(image_height).unsqueeze(
            1).expand(image_height, image_width) / (image_height - 1.0) - 1.0
        x_coords = 2.0 * torch.arange(image_width).unsqueeze(
            0).expand(image_height, image_width) / (image_width - 1.0) - 1.0

        coords = torch.stack((y_coords, x_coords), dim=0)

        if self.with_r:
            rs = ((y_coords ** 2) + (x_coords ** 2)) ** 0.5
            rs = rs / torch.max(rs)
            rs = torch.unsqueeze(rs, dim=0)
            coords = torch.cat((coords, rs), dim=0)

        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1)

        image = torch.cat((coords.to(image.device), image), dim=1)

        return image


class CoordConv(nn.Module):

    r"""2D Convolution Module Using Extra Coordinate Information as defined
    in 'An Intriguing Failing of Convolutional Neural Networks and the
    CoordConv Solution' (https://arxiv.org/pdf/1807.03247.pdf).
    Args:
        Same as `torch.nn.Conv2d` with two additional arguments
        with_r (bool, optional): If `True`, adds radius (`r`) coordinate
            information to input image. Default: `False`
    Shape:
        - Input: `(N, C_{in}, H_{in}, W_{in})`
        - Output: `(N, C_{out}, H_{out}, W_{out})`
    Examples:
        >>> coord_conv = CoordConv(3, 16, 3, with_r=True)
        >>> input = torch.randn(8, 3, 64, 64)
        >>> output = coord_conv(input)
        >>> coord_conv = CoordConv(3, 16, 3, with_r=True).cuda()
        >>> input = torch.randn(8, 3, 64, 64).cuda()
        >>> output = coord_conv(input)
        >>> device = torch.device("cuda:0")
        >>> coord_conv = CoordConv(3, 16, 3, with_r=True).to(device)
        >>> input = torch.randn(8, 3, 64, 64).to(device)
        >>> output = coord_conv(input)
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 with_r=False):
        super(CoordConv, self).__init__()

        in_channels += 2
        if with_r:
            in_channels += 1

        self.conv_layer = nn.Conv2d(in_channels, out_channels,
                                    kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    groups=groups, bias=bias)

        self.coord_adder = AddCoordinates(with_r)

    def forward(self, x):
        x = self.coord_adder(x)
        x = self.conv_layer(x)

        return x

# def get_freq_indices(method):
#     assert method in ['top1','top2','top4','top8','top16','top32',
#                       'bot1','bot2','bot4','bot8','bot16','bot32',
#                       'low1','low2','low4','low8','low16','low32']
#     num_freq = int(method[3:])
#     if 'top' in method:
#         all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
#         all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
#         mapper_x = all_top_indices_x[:num_freq]
#         mapper_y = all_top_indices_y[:num_freq]
#     elif 'low' in method:
#         all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
#         all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
#         mapper_x = all_low_indices_x[:num_freq]
#         mapper_y = all_low_indices_y[:num_freq]
#     elif 'bot' in method:
#         all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
#         all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
#         mapper_x = all_bot_indices_x[:num_freq]
#         mapper_y = all_bot_indices_y[:num_freq]
#     else:
#         raise NotImplementedError
#     return mapper_x,mapper_y
#
# class MultiSpectralAttentionLayer(torch.nn.Module):
#     def __init__(self, channel, dct_h, dct_w, reduction = 16, freq_sel_method = 'top16'):
#         super(MultiSpectralAttentionLayer, self).__init__()
#         self.reduction = reduction
#         self.dct_h = dct_h
#         self.dct_w = dct_w
#
#         mapper_x, mapper_y = get_freq_indices(freq_sel_method)
#         self.num_split = len(mapper_x)
#         mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
#         mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
#         # make the frequencies in different sizes are identical to a 7x7 frequency space
#         # eg, (2,2) in 14x14 is identical to (1,1) in 7x7
#
#         self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         n,c,h,w = x.shape
#         x_pooled = x
#         if h != self.dct_h or w != self.dct_w:
#             x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
#             # If you have concerns about one-line-change, don't worry.   :)
#             # In the ImageNet models, this line will never be triggered.
#             # This is for compatibility in instance segmentation and object detection.
#         y = self.dct_layer(x_pooled)
#
#         y = self.fc(y).view(n, c, 1, 1)
#         #print(x)
#         return x * y.expand_as(x)
#
#
# class MultiSpectralDCTLayer(nn.Module):
#     """
#     Generate dct filters
#     """
#
#     def __init__(self, height, width, mapper_x, mapper_y, channel):
#         super(MultiSpectralDCTLayer, self).__init__()
#
#         assert len(mapper_x) == len(mapper_y)
#         assert channel % len(mapper_x) == 0
#
#         self.num_freq = len(mapper_x)
#
#         # fixed DCT init
#         #self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
#
#         self.weight = nn.Parameter(self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
#
#         # fixed random init
#         # self.register_buffer('weight', torch.rand(channel, height, width))
#
#         # learnable DCT init
#         #self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
#
#         # learnable random init
#         # self.register_parameter('weight', torch.rand(channel, height, width))
#
#         # num_freq, h, w
#
#     def forward(self, x):
#         assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
#         n, c, h, w = x.shape
#
#         #x = x * self.weight
#
#         result = torch.sum(x, dim=[2, 3])
#         return result
#
#     def build_filter(self, pos, freq, POS):
#         result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
#         if freq == 0:
#             return result
#         else:
#             return result * math.sqrt(2)
#
#     def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
#         dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)
#
#         c_part = channel // len(mapper_x)
#
#         for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
#             for t_x in range(tile_size_x):
#                 for t_y in range(tile_size_y):
#                     dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x,
#                                                                                            tile_size_x) * self.build_filter(
#                         t_y, v_y, tile_size_y)
#
#         return