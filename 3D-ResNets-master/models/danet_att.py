from torch import nn
import torch
from torch.nn import Module, Parameter, Softmax

class DANet_Attention(nn.Module):
    def __init__(self, in_dim):
        super(DANet_Attention, self).__init__()
        self.channel_in = in_dim
        self.pam = PAM_Module(in_dim)
        self.cam = CAM_Module(in_dim)
        self.tim = TIM_Moudle(in_dim)

    def forward(self, x):
        # print("进入注意力:", x.shape)
        pam_out = self.pam(x)
        cam_out = self.cam(x)
        tim_out = self.tim(x)
        out = pam_out + cam_out + tim_out
        # print("出去注意力:", out.shape)
        return out


class PAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X T X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        # print('x', x.shape)
        m_batchsize, C, T, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        # print('out',out.shape)
        out = out.view(m_batchsize, -1, height, width).view(m_batchsize, C, T, height, width)

        out = self.gamma*out + x
        return out


class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, T, height, width = x.size()
        proj_query = x.view(m_batchsize, C, T, -1).view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, T, -1).view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, T, -1).view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, T, -1).view(m_batchsize, C, T, height, width)

        out = self.gamma*out + x
        return out

class TIM_Moudle(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(TIM_Moudle, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        # 转换 x=(b, t, c, h , w)
        x_1 = x
        x = x.permute(0, 2, 1, 3, 4)
        m_batchsize, T, C, height, width = x.size()
        proj_query = x.contiguous().view(m_batchsize, T, C, height*width).view(m_batchsize, T, -1)
        proj_key = x.contiguous().view(m_batchsize, T, C, height*width).view(m_batchsize, T, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.contiguous().view(m_batchsize, T, C, -1).view(m_batchsize, T, -1)

        out = torch.bmm(attention, proj_value)
        out = out.contiguous().view(m_batchsize, T, C, -1).view(m_batchsize, T, C, height, width)
        out = out.permute(0, 2, 1, 3, 4)

        out = self.gamma*out + x_1
        return out