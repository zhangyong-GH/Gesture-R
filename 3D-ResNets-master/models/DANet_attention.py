from torch import nn
import torch

class DANet_Attention(nn.Module):
    def __init__(self, in_dim, bn_dim):
        super(DANet_Attention, self).__init__()
        self.channel_in = in_dim
        self.pam = PAM_Module(in_dim)
        self.cam = CAM_Module(in_dim)
        self.tim = TIM_Moudle(in_dim)
        self.bn = nn.BatchNorm3d(bn_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # print("x_shape:", x.shape)
        pam_out = self.pam(x)
        cam_out = self.cam(x)
        out = pam_out + cam_out
        # tim_out = self.tim(cam_out)
        # tim_out = self.bn(tim_out)
        # tim_out = self.relu(tim_out)
        # print("cam_out:", cam_out.shape)
        return pam_out


class PAM_Module(nn.Module):
    # 空间注意力模块
    def __init__(self , in_dim):
        super(PAM_Module, self).__init__()
        self.channel_in = in_dim


        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim //8 , kernel_size =1 )
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim // 8 , kernel_size = 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size = 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim = -1)
    def forward(self , x):
        # print('in_dim', self.channel_in)
        # print(x.shape)
        m_batch, m_c, m_time, m_h, m_w = x.size()
        # print('m_batch, m_time, m_c, m_h, m_w:', m_batch, m_time, m_c, m_h, m_w)
        """
        x_shape: torch.Size([128, 3, 16, 112, 112])---->[128, 3*16, 112, 112]
        """
        x = x.view(m_batch, -1, m_h, m_w)
        x = x.squeeze(-1)
        # print(x.shape)
        m_batchsize , C,height , width = x.size()
        # print("m_batchsize :",m_batchsize , "  C:",C , " heigth:",height , " width:" , width)
        # permute:维度换位
        # proj_query: (1,60,9,9) -> (1,7,9,9) -> (1,7,81) -> (1,81,7)
        proj_query = self.query_conv(x).view(m_batchsize , -1 , width*height).permute(0,2,1)
        # print("proj_equery : " , proj_query.shape)
        # proj_key: (1,60,9,9) -> (1,7,9,9) -> (1,7,81)
        proj_key = self.key_conv(x).view(m_batchsize , -1 , width*height)
        # print("proj_key:" , proj_key.shape)
        # energy : (1 , 81 , 81) 空间位置上每个位置相对与其他位置的注意力energy
        energy = torch.bmm(proj_query , proj_key)
        attention = self.softmax(energy) #对第三个维度求softmax，某一维度的所有行求softmax
        proj_value = self.value_conv(x).view(m_batchsize , -1 , width*height)
        # print("proj_value : " , proj_value.shape)
        #proj_value : （1,60,81） attetnion:(1,81,81) -> (1,60,81)
        out = torch.bmm(proj_value , attention.permute(0,2,1)) #60行81列，每一行81个元素都是每个元素对其他位置的注意力权重乘以value后的值
        out = out.view(m_batchsize , C , height , width)
        out = (self.gamma*out + x).view(m_batch, m_c, m_time, m_h, m_w)
        # print("pam_out_shape", out.shape)
        return out

class CAM_Module(nn.Module):
    # 通道注意力模块
    def __init__(self , in_dim) :
        super(CAM_Module, self).__init__()
        self.channel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1)) #可学习参数
        self.softmax = torch.nn.Softmax(dim = -1)
    def forward(self , x):
        m_batch, m_c, m_time, m_h, m_w = x.size()
        # print('m_batch, m_time, m_c, m_h, m_w:', m_batch, m_time, m_c, m_h, m_w)
        """
        x_shape: torch.Size([128, 3, 16, 112, 112])---->[128, 3*16, 112, 112]
        """
        x = x.view(m_batch, m_c, m_time, -1)
        # print("x_cam_shape:", x)
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = (self.gamma * out + x).view(m_batch, m_c, m_time, m_h, m_w)

        return out

# 时间自注意力
class TIM_Moudle(nn.Module):
    def __init__(self, in_dim):
        super(TIM_Moudle, self).__init__()
        self.channel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))  # 可学习参数
        self.softmax = torch.nn.Softmax(dim=-1)


    def forward(self , x):
        # 只需将chanel与time位置互换就行
        # print("x转换前：", x.shape)
        x = x.permute(0, 2, 1, 3, 4)
        # print("x转换后：", x.shape)
        m_batch, m_time, m_c, m_h, m_w = x.size()
        # print('m_batch, m_time, m_c, m_h, m_w:', m_batch, m_time, m_c, m_h, m_w)
        """
        x_shape: torch.Size([128, 3, 16, 112, 112])---->[128, 3*16, 112, 112]
        """
        x = x.view(m_batch, m_time, m_c, -1)
        # print("x_cam_shape:", x.shape)
        m_batchsize, T, c, h_width = x.size()
        # x.contiguous()如果不加，就是维度过大，张量分开存放会
        """
        RuntimeError: invalid argument 2: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Call .contiguous() before .view(). at /pytorch/aten/src/THC/generic/THCTensor.cpp:209
        原因：用多卡训练的时候tensor不连续，即tensor分布在不同的内存或显存中。
        版权声明：本文为CSDN博主「hyliuisme」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
        原文链接：https://blog.csdn.net/lhyyhlfornew/article/details/103338932
        """
        proj_query = x.contiguous().view(m_batchsize, T, -1)
        proj_key = x.contiguous().view(m_batchsize, T, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.contiguous().view(m_batchsize, T, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, T, c, -1)

        out = (self.gamma * out + x).view(m_batch, m_time, m_c, m_h, m_w).permute(0, 2, 1, 3, 4)
        return out