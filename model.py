import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()

        self.relu = nn.ReLU()

        # --------------------------
        #第一次
        # 输入：torch.Size([4, 1, 85, 85])
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        # 返回值：torch.Size([4, 1, 255, 255])

        # --------------------------
        #第二次
        # 输入：torch.Size([4, 1, 60, 60])
        self.conv1_2 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2_2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3_2 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4_2 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle_2 = nn.PixelShuffle(upscale_factor)
        # 返回值：torch.Size([4, 1, 180, 180])

        # --------------------------
        # 第三次
        # 输入：torch.Size([4, 1, 40, 40])
        self.conv1_3 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2_3 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3_3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4_3 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle_3 = nn.PixelShuffle(upscale_factor)
        # 返回值：torch.Size([4, 1, 120, 120])

        self._initialize_weights()  # 初始化权重

    def forward(self, x):
        # x的大小：torch.Size([4, 1, 85, 85])
        #对x进行下采样
        x_min = F.interpolate(x, size=[60, 60])# x_min的大小：torch.Size([4, 1, 60, 60])
        x_min_min = F.interpolate(x_min, size=[40,40])# x_min_min的大小：torch.Size([4, 1, 40, 40])

        #-------------------
        # 第一次
        # 输入x：torch.Size([4, 1, 85, 85])

        x = self.relu(self.conv1(x))
        # 第一次卷积后：torch.Size([4, 64, 85, 85])

        x = self.relu(self.conv2(x))
        # 第二次卷积后：torch.Size([4, 64, 85, 85])

        x = self.relu(self.conv3(x))
        # 第三次卷积后：torch.Size([4, 32, 85, 85])

        x = self.pixel_shuffle(self.conv4(x))
        # 返回值：torch.Size([4, 1, 255, 255])

        # -------------------
        # 第二次
        # 输入x_min：torch.Size([4, 1, 60, 60])

        x_min = self.relu(self.conv1_2(x_min))
        # 第一次卷积后：torch.Size([4, 64, 60, 60])

        x_min = self.relu(self.conv2_2(x_min))
        # 第二次卷积后：torch.Size([4, 64, 60, 60])

        x_min = self.relu(self.conv3_2(x_min))
        # 第三次卷积后：torch.Size([4, 32, 60, 60])

        x_min = self.pixel_shuffle_2(self.conv4_2(x_min))
        # 返回值：torch.Size([4, 1, 180, 180])

        # -------------------
        # 第三次
        # 输入x_min_min：torch.Size([4, 1, 40, 40])

        x_min_min = self.relu(self.conv1_3(x_min_min))
        # 第一次卷积后：torch.Size([4, 64, 40, 40])

        x_min_min = self.relu(self.conv2_3(x_min_min))
        # 第二次卷积后：torch.Size([4, 64, 40, 40])

        x_min_min = self.relu(self.conv3_3(x_min_min))
        # 第三次卷积后：torch.Size([4, 32, 40, 40])

        x_min_min = self.pixel_shuffle_3(self.conv4_3(x_min_min))
        # 返回值：torch.Size([4, 1, 120, 120])

        #对x_min,x_min_min进行上采样
        x_min = F.interpolate(x, size=[255, 255])  # x_min的大小：torch.Size([4, 1, 255, 255])
        x_min_min = F.interpolate(x_min, size=[255, 255])  # x_min_min的大小：torch.Size([4, 1, 255, 255])

        #返回三者之和
        return x + x_min + x_min_min

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)
