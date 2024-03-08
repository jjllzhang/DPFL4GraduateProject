import torch
import torch.nn as nn
import torch.nn.functional as F

class FMNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(8, 2), stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 2), stride=1, padding=0)

        # 使用一个辅助函数来计算卷积层输出的大小
        fc_input_size = self._get_conv_output((1, 28, 28))

        self.fc1 = nn.Linear(fc_input_size, 32)
        self.fc2 = nn.Linear(32, 10)

    def _get_conv_output(self, shape):
        # 使用假数据通过卷积层来确定全连接层的输入大小
        input = torch.autograd.Variable(torch.rand(1, *shape))
        output = self.conv1(input)
        output = self.conv2(output)
        n_size = output.data.view(1, -1).size(1)
        return n_size
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # 展平操作
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
