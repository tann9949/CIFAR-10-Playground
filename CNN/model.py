# import neccessary libraries
import torch.nn as nn
import torch.nn.functional as F

class VGG_lite(nn.Module):
    '''
    Implemented from VGG (https://arxiv.org/pdf/1409.1556.pdf) model B 
    with lesser number of channels. Batch normalization is added together with
    global average pooling layer instead of fully connected+dropout layers
    '''
    def __init__(self):
        super(VGG_lite, self).__init__()

        # defines layers
        self.conv1a = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.batchnorm1a = nn.BatchNorm2d(16)
        self.conv1b = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.batchnorm1b = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d((2,2), stride=(2,2))
       
        
        self.conv2a = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.batchnorm2a = nn.BatchNorm2d(32)
        self.conv2b = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.batchnorm2b = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d((2,2), stride=(2,2))
        
        self.conv3a = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.batchnorm3a = nn.BatchNorm2d(64)
        self.conv3b = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.batchnorm3b = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d((2,2), stride=(2,2))
        
        self.conv4a = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.batchnorm4a = nn.BatchNorm2d(128)
        self.conv4b = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.batchnorm4b = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d((2,2), stride=(2,2))
        
        
        self.logits = nn.Linear(128, 10)
        
    # define a forward function for model
    # this function connects the layers defined above
    def forward(self, inp):
        x = F.relu(self.batchnorm1a(self.conv1a(inp)))
        x = F.relu(self.batchnorm1b(self.conv1b(x)))
        x = self.pool1(x)

        x = F.relu(self.batchnorm2a(self.conv2a(x)))
        x = F.relu(self.batchnorm2b(self.conv2b(x)))
        x = self.pool2(x)

        x = F.relu(self.batchnorm3a(self.conv3a(x)))
        x = F.relu(self.batchnorm3b(self.conv3b(x)))
        x = self.pool3(x)

        x = F.relu(self.batchnorm4a(self.conv4a(x)))
        x = F.relu(self.batchnorm4b(self.conv4b(x)))
        x = self.pool4(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(-1, 128)
        
        out = self.logits(x)
        return out