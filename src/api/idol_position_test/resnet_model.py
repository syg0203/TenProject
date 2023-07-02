import torch.nn as nn

def conv_start():
    #print('conv_start')
    return nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=4),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
    )

def bottleneck_block(in_dim, mid_dim, out_dim, down=False):
    #print('bottleneck_block')
    layers = []
    if down:
        layers.append(nn.Conv2d(in_dim, mid_dim, kernel_size=1, stride=2, padding=0))
    else:
        layers.append(nn.Conv2d(in_dim, mid_dim, kernel_size=1, stride=1, padding=0))
    #print('bottleneck_block_2')
    layers.extend([
        nn.BatchNorm2d(mid_dim),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_dim, mid_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(mid_dim),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_dim, out_dim, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(out_dim),
    ])
    #print('bottleneck_block_3')
    return nn.Sequential(*layers)

class Bottleneck(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, down:bool = False, starting:bool=False) -> None:
        #print('Bottleneck')
        super(Bottleneck, self).__init__()
        if starting:
            down = False
        self.block = bottleneck_block(in_dim, mid_dim, out_dim, down=down)
        self.relu = nn.ReLU(inplace=True)
        #print('Bottleneck_2')
        if down:
            conn_layer = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=2, padding=0)
        else:
            conn_layer = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)
        #print('Bottleneck_3')
        self.changedim = nn.Sequential(conn_layer, nn.BatchNorm2d(out_dim))
        #print('Bottleneck_4')

    def forward(self, x):
        identity = self.changedim(x)
        x = self.block(x)
        x += identity
        x = self.relu(x)
        return x

def make_layer(in_dim, mid_dim, out_dim, repeats, starting=False):
        layers = []
        layers.append(Bottleneck(in_dim, mid_dim, out_dim, down=True, starting=starting))
        for _ in range(1, repeats):
            layers.append(Bottleneck(out_dim, mid_dim, out_dim, down=False))
        return nn.Sequential(*layers)

class ResNet(nn.Module):
    def __init__(self, repeats:list = [3,4,6,3], num_classes=36):
        super(ResNet, self).__init__()
        self.num_classes = num_classes
        self.conv1 = conv_start()
        
        base_dim = 64
        self.conv2 = make_layer(base_dim, base_dim, base_dim*4, repeats[0], starting=True)
        self.conv3 = make_layer(base_dim*4, base_dim*2, base_dim*8, repeats[1])
        self.conv4 = make_layer(base_dim*8, base_dim*4, base_dim*16, repeats[2])
        self.conv5 = make_layer(base_dim*16, base_dim*8, base_dim*32, repeats[3])
        
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.classifer1 = nn.Linear(2048, 1024)
        self.dropuot_1 = nn.Dropout(p=0.5)
        self.classifer2 = nn.Linear(1024, 512)
        self.dropuot_2 = nn.Dropout(p=0.4)
        self.classifer3 = nn.Linear(512, 256)
        self.dropuot_3 = nn.Dropout(p=0.3)
        self.classifer4 = nn.Linear(256, 128)
        self.dropuot_4 = nn.Dropout(p=0.2)        
        self.classifer5 = nn.Linear(128, 64)
        self.classifer_out = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifer1(x)
        x = self.dropuot_1(x)
        x = self.classifer2(x)
        x = self.dropuot_2(x)
        x = self.classifer3(x)
        x = self.dropuot_3(x)
        x = self.classifer4(x)
        x = self.dropuot_4(x)
        x = self.classifer5(x)
        x = self.classifer_out(x)
        return x
    