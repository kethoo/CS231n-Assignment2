# 5 layer cnn + batchnorm (wina experimentis sauketeso modeli)
#+ damatebiti skip connections

import torch
import torch.nn as nn
import torch.nn.functional as plw  

m_zm = 4
k_cons = 48
dim_arr = [1, 16, 32]
emotion_const = 7
SQ1 = 32
SQ2 = 64
SQ3 = 128
SQ4 = 256
SQ5 = 512
whole_size_cons = SQ5 * 6 * 6
fc_const_mine = SQ5

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv(x)
        out = self.bn(out)
        out = plw.relu(out)
        
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = plw.relu(out)
        
        return out

class FiveLayerResNet(nn.Module):
    def __init__(self, num_classes=emotion_const):
        super(FiveLayerResNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=SQ1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(SQ1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 48->24
        self.conv2 = nn.Conv2d(in_channels=SQ1, out_channels=SQ2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(SQ2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 24->12
        self.skip2 = nn.Sequential(
            nn.Conv2d(SQ1, SQ2, kernel_size=1, bias=False),
            nn.BatchNorm2d(SQ2)
        )
        self.conv3 = nn.Conv2d(in_channels=SQ2, out_channels=SQ3, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(SQ3)
        self.skip3 = nn.Sequential(
            nn.Conv2d(SQ2, SQ3, kernel_size=1, bias=False),
            nn.BatchNorm2d(SQ3)
        )
        self.conv4 = nn.Conv2d(in_channels=SQ3, out_channels=SQ4, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(SQ4)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 12->6
        self.skip4 = nn.Sequential(
            nn.Conv2d(SQ3, SQ4, kernel_size=1, bias=False),
            nn.BatchNorm2d(SQ4)
        )
        self.conv5 = nn.Conv2d(in_channels=SQ4, out_channels=SQ5, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(SQ5)
        self.skip5 = nn.Sequential(
            nn.Conv2d(SQ4, SQ5, kernel_size=1, bias=False),
            nn.BatchNorm2d(SQ5)
        )
        self.fc1 = nn.Linear(whole_size_cons, fc_const_mine)
        self.fc2 = nn.Linear(fc_const_mine, SQ3)
        self.fc3 = nn.Linear(SQ3, num_classes)

    def forward(self, x):
        #1 
        kv1 = self.conv1(x)
        kv1_bn = self.bn1(kv1)
        kv1_relu = plw.relu(kv1_bn)
        x = self.pool1(kv1_relu)
        
        #2 
        identity2 = self.skip2(x)  
        kv2 = self.conv2(x)
        kv2_bn = self.bn2(kv2)
        kv2_relu = plw.relu(kv2_bn)
        kv2_pooled = self.pool2(kv2_relu)
        identity2_pooled = self.pool2(identity2)
        x = kv2_pooled + identity2_pooled 
        x = plw.relu(x)
        
        #3 
        identity3 = self.skip3(x)  
        kv3 = self.conv3(x)
        kv3_bn = self.bn3(kv3)
        kv3_relu = plw.relu(kv3_bn)
        x = kv3_relu + identity3 
        x = plw.relu(x)
        
        #4 
        identity4 = self.skip4(x)  
        kv4 = self.conv4(x)
        kv4_bn = self.bn4(kv4)
        kv4_relu = plw.relu(kv4_bn)
        kv4_pooled = self.pool4(kv4_relu)
        identity4_pooled = self.pool4(identity4)
        x = kv4_pooled + identity4_pooled  
        x = plw.relu(x)
        
        #5 
        identity5 = self.skip5(x) 
        kv5 = self.conv5(x)
        kv5_bn = self.bn5(kv5)
        kv5_relu = plw.relu(kv5_bn)
        x = kv5_relu + identity5  
        x = plw.relu(x)
        inp_br_x = x.size(0)
        x = x.view(inp_br_x, -1)
        x = plw.relu(self.fc1(x))
        x = plw.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

    def parameters_to_train(self):
        total_ans = 0
        for p in self.parameters():
            if p.requires_grad:
                to_dam = p.numel()
                total_ans = total_ans + to_dam
        return total_ans

    def curr_m_params(self):
        return {
            'model_name': 'FiveLayerResNet',
            'details': '5 layer CNN with batch normalization and ResNet-style skip connections',
            'my_prediction': 'Better gradient flow, reduced vanishing gradient, improved training stability',
            'all_params': self.parameters_to_train(),
            'layers used': [
                'Conv2d(1->32, 3x3)', 'BatchNorm2d(32)', 'MaxPool2d(2x2)',
                'Conv2d(32->64, 3x3) + Skip(32->64)', 'BatchNorm2d(64)', 'MaxPool2d(2x2)',
                'Conv2d(64->128, 3x3) + Skip(64->128)', 'BatchNorm2d(128)',
                'Conv2d(128->256, 3x3) + Skip(128->256)', 'BatchNorm2d(256)', 'MaxPool2d(2x2)',
                'Conv2d(256->512, 3x3) + Skip(256->512)', 'BatchNorm2d(512)',
                'Linear(18432->512)', 'Linear(512->128)', 'Linear(128->7)'
            ],
            'skip_connections': [
                'Layer2: Conv1x1(32->64) for dimension matching',
                'Layer3: Conv1x1(64->128) for dimension matching', 
                'Layer4: Conv1x1(128->256) for dimension matching',
                'Layer5: Conv1x1(256->512) for dimension matching'
            ]
        }

def display_details(model):
    my_pars = model.curr_m_params()
    print(f"\nCurrent Model: {my_pars['model_name']}")
    print(f"Model Details: {my_pars['details']}")
    print(f"Parameters: {my_pars['all_params']:,}")
    print(f"My Prediction: {my_pars['my_prediction']}")
    print("\nSkip Connections:")
    for skip in my_pars['skip_connections']:
        print(f"  - {skip}")

def display_shedegi(pr1, pr2):
    print(f"\nTesting Forward Pass:")
    print(f"Before transform size: {pr1.shape}")
    print(f"After transform size: {pr2.shape}")

def my_zom_pr(model):
    for dim_vim in dim_arr:
        curr_in = torch.randn(dim_vim, 1, k_cons, k_cons)
        with torch.no_grad():
            my_lv = model(curr_in)
        expected_shape = (dim_vim, emotion_const)
        actual_shape = my_lv.shape
        if actual_shape != expected_shape:
            raise ValueError(f"Expected {expected_shape}, got {actual_shape}")

def pr_ind(input_tensor, model):
    with torch.no_grad():
        output = model(input_tensor)
    return output

def build_FiveLayerResNet(num_classes=emotion_const):
    return FiveLayerResNet(num_classes)

def build_FiveLayerBatchNormSkipping(num_classes=emotion_const):
    return FiveLayerResNet(num_classes)
    
def chemi_preds_final():
    print("5 ლეიერიან Batchnorm CNN-ს დავუმატე ResNet-style skip connections")
    print("გვაქვს საშუალება:")
    print("- უკეთეს gradient flow-ს deep network-ში")
    print("- vanishing gradient problem-ის შემცირებას")
    print("- უფრო სტაბილურ training-ს")
    print("- identity mapping-ის შესაძლებლობას")
    
if __name__ == "__main__":
    print("Five layer ResNet-style CNN is loading...")
    curr_m = build_FiveLayerResNet()
    display_details(curr_m)
    curr_in = torch.randn(m_zm, 1, k_cons, k_cons)
    my_lv = pr_ind(curr_in, curr_m)
    display_shedegi(curr_in, my_lv)
    my_zom_pr(curr_m)
    chemi_preds_final()