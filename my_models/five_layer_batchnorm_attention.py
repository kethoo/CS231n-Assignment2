#  5-layer CNN + BatchNorm + Attention

import torch
import torch.nn as nn
import torch.nn.functional as plw  

m_zm = 4
k_cons = 48
dim_arr = [1, 16, 32]
emotion_const = 7
SQ1=32
SQ2=64
SQ3=128
SQ4=256
SQ5=512
whole_size_cons = SQ5 * 6 * 6
fc_const_mine = SQ5

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        dim_k_1 = in_channels // reduction_ratio
        self.fc = nn.Sequential(
          nn.Linear(in_channels, dim_k_1, bias=False),
          nn.ReLU(),
          nn.Linear(dim_k_1, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, san1, san2 = x.size()
        msash = self.avg_pool(x).view(b, c)
        msash = self.fc(msash)
        hg_pr = self.max_pool(x).view(b, c)
        hg_pr = self.fc(hg_pr)
        added_gar = msash + hg_pr
        attention = self.sigmoid(added_gar).view(b, c, 1, 1)
        my_final_ans = x * attention.expand_as(x)
        return my_final_ans

class FiveLayerBatchNormAttention(nn.Module):
    def __init__(self, num_classes=emotion_const):
        super(FiveLayerBatchNormAttention, self).__init__()  
        
        #  1: 
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=SQ1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(SQ1)  
        self.att1 = ChannelAttention(SQ1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  
        
        #2: 
        self.conv2 = nn.Conv2d(in_channels=SQ1, out_channels=SQ2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(SQ2)  
        self.att2 = ChannelAttention(SQ2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  
        
        #3: 
        self.conv3 = nn.Conv2d(in_channels=SQ2, out_channels=SQ3, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(SQ3)  
        self.att3 = ChannelAttention(SQ3)
        
        #4: 
        self.conv4 = nn.Conv2d(in_channels=SQ3, out_channels=SQ4, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(SQ4)  
        self.att4 = ChannelAttention(SQ4)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  
        
        #5: 
        self.conv5 = nn.Conv2d(in_channels=SQ4, out_channels=SQ5, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(SQ5)  
        self.att5 = ChannelAttention(SQ5)
        self.fc1 = nn.Linear(whole_size_cons, fc_const_mine)
        self.fc2 = nn.Linear(fc_const_mine, SQ3)
        self.fc3 = nn.Linear(SQ3, num_classes)

    def forward(self, x):
        kv1 = self.conv1(x)
        kv1_bn = self.bn1(kv1)      
        kv1_relu = plw.relu(kv1_bn)
        kv1_att = self.att1(kv1_relu) 
        x = self.pool1(kv1_att)
        kv2 = self.conv2(x)
        kv2_bn = self.bn2(kv2)      
        kv2_relu = plw.relu(kv2_bn)
        kv2_att = self.att2(kv2_relu)
        x = self.pool2(kv2_att)
        kv3 = self.conv3(x)
        kv3_bn = self.bn3(kv3)      
        kv3_relu = plw.relu(kv3_bn)
        kv3_att = self.att3(kv3_relu)
        x = kv3_att
        kv4 = self.conv4(x)
        kv4_bn = self.bn4(kv4)    
        kv4_relu = plw.relu(kv4_bn)
        kv4_att = self.att4(kv4_relu)
        x = self.pool4(kv4_att)
        kv5 = self.conv5(x)
        kv5_bn = self.bn5(kv5)     
        kv5_relu = plw.relu(kv5_bn)
        kv5_att = self.att5(kv5_relu)
        x = kv5_att
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
            'model_name': 'FiveLayerBatchNormAttention',  
            'details': 'Phase 3: 5-layer CNN with BatchNorm + Channel Attention - Best of both worlds',  
            'my_prediction': 'Should significantly outperform Phase 2 BatchNorm-only model (68-72% expected)',
            'all_params': self.parameters_to_train(),
            'optimizations': [
                'BatchNorm after each conv layer (from Phase 2 winner)', 
                'Channel attention after each layer',
                'No dropout (Phase 2 showed BatchNorm alone was better)'
            ],
            'layers used': [
                'Conv2d(1->32, 3x3)', 'BatchNorm2d(32)', 'ChannelAttention(32)', 'MaxPool2d(2x2)',
                'Conv2d(32->64, 3x3)', 'BatchNorm2d(64)', 'ChannelAttention(64)', 'MaxPool2d(2x2)', 
                'Conv2d(64->128, 3x3)', 'BatchNorm2d(128)', 'ChannelAttention(128)',
                'Conv2d(128->256, 3x3)', 'BatchNorm2d(256)', 'ChannelAttention(256)', 'MaxPool2d(2x2)',
                'Conv2d(256->512, 3x3)', 'BatchNorm2d(512)', 'ChannelAttention(512)',
                'Linear(18432->512)', 'Linear(512->128)', 'Linear(128->7)'
            ],
        }

def display_details(model):
    my_pars = model.curr_m_params()
    print(f"\nCurrent Model: {my_pars['model_name']}")
    print(f"Model Details: {my_pars['details']}")  
    print(f"Parameters: {my_pars['all_params']:,}")
    print(f"My Prediction: {my_pars['my_prediction']}")
    if 'optimizations' in my_pars:
        print("Optimizations:")
        for opt in my_pars['optimizations']:
            print(f"  - {opt}")

def display_shedegi(pr1, pr2):
    print(f"\nTesting Forward Pass:")
    print(f"Before transform size: {pr1.shape}")
    print(f"After transform size: {pr2.shape}")

def my_zom_pr(model):  
    model.eval()
    for dim_vim in dim_arr:
        curr_in = torch.randn(dim_vim, 1, k_cons, k_cons)
        with torch.no_grad():
            my_lv = model(curr_in)  
        expected_shape = (dim_vim, emotion_const)
        actual_shape = my_lv.shape
        if actual_shape != expected_shape:
            raise ValueError(f"Expected {expected_shape}, got {actual_shape}")

def pr_ind(input_tensor, model):
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    return output

def build_FiveLayerBatchNormAttention(num_classes=emotion_const):
    return FiveLayerBatchNormAttention(num_classes)  

def chemi_preds_final():
    print("Channel Attention კი ეხმარება მოდელს ")
    print("ყურადღება გაამახვილოს მნიშვნელოვან ფიჩერებზე")


if __name__ == "__main__":
    print("Five layer CNN with BatchNorm + Channel Attention is loading...")
    curr_m = build_FiveLayerBatchNormAttention()
    display_details(curr_m)
    curr_in = torch.randn(m_zm, 1, k_cons, k_cons)
    my_lv = pr_ind(curr_in, curr_m)
    display_shedegi(curr_in, my_lv)
    my_zom_pr(curr_m)  
    chemi_preds_final()