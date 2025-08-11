#ამჟამად 5 ლეიერიანს სქიფინგი ჩავუმატე
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

class SkipConnection(nn.Module):
    def choose_gam(self, x):
        conv_b = self.skip_conv
        ans1 = self.skip_conv(x)
        ans2 = x
        if conv_b is not None:
            return ans1
        return ans2

    def aarchie_from_shem(self, in_ch, out_ch, nab):
        bool1 = in_ch != out_ch
        bool2 = nab != 1
        if bool1 or bool2:
            return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=nab, bias=False)
        return None

    def __init__(self, in_ch, out_ch, st=1):
        super(SkipConnection, self).__init__()
        self.skip_conv = self.aarchie_from_shem(in_ch, out_ch, st)
        
    def forward(self, x, shed_conv):
        gam = self.choose_gam(x)
        ans_fin = gam + shed_conv
        return ans_fin

class FiveLayerSkipConnections(nn.Module):
    def __init__(self, num_classes=emotion_const):
        super(FiveLayerSkipConnections, self).__init__()  
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=SQ1, kernel_size=3, padding=1)
        self.skip1 = SkipConnection(1, SQ1)  
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  
        self.conv2 = nn.Conv2d(in_channels=SQ1, out_channels=SQ2, kernel_size=3, padding=1)
        self.skip2 = SkipConnection(SQ1, SQ2)  
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  
        self.conv3 = nn.Conv2d(in_channels=SQ2, out_channels=SQ3, kernel_size=3, padding=1)
        self.skip3 = SkipConnection(SQ2, SQ3)  
        self.conv4 = nn.Conv2d(in_channels=SQ3, out_channels=SQ4, kernel_size=3, padding=1)
        self.skip4 = SkipConnection(SQ3, SQ4)  
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  
        self.conv5 = nn.Conv2d(in_channels=SQ4, out_channels=SQ5, kernel_size=3, padding=1)
        self.skip5 = SkipConnection(SQ4, SQ5)  
        self.dropout = nn.Dropout(0.25)  
        self.fc1 = nn.Linear(whole_size_cons, fc_const_mine)
        self.fc2 = nn.Linear(fc_const_mine, SQ3)
        self.fc3 = nn.Linear(SQ3, num_classes)
        
        
        
        
        
        
        

    def forward(self, x):
        #1
        x_input1 = x
        kv1 = self.conv1(x)
        kv1_skip = self.skip1(x_input1, kv1)      
        kv1_relu = plw.relu(kv1_skip)  
        x = self.pool1(kv1_relu)
        #2
        x_input2 = x
        kv2 = self.conv2(x)
        kv2_skip = self.skip2(x_input2, kv2)      
        kv2_relu = plw.relu(kv2_skip)
        x = self.pool2(kv2_relu)
        #3
        x_input3 = x
        kv3 = self.conv3(x)
        kv3_skip = self.skip3(x_input3, kv3)      
        kv3_relu = plw.relu(kv3_skip)
        x = kv3_relu
        # 4
        x_input4 = x
        kv4 = self.conv4(x)
        kv4_skip = self.skip4(x_input4, kv4)    
        kv4_relu = plw.relu(kv4_skip)
        x = self.pool4(kv4_relu)
        #5
        x_input5 = x
        kv5 = self.conv5(x)
        kv5_skip = self.skip5(x_input5, kv5)     
        kv5_relu = plw.relu(kv5_skip)
        x = kv5_relu
        inp_br_x = x.size(0)
        x = x.view(inp_br_x, -1)
        x = self.dropout(x)
        x = plw.relu(self.fc1(x))     
        x = self.dropout(x)
        x = plw.relu(self.fc2(x))
        x = self.dropout(x)
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
            'model_name': 'FiveLayerWithSkipConnections',  
            'details': '5 layer cnn with additional skip connections',  
            'my_prediction': 'Better than batch norm version of 5 layer CNN',
            'all_params': self.parameters_to_train(),
            'layers used': [
                'Conv2d(1->32, 3x3)', 'SkipConnection(1->32)', 'MaxPool2d(2x2)','Conv2d(32->64, 3x3)', 'SkipConnection(32->64)', 'MaxPool2d(2x2)', 'Conv2d(64->128, 3x3)', 'SkipConnection(64->128)',
                'Conv2d(128->256, 3x3)', 'SkipConnection(128->256)', 'MaxPool2d(2x2)','Conv2d(256->512, 3x3)', 'SkipConnection(256->512)','Linear(18432->512)', 'Linear(512->128)', 'Linear(128->7)'
            ],
        }

def display_details(model):
    my_pars = model.curr_m_params()
    print(f"\nCurrent Model: {my_pars['model_name']}")
    print(f"Model Details: {my_pars['details']}")  
    print(f"Parameters: {my_pars['all_params']:,}")
    print(f"My Prediction: {my_pars['my_prediction']}")

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

def build_FiveLayerSkipConnections(num_classes=emotion_const):
    return FiveLayerSkipConnections(num_classes)  

def chemi_preds_final():
    print("ამჟამად განვიხილავ 5 ლეიერიან CNN + სქიფინგს და საკმაოდ კარგად იმუშავებს")
    print("vanishing gradient-ს ეს უმკლავდება კარგად და ვფიქრობ, ყველაზე კარგ შედეგს ეს დადებს")


if __name__ == "__main__":
    print("Five layer with additional Skip Connections CNN is loading...")
    curr_m = build_FiveLayerSkipConnections()
    display_details(curr_m)
    curr_in = torch.randn(m_zm, 1, k_cons, k_cons)
    my_lv = pr_ind(curr_in, curr_m)
    display_shedegi(curr_in, my_lv)
    my_zom_pr(curr_m)  
    chemi_preds_final()