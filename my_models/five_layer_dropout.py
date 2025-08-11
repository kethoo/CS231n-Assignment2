# 5-layer CNN + Dropout
# ეს არის 5 ლეიერიანი cnn dropout-ით overfitting-ის თავიდან ასაცილებლად
import torch
import torch.nn as nn
import torch.nn.functional as plw  

m_zm = 4
k_cons = 48
dim_arr = [1, 16, 32]
emotion_const = 7
# 512 * 6 * 6 = 18432
whole_size_cons = 512*6*6
fc_const_mine = 512  

class FiveLayerCNNWithDropout(nn.Module):
    def __init__(self, num_classes=emotion_const):
        super(FiveLayerCNNWithDropout, self).__init__()  
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 48->24
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 24->12
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 12->6
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.dropout_conv = nn.Dropout2d(p=0.25)  
        self.dropout_fc1 = nn.Dropout(p=0.5)      
        self.dropout_fc2 = nn.Dropout(p=0.3)      
        self.fc1 = nn.Linear(whole_size_cons, fc_const_mine)
        self.fc2 = nn.Linear(fc_const_mine, 128)  
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        #1
        kv1 = self.conv1(x)
        kv1_relu = plw.relu(kv1)  
        x = self.pool1(kv1_relu)
        
        #2
        kv2 = self.conv2(x)
        kv2_relu = plw.relu(kv2)
        x = self.pool2(kv2_relu)
        
        #3
        kv3 = self.conv3(x)
        kv3_relu = plw.relu(kv3)
        x = kv3_relu
        
        #4
        kv4 = self.conv4(x)
        kv4_relu = plw.relu(kv4)
        x = self.pool4(kv4_relu)
        
        #5
        kv5 = self.conv5(x)
        kv5_relu = plw.relu(kv5)
        x = self.dropout_conv(kv5_relu) 
        inp_br_x = x.size(0)
        x = x.view(inp_br_x, -1)
        x = plw.relu(self.fc1(x))
        x = self.dropout_fc1(x)  
        x = plw.relu(self.fc2(x))
        x = self.dropout_fc2(x) 
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
            'model_name': 'FiveLayerCNN_WithDropout',  
            'details': '5 layered CNN with dropout regularization to prevent overfitting',  
            'my_prediction': 'Much less overfitting than basic 5-layer, but may train slower',
            'all_params': self.parameters_to_train(),
            'dropout_rates': ['Conv Dropout: 25%', 'FC1 Dropout: 50%', 'FC2 Dropout: 30%'],
            'layers used': ['Conv2d(1->32, 3x3)', 'MaxPool2d(2x2)','Conv2d(32->64, 3x3)', 
                          'MaxPool2d(2x2)', 'Conv2d(64->128, 3x3)','Conv2d(128->256, 3x3)', 
                          'MaxPool2d(2x2)','Conv2d(256->512, 3x3)', 'Dropout2d(25%)',
                          'Linear(18432->512)', 'Dropout(50%)', 'Linear(512->128)', 
                          'Dropout(30%)', 'Linear(128->7)'],
        }

def display_details(model):
    my_pars = model.curr_m_params()
    print(f"\nCurrent Model: {my_pars['model_name']}")
    print(f"Model Details: {my_pars['details']}")  
    print(f"Parameters: {my_pars['all_params']:,}")
    print(f"My Prediction: {my_pars['my_prediction']}")
    if 'dropout_rates' in my_pars:
        print(f"Dropout Configuration: {', '.join(my_pars['dropout_rates'])}")

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

def build_FiveLayerCNNWithDropout(num_classes=emotion_const):
    return FiveLayerCNNWithDropout(num_classes)  

def chemi_preds_final():
    print("ახლა ვცდით 5 ლეიერიან cnn-ს dropout-ით")
    print("dropout randomly zeros out neurons during training")
    print("ეს ხელს უშლის overfitting-ს და აუმჯობესებს generalization-ს")
    
if __name__ == "__main__":
    print("Five layer CNN with Dropout is loading...")
    curr_m = build_FiveLayerCNNWithDropout()
    display_details(curr_m)
    curr_in = torch.randn(m_zm, 1, k_cons, k_cons)
    my_lv = pr_ind(curr_in, curr_m)
    display_shedegi(curr_in, my_lv)
    my_zom_pr(curr_m)  
    chemi_preds_final()