# ჯერ გავაკეთებ ყველაზე მარტივ მოდელს
# მინიმალური რაოდენობის ლეიერიან cnn-ს
# 1 layeristvis
#maximalurad shevecade animal faceis msgavsad damewera...

import torch
import torch.nn as nn
import torch.nn.functional as plw  

m_zm = 4
k_cons = 48
dim_arr = [1, 16, 32]
emotion_const = 7
whole_size_cons = 64*12 *12
fc_const_mine = 128

class SingleCNN(nn.Module):
    def __init__(self, num_classes=emotion_const):
        super(SingleCNN, self).__init__()  
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.fc1 = nn.Linear(whole_size_cons, fc_const_mine)
        self.fc2 = nn.Linear(fc_const_mine, num_classes)
        
    def forward(self, x):
        kv = self.conv1(x)
        kv2 = plw.relu(kv)  
        x = self.pool1(kv2)
        x = self.pool2(x)
        inp_br_x = x.size(0)
        x = x.view(inp_br_x, -1)
        x = plw.relu(self.fc1(x))     
        x = self.fc2(x) 
        
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
            'model_name': 'SingleCNN',  
            'details': '1 layered CNN - Minimal architecture',  
            'my_prediction': 'Will severely underfit - too simple',
            'all_params': self.parameters_to_train(),
            'layers used': ['Conv2d(1->64, 3x3)','MaxPool2d(2x2)', 'MaxPool2d(2x2)','Linear(9216->128)','Linear(128->7)'],
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

def build_SingleCNN(num_classes=emotion_const):
    return SingleCNN(num_classes)  

def chemi_preds_final():
    print("1 ლეიერიანი CNN-ისთვის")
    print("ეს არის clean version dropout-ის გარეშე") 
    print("ამ მოდელის შემთვხევაში აუცილებლად მოხდება severe underfitting")
    print("მთავარი დანიშნულებაა ვანახო, რომ უფრო კომპლექსური მოდელი მჭირდება")
    print("საუკეთესო baseline იქნება უფრო ღრმა არქიტექტურების საჩვენებლად")
    



if __name__ == "__main__":
    print("Single layer CNN is loading...")
    curr_m = build_SingleCNN()
    display_details(curr_m)
    curr_in = torch.randn(m_zm, 1, k_cons, k_cons)
    my_lv = pr_ind(curr_in, curr_m)  
    display_shedegi(curr_in, my_lv)  
    my_zom_pr(curr_m)  
    chemi_preds_final()