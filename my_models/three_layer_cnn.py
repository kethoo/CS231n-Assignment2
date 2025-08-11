#ახლა ვცდი 3 layer-იანს
#უკეთესი შედეგი იქნება, მაგრამ ალბათ მაინც
#საკმაოდ underfit და საკმარისად კარგი არ იქნება
import torch
import torch.nn as nn
import torch.nn.functional as plw  

m_zm = 4
k_cons = 48
dim_arr = [1, 16, 32]
emotion_const = 7
#128 * 12 * 12 = 18432
whole_size_cons = 128 * 12 * 12
fc_const_mine = 256  

class TripleCNN(nn.Module):
    def __init__(self, num_classes=emotion_const):
        super(TripleCNN, self).__init__()  
        #layer 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        #layer 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        #layer 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(whole_size_cons, fc_const_mine)
        self.fc2 = nn.Linear(fc_const_mine, num_classes)
        
        
        
    def forward(self, x):
        kv1 = self.conv1(x)
        kv1_relu = plw.relu(kv1)  
        x = self.pool1(kv1_relu)
        kv2 = self.conv2(x)
        kv2_relu = plw.relu(kv2)
        x = self.pool2(kv2_relu)
        kv3 = self.conv3(x)
        kv3_relu = plw.relu(kv3)
        x = kv3_relu 
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
            'model_name': 'TripleCNN',  
            'details': '3 layered CNN - Not really deep ',  
            'my_prediction': 'Will likely underfit - no regularization is bad',
            'all_params': self.parameters_to_train(),
            'layers used': [
                'Conv2d(1->32, 3x3)', 'MaxPool2d(2x2)','Conv2d(32->64, 3x3)', 'MaxPool2d(2x2)','Conv2d(64->128, 3x3)', 'Linear(18432->256)', 'Linear(256->7)'
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

def build_TripleCNN(num_classes=emotion_const):
    return TripleCNN(num_classes)  

def chemi_preds_final():
    print("3 ლეიერიანი CNN-ისთვის")
    print("ეს არის clean version dropout-ის გარეშე")
    print("ალბათ მაინც underfitting იქნება - ძალიან მარტივი არქიტექტურა")
    print("კარგი საშუალო ვარიანტია 1-ლეიერსა და 5-ლეიერს შორის")
    

if __name__ == "__main__":
    print("Three layer CNN  is loading...")
    curr_m = build_TripleCNN()
    display_details(curr_m)
    curr_in = torch.randn(m_zm, 1, k_cons, k_cons)
    my_lv = pr_ind(curr_in, curr_m)
    display_shedegi(curr_in, my_lv)
    my_zom_pr(curr_m)  
    chemi_preds_final()