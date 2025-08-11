import torch
import torch.nn as nn
import torch.nn.functional as plw
from torchvision import transforms

m_zm = 4
k_cons = 48
dim_arr = [1, 16, 32]
emotion_const = 7
# 512 * 6 * 6 = 18432
whole_size_cons = 512 * 6 * 6
fc_const_mine = 512  

class FiveLayerCNNWithAugmentation(nn.Module):
    def __init__(self, num_classes=emotion_const):
        super(FiveLayerCNNWithAugmentation, self).__init__()  
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 48->24
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 24->12
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 12->6
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(whole_size_cons, fc_const_mine)
        self.fc2 = nn.Linear(fc_const_mine, 128)  
        self.fc3 = nn.Linear(128, num_classes)
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=15),  
            transforms.RandomHorizontalFlip(p=0.5),  
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  
            transforms.ToTensor(),  
        ])
        
    def forward(self, x):
        if self.training:
            batch_size = x.size(0)
            device = x.device  
            augmented_batch = []
            
            for i in range(batch_size):
               
                single_img = x[i]  # (1, 48, 48)
                single_img_cpu = single_img.cpu()
                single_img_2d = single_img_cpu.squeeze(0)  # (48, 48)
                augmented_img = self.train_transform(single_img_2d)
                augmented_img = augmented_img.to(device)
                augmented_batch.append(augmented_img)
                
                
            
            
            x = torch.stack(augmented_batch, dim=0)
        
        # chemi dzveli kodi
        # 1
        kv1 = self.conv1(x)
        kv1_relu = plw.relu(kv1)  
        x = self.pool1(kv1_relu)
        # 2
        kv2 = self.conv2(x)
        kv2_relu = plw.relu(kv2)
        x = self.pool2(kv2_relu)
        # 3
        kv3 = self.conv3(x)
        kv3_relu = plw.relu(kv3)
        x = kv3_relu
        # 4
        kv4 = self.conv4(x)
        kv4_relu = plw.relu(kv4)
        x = self.pool4(kv4_relu)
        # 5
        kv5 = self.conv5(x)
        kv5_relu = plw.relu(kv5)
        x = kv5_relu
        
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
            'model_name': 'FiveLayerCNN_WithAugmentation',  
            'details': '5 layered CNN with data augmentation - Should reduce overfitting',  
            'my_prediction': 'Less overfitting than basic 5-layer due to data augmentation regularization',
            'all_params': self.parameters_to_train(),
            'augmentations': ['RandomRotation(±15°)', 'RandomHorizontalFlip(50%)', 
                            'ColorJitter(brightness/contrast)', 'RandomAffine(translation)'],
            'layers used': ['Conv2d(1->32, 3x3)', 'MaxPool2d(2x2)','Conv2d(32->64, 3x3)', 
                          'MaxPool2d(2x2)', 'Conv2d(64->128, 3x3)','Conv2d(128->256, 3x3)', 
                          'MaxPool2d(2x2)','Conv2d(256->512, 3x3)','Linear(18432->512)', 
                          'Linear(512->128)', 'Linear(128->7)'],
        }

def display_details(model):
    my_pars = model.curr_m_params()
    print(f"\nCurrent Model: {my_pars['model_name']}")
    print(f"Model Details: {my_pars['details']}")  
    print(f"Parameters: {my_pars['all_params']:,}")
    print(f"My Prediction: {my_pars['my_prediction']}")
    if 'augmentations' in my_pars:
        print(f"Data Augmentations: {', '.join(my_pars['augmentations'])}")

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

def build_FiveLayerCNNWithAugmentation(num_classes=emotion_const):
    return FiveLayerCNNWithAugmentation(num_classes)  

def chemi_preds_final():
    print("ახლა ვცდი 5 ლეიერიან cnn-ს data augmentation-ით")
    print("data augmentation აძლევს მოდელს მეტ variety-ს training data-ში")
    print("ამან უნდა შეამციროს overfitting და გააუმჯობესოს generalization")
 

if __name__ == "__main__":
    print("Five layer CNN with Data Augmentation is loading...")
    curr_m = build_FiveLayerCNNWithAugmentation()
    display_details(curr_m)
    curr_in = torch.randn(m_zm, 1, k_cons, k_cons)
    my_lv = pr_ind(curr_in, curr_m)
    display_shedegi(curr_in, my_lv)
    my_zom_pr(curr_m)  
    chemi_preds_final()