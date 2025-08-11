import torch
import numpy as np
import random 

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}")

def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("Using CPU")
    return device

def print_model_info(model, model_name):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f" {model_name} Info:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    return total_params, trainable_params

def display_my_result(model_name, results_dict):
    if model_name not in results_dict:
        print(f"Model '{model_name}' not found in results")
        return
    
    result = results_dict[model_name]
    display_name = model_name.replace('_', ' ').title()
    
    print(" ")
    print(f"Model results : {display_name}")
    print(" ")
    print(f"Final Validation Accuracy: {result['best_val_acc']:.2f}%")
    print(f"Final Training Accuracy:  {result['final_train_acc']:.2f}%")
    if 'val_losses' in result and result['val_losses']:
        final_val_loss = result['val_losses'][-1]  # Last epoch loss
        print(f"Final Validation Loss:    {final_val_loss:.4f}")
    
    if 'train_losses' in result and result['train_losses']:
        final_train_loss = result['train_losses'][-1]  # Last epoch loss
        print(f"Final Training Loss:      {final_train_loss:.4f}")
    
    accuracy_gap = result['final_train_acc'] - result['best_val_acc']
    print(f"Gap in accuracy:            {accuracy_gap:.2f}%")
    if 'val_losses' in result and 'train_losses' in result and result['val_losses'] and result['train_losses']:
        final_val_loss = result['val_losses'][-1]
        final_train_loss = result['train_losses'][-1]
        loss_gap = final_val_loss - final_train_loss
        print(f"Gap in Loss:                {loss_gap:.4f}")
    
    print(f"Total Parameters:        {result['parameters']:,}")
    
    if 'epochs_trained' in result:
        print(f"Epochs Trained:          {result['epochs_trained']}")
    
    if 'learning_rate' in result:
        print(f"Learning Rate:           {result['learning_rate']}")
    
    if 'actual_results' in result and 'fit_status' in result['actual_results']:
        status = result['actual_results']['fit_status']
        print(f"Fit Type:          {status}")
    
    if 'actual_results' in result:
        if 'performance' in result['actual_results']:
            performance = result['actual_results']['performance']
            print(f"Performance Rating:      {performance}")
        
        if 'efficiency_status' in result['actual_results']:
            efficiency = result['actual_results']['efficiency_status']
            print(f"Training Efficiency:     {efficiency}")
    
    overfitting_score = max(0, accuracy_gap / 10) 
    print(f"Overfitting Score:       {overfitting_score:.2f}/10")
    
    if 'val_accs' in result:
        val_accs = result['val_accs']
        print(f"Best Epoch:              {val_accs.index(max(val_accs)) + 1}")
        print(f"Final Epoch Val Acc:     {val_accs[-1]:.2f}%")
        if len(val_accs) > 5:
            early_avg = sum(val_accs[:5]) / 5
            late_avg = sum(val_accs[-5:]) / 5
            improvement = late_avg - early_avg
            print(f"Improvement (last 5):    {improvement:+.2f}%")
    
    print(" ")

