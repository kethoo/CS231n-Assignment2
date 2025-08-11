"""
Visualization functions for emotion CNN experiments
Separated from training logic for better organization and reusability
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

def plot_model_comparison(results, title="Model Comparison"):
    if not results:
        print("No results to plot")
        return
    
    names = [name.replace('_', ' ').title() for name in results.keys()]
    accs = [result['best_val_acc'] for result in results.values()]
    colors = []
    status_labels = []
    for result in results.values():
        status = result['actual_results']['fit_status']
        if 'Overfitting' in status:
            colors.append('#FF6B6B')  
            status_labels.append('Overfitting')
        elif 'Underfitting' in status:
            colors.append('#FFB347')   
            status_labels.append('Underfitting')
        elif 'Slight' in status:
            colors.append('#FFD93D')  
            status_labels.append('Slight Overfitting')
        else:
            colors.append('#FF69B4')  
            status_labels.append('Good Fit')
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(names, accs, color=colors, edgecolor='darkmagenta', linewidth=2)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('Validation Accuracy (%)')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    
    
    unique_statuses = list(set(status_labels))
    color_map = {
        'Good Fit': '#FF69B4',
        'Overfitting': '#FF6B6B', 
        'Underfitting': '#FFB347',
        'Slight Overfitting': '#FFD93D'
    }
    
    legend_elements = [Patch(facecolor=color_map[status], label=status) 
                      for status in unique_statuses if status in color_map]
    plt.legend(handles=legend_elements, loc='upper left')
    
    
    for i, (bar, acc, status) in enumerate(zip(bars, accs, status_labels)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{acc:.1f}%', ha='center', fontweight='bold', fontsize=11)
        
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 2, 
                status.split()[0], ha='center', fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    #plt.show()

def plot_training_curves(results, title="Training Curves"):
    if not results:
        print("No results to plot training curves")
        return
    
    num_models = len(results)
    if num_models == 1:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        axes = [ax]
    elif num_models == 2:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    elif num_models <= 4:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
    
    for i, (model_name, result) in enumerate(results.items()):
        if i >= len(axes):
            break
            
        if 'val_accs' in result:
            epochs = range(1, len(result['val_accs']) + 1)
            val_accs = result['val_accs']
            
            axes[i].plot(epochs, val_accs, 'b-', label='Validation', linewidth=2, marker='o', markersize=3)
            
            if 'train_accs' in result and len(result['train_accs']) == len(val_accs):
                train_accs = result['train_accs']
                axes[i].plot(epochs, train_accs, 'r-', label='Training', linewidth=2, marker='s', markersize=3)
            else:
               
                final_train_acc = result.get('final_train_acc', 0)
                axes[i].axhline(y=final_train_acc, color='red', linestyle='--', 
                               label=f'Final Train ({final_train_acc:.1f}%)', linewidth=2)
            
            status = result['actual_results']['fit_status']
            if 'Overfitting' in status:
                axes[i].set_facecolor('#FFE5E5')  # red 
            elif 'Underfitting' in status:
                axes[i].set_facecolor('#FFF4E5')  #orange 
            else:
                axes[i].set_facecolor('#F0F8FF')  # blue 
            
            clean_name = model_name.replace('_', ' ').title()
            axes[i].set_title(f'{clean_name}\n({status})', fontweight='bold')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel('Accuracy (%)')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
   
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    #plt.show()

def plot_validation_comparison(results, title="Validation Learning Curves"):
   
    if not results:
        print("No results to plot")
        return
    
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for i, (model_name, result) in enumerate(results.items()):
        if 'val_accs' in result:
            epochs = range(1, len(result['val_accs']) + 1)
            clean_name = model_name.replace('_', ' ').title()
            color = colors[i % len(colors)]
            plt.plot(epochs, result['val_accs'], label=clean_name, 
                    linewidth=2, color=color, marker='o', markersize=4)
    
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy (%)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    #plt.show()

def plot_parameter_efficiency(results, title="Parameter Efficiency"):
    if not results:
        print("No results to plot")
        return
    
    params = [r['parameters'] for r in results.values()]
    accs = [r['best_val_acc'] for r in results.values()]
    names = [name.replace('_', ' ').title() for name in results.keys()]
    
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (p, a, name) in enumerate(zip(params, accs, names)):
        color = colors[i % len(colors)]
        plt.scatter(p, a, s=120, c=color, alpha=0.7, edgecolors='black', linewidth=1)
        plt.annotate(name, (p, a), xytext=(8, 8), 
                    textcoords='offset points', fontsize=11, fontweight='bold')
    
    plt.xlabel('Parameters')
    plt.ylabel('Validation Accuracy (%)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    #plt.show()

     
def plot_metrics_over_epochs(results_dict, title="Model Metrics Over Training Epochs"):
   
    if not results_dict:
        print("No results to plot")
        return
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    linestyles = ['-', '--', '-.', ':']
    
    short_names = []
    for name in results_dict.keys():
        if '5_layer_cnn' in name:
            short_names.append('CNN')
        elif '5_layer_normalization' in name:
            short_names.append('BatchNorm')
        elif 'attention' in name:
            short_names.append('Attention')
        elif 'skipping' in name:
            short_names.append('Skip')
        else:
            short_names.append(name.replace('_layer', 'L').replace('_', ''))
    
    
    ax = axes[0]
    for i, (model_name, result) in enumerate(results_dict.items()):
        if 'val_accs' in result:
            epochs = range(1, len(result['val_accs']) + 1)
            ax.plot(epochs, result['val_accs'], label=short_names[i], 
                   color=colors[i % len(colors)], linewidth=2, 
                   linestyle=linestyles[i % len(linestyles)], marker='o', markersize=3)
    
    ax.set_title('Validation Accuracy Over Epochs', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
   
    ax = axes[1]
    for i, (model_name, result) in enumerate(results_dict.items()):
        if 'train_accs' in result:
            epochs = range(1, len(result['train_accs']) + 1)
            ax.plot(epochs, result['train_accs'], label=short_names[i], 
                   color=colors[i % len(colors)], linewidth=2, 
                   linestyle=linestyles[i % len(linestyles)], marker='s', markersize=3)
    
    ax.set_title('Training Accuracy Over Epochs', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Accuracy (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[2]
    for i, (model_name, result) in enumerate(results_dict.items()):
        if 'train_accs' in result and 'val_accs' in result:
            epochs = range(1, min(len(result['train_accs']), len(result['val_accs'])) + 1)
            gap = [t - v for t, v in zip(result['train_accs'][:len(epochs)], result['val_accs'][:len(epochs)])]
            ax.plot(epochs, gap, label=short_names[i], 
                   color=colors[i % len(colors)], linewidth=2, 
                   linestyle=linestyles[i % len(linestyles)], marker='^', markersize=3)
    
    ax.set_title('Overfitting Gap Over Epochs', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train - Val Accuracy (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    ax = axes[3]
    for i, (model_name, result) in enumerate(results_dict.items()):
        if 'train_losses' in result:
            epochs = range(1, len(result['train_losses']) + 1)
            ax.plot(epochs, result['train_losses'], label=short_names[i], 
                   color=colors[i % len(colors)], linewidth=2, 
                   linestyle=linestyles[i % len(linestyles)])
    
    ax.set_title('Training Loss Over Epochs', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[4]
    for i, (model_name, result) in enumerate(results_dict.items()):
        if 'val_accs' in result:
            val_accs = result['val_accs']
            initial_acc = val_accs[0]
            improvement = [acc - initial_acc for acc in val_accs]
            epochs = range(1, len(val_accs) + 1)
            ax.plot(epochs, improvement, label=short_names[i], 
                   color=colors[i % len(colors)], linewidth=2, 
                   linestyle=linestyles[i % len(linestyles)], marker='D', markersize=3)
    
    ax.set_title('Learning Progress (Improvement from Start)', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Improvement in Val Accuracy (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    ax = axes[5]
    for i, (model_name, result) in enumerate(results_dict.items()):
        if 'val_losses' in result:
            epochs = range(1, len(result['val_losses']) + 1)
            ax.plot(epochs, result['val_losses'], label=short_names[i], 
                   color=colors[i % len(colors)], linewidth=2, 
                   linestyle=linestyles[i % len(linestyles)])
    
    ax.set_title('Validation Loss Over Epochs', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    #plt.show()

def plot_parameter_effectiveness(results_dict, title="Parameter Effectiveness"):
    
    plt.figure(figsize=(10, 6))
    results_list = list(results_dict.values())
    model_names = list(results_dict.keys())
    params = [r['parameters'] for r in results_list]
    accs = [r['best_val_acc'] for r in results_list]
    
    
    colors = plt.cm.Set1(range(len(model_names))) 
    
    plt.scatter(params, accs, s=120, c=colors, alpha=0.7, 
               edgecolors='black', linewidth=1)
    
    
    short_names = []
    for name in model_names:
        if 'layer' in name:
            
            layer_num = name.split('_')[0]
            short_names.append(f"{layer_num}L")
        else:
            short_names.append(name[:6])
    
    for i, short_name in enumerate(short_names):
        plt.annotate(short_name, (params[i], accs[i]), xytext=(8, 8), 
                    textcoords='offset points', fontsize=11, fontweight='bold')
    
    plt.xlabel('Parameters')
    plt.ylabel('Validation Accuracy (%)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    
    plt.tight_layout()
    #plt.show()

def quick_analysis(results):
    print("Creating comprehensive visualization analysis...")
    plot_model_comparison(results)
    plot_training_curves(results)
    print("Analysis complete!")