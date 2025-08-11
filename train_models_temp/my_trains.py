import torch
import torch.nn as nn
import torch.optim as optim
import time
import wandb

class EmotionCNNTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        self.wandb_enabled = config.get('wandb_enabled', False)
        self.experiment_name = config.get('experiment_name', 'default_experiment')
        print(f"Trainer ready for {self.device}")
    
    def extract_model_prediction(self, model):
        try:
            if hasattr(model, 'curr_m_params'):
                model_info = model.curr_m_params()
                return {
                    'model_name': model_info.get('model_name', 'Unknown'),
                    'prediction': model_info.get('my_prediction', 'No prediction available'),
                    'details': model_info.get('details', 'No details'),
                    'expected_params': model_info.get('all_params', 'Unknown')
                }
        except Exception as e:
            print(f"Could not extract prediction: {e}")
        
        return {
            'model_name': 'Unknown',
            'prediction': 'No prediction available',
            'details': 'No details available',
            'expected_params': 'Unknown'
        }
    
    def analyze_actual_results(self, best_val_acc, final_train_acc, training_time, total_params):
        ind1 = final_train_acc - best_val_acc 
        ind2 = final_train_acc                
        
        def check_fitting():
            if ind1 > 12 and ind2 > 80:
                actual_fitting = "Overfitting"
                fitting_analysis = "Training accurac მაღალია და სხვაობაც საკმაოდ დიდია"
            elif ind1 > 20:
                actual_fitting = "Overfitting"
                fitting_analysis = "ძალიან დიდი სხვაობაა თრეინსა და ვალიდაციას შორის"
            elif best_val_acc < 50 and ind1 < 8:
                actual_fitting = "Underfitting"
                fitting_analysis = "ამჯერად მცირეა სხვაობა, თუმცა მოდელია ზედმეტად მარტივი და უბრალო"
            elif best_val_acc < 35:
                actual_fitting = "Underfitting"
                fitting_analysis = "ძალიან ცუდი მოდელია"
            elif ind1 > 8:
                actual_fitting = "Slight overfitting"
                fitting_analysis = " Overfitting გვაქვს, მაგრამ ნორმალური მოდელია"
            else:
                actual_fitting = "Good fitting"
                fitting_analysis = "ბალანსირებული მოდელი,კარგი შედეგით"
            
            return actual_fitting, fitting_analysis

        def check_perf():
            if best_val_acc >= 80:
                perf = "Great performance"
            elif best_val_acc >= 70:
                perf = "Good performance"
            elif best_val_acc >= 60:
                perf = "Normal performance"
            else:
                perf = "Bad performance"
            
            return perf

        def check_training():
            efficiency = best_val_acc / max(training_time / 60, 0.1)  
            
            if efficiency > 20:
                training_speed_vel = "Very good training speed"
            elif efficiency > 10:
                training_speed_vel = "Good training speed"
            elif efficiency > 5:
                training_speed_vel = "Normal training speed"
            else:
                training_speed_vel = "Slow training speed"
            
            return training_speed_vel

        #ქოლ
        fit_status, fit_analysis = check_fitting()
        performance = check_perf()
        efficiency_status = check_training()
        
        
        return {
            'fit_status': fit_status,
            'fit_analysis': fit_analysis,
            'performance': performance,
            'efficiency_status': efficiency_status,
            'train_val_gap': ind1,
            'final_train_acc': ind2,
            'best_val_acc': best_val_acc
        }

    def train_single_model(self, model, train_loader, val_loader, model_name):
        print(f"Training {model_name}...")
        
        
        if not self.wandb_enabled:
               
            self.wandb_enabled = True
        
        
        def get_project_name():
            experiment = self.experiment_name.lower()
            
            if 'regularization' in experiment:
                return "regularization_exp"
            elif 'batchnorm' in experiment or 'normalization' in experiment:
                return "batchnorm_experiment"
            elif 'depth' in experiment or 'layer' in experiment:
                return "depth_experiment"  
            elif 'attention' in experiment:
                return "attention_experiment"
            elif 'skip' in experiment:
                return "skip_experiment"
            elif 'combined' in experiment or 'all' in experiment:
                return "combined_experiment"
            elif 'advanced_arch' in experiment:  
                return "advanced_architecture_experiment"
            else:
                return "emotion-cnn-experiments"  # Default
        
        
        project_name = get_project_name()
        wandb.init(
            project=project_name,  
            name=f"{self.experiment_name}_{model_name}",  
            config={
                "model_name": model_name,
                "learning_rate": self.config['learning_rate'],
                "num_epochs": self.config['num_epochs'],
                "patience": self.config['patience'],
                "experiment": self.experiment_name
            },
            reinit=True  
        )
        
        model_prediction = self.extract_model_prediction(model)
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        
        best_val_acc = 0.0
        patience_counter = 0
        val_accs = []
        train_accs = []
        train_losses = []
        val_losses = []
        
        start_time = time.time()
        
        for epoch in range(self.config['num_epochs']):
            model.train()
            train_correct = 0
            train_total = 0
            train_loss_sum = 0
            train_batches = 0
            
            for data, targets in train_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss_sum += loss.item()
                train_batches += 1
                
                _, predicted = torch.max(outputs, 1)
                train_total += targets.size(0)
                train_correct += (predicted == targets).sum().item()
            
            train_acc = 100 * train_correct / train_total
            train_accs.append(train_acc)
            epoch_train_loss = train_loss_sum / train_batches
            train_losses.append(epoch_train_loss)
            
            model.eval()
            val_correct = 0
            val_total = 0
            val_loss_sum = 0
            val_batches = 0
            
            with torch.no_grad():
                for data, targets in val_loader:
                    data, targets = data.to(self.device), targets.to(self.device)
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                    
                    val_loss_sum += loss.item()
                    val_batches += 1
                    
                    _, predicted = torch.max(outputs, 1)
                    val_total += targets.size(0)
                    val_correct += (predicted == targets).sum().item()
            
            val_acc = 100 * val_correct / val_total
            val_accs.append(val_acc)
            epoch_val_loss = val_loss_sum / val_batches
            val_losses.append(epoch_val_loss)
            
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            
            if epoch % 3 == 0:
                print(f'logging epoch {epoch}')
                wandb.log({
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc,
                    "train_loss": train_loss_sum / train_batches,
                    "val_loss": val_loss_sum / val_batches,
                    "accuracy_gap": train_acc - val_acc,
                    "loss_gap": (val_loss_sum / val_batches) - (train_loss_sum / train_batches),
                    "learning_rate": self.config['learning_rate'],
                    "best_val_acc_so_far": best_val_acc,
                }, step=epoch)
            
            if (epoch) %3 == 0 :
                print(f"  Epoch {epoch:3d}: Train {train_acc:.1f}%, Val {val_acc:.1f}%")
            
            if patience_counter >= self.config['patience']:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        training_time = time.time() - start_time
        total_params = sum(p.numel() for p in model.parameters())
        final_train_acc = train_acc
        print(f"{model_name}: {best_val_acc:.1f}%")
        actual_results = self.analyze_actual_results(best_val_acc, final_train_acc, training_time, total_params)
        
       
        wandb.finish()
        
        prediction_lower = model_prediction['prediction'].lower()
        actual_lower = actual_results['fit_analysis'].lower()
        
        prediction_correct = False
        if 'overfitting' in prediction_lower and 'overfitting' in actual_lower:
            prediction_correct = True
        elif 'balanced' in prediction_lower and 'balanced' in actual_lower:
            prediction_correct = True
        elif 'underfitting' in prediction_lower and 'underfitting' in actual_lower:
            prediction_correct = True
        
        result = {
            'model_name': model_name,
            'best_val_acc': best_val_acc,
            'parameters': total_params,
            'epochs': len(val_accs),
            'val_accs': val_accs,
            'train_accs': train_accs,
            'train_losses': train_losses,
            'val_losses': val_losses,        
            'prediction': model_prediction,
            'actual_results': actual_results,
            'prediction_correct': prediction_correct,
            'final_train_acc': final_train_acc,
            'training_time': training_time
        }
        
        return result
    
    def show_prediction_summary(self, results):
        if not results:
            return
        
        print(f"\nFull Predictions")
        
        for result in results:
            model_name = result['model_name'].replace('_', ' ').title()
            prediction = result['prediction']['prediction']
            actual = result['actual_results']['fit_analysis']
            correct = "" if result['prediction_correct'] else "X "
            
            print(f"{correct}{model_name}:")
            print(f"   Predicted: {prediction}")
            print(f"   Reality:   {actual}")
            print()
    
    def compare_models(self, models_to_compare, train_loader, val_loader, model_builder):
        print(f"\nComparing {len(models_to_compare)} models...")
        
        results = []
        for model_name in models_to_compare:
            try:
                model = model_builder(model_name)
                result = self.train_single_model(model, train_loader, val_loader, model_name)
                results.append(result)
                self.results[model_name] = result
            except Exception as e:
                print(f"Error with {model_name}: {e}")
        
        self.show_results(results)
        self.show_prediction_summary(results)
        return results
    
    def show_results(self, results):
        if not results:
            return
        
        print(f"\nFinal outcome")
        sorted_results = sorted(results, key=lambda x: x['best_val_acc'], reverse=True)
        
        for i, r in enumerate(sorted_results, 1):
            name = r['model_name'].replace('_', ' ').title()
            print(f"{i}. {name}")
            print(f"   Accuracy: {r['best_val_acc']:.1f}%")
            print(f"   Params: {r['parameters']:,}")
            print(f"   Status: {r['actual_results']['fit_status']}")
            print()
        winner = sorted_results[0]
        winner_name = winner['model_name'].replace('_', ' ').title()
        print(f"Better model is: {winner_name} ({winner['best_val_acc']:.1f}%)")

      
def notebook_helper(trainer, train_loader, val_loader, model_builder, models, description):
    print(description)
    return trainer.compare_models(models, train_loader, val_loader, model_builder)

def run_depth_experiment(trainer, train_loader, val_loader, model_builder):
    com = "Experiment to compare depth"
    models = ['1_layer_cnn', '3_layer_cnn', '5_layer_cnn', '7_layer_cnn']
    return notebook_helper(trainer, train_loader, val_loader, model_builder, models, com)

def run_batchnorm_experiment(trainer, train_loader, val_loader, model_builder):
    com = "Experiment to see the effect of batch normalization"
    models = ['5_layer_cnn', '5_layer_normalization']
    return notebook_helper(trainer, train_loader, val_loader, model_builder, models, com)

def run_attention_experiment(trainer, train_loader, val_loader, model_builder):
    com = "Experiment to see the effect of attention"
    models = ['5_layer_cnn', '5_layer_with_attention']
    return notebook_helper(trainer, train_loader, val_loader, model_builder, models, com)

def run_skip_experiment(trainer, train_loader, val_loader, model_builder):
    com = "Experiment too see the effect of skip connections"
    models = ['5_layer_cnn', '5_layer_with_skipping']
    return notebook_helper(trainer, train_loader, val_loader, model_builder, models, com)

def run_combined_experiment(trainer, train_loader, val_loader, model_builder):
    com = "Experiment too see the effect of attention + batchnorm + skipping"
    models = ['5_layer_cnn', '5_layer_with_all']
    return notebook_helper(trainer, train_loader, val_loader, model_builder, models, com)

def run_all_five_layer_experiments(trainer, train_loader, val_loader, model_builder):
    com = "Experiment to compare all 5 layered CNNs"
    models = ['5_layer_cnn', '5_layer_normalization', '5_layer_with_attention', 
              '5_layer_with_skipping', '5_layer_with_all']
    return notebook_helper(trainer, train_loader, val_loader, model_builder, models, com)
    
def run_regularization_experiment(trainer, train_loader, val_loader, model_builder):
    com = "Experiment to compare regularization methods"
    models = ['5_layer_aug', '5_layer_normalization', '5_layer_dropout', 
              '5_layer_norm_dropout']
    return notebook_helper(trainer, train_loader, val_loader, model_builder, models, com)

def run_adv_arch(trainer, train_loader, val_loader, model_builder):
    com = "Experiment to compare advanced architectural combinations"
    models = ['5_layer_batchnorm_attention', '5_layer_batchnorm_skipping', '5_layer_batchnorm_combo']
    return notebook_helper(trainer, train_loader, val_loader, model_builder, models, com)

def run_every_experiment(trainer, train_loader, val_loader, model_builder):
    com = "Experiment to compare every model i did for this assignment"
    models = ['1_layer_cnn', '3_layer_cnn', '5_layer_cnn', '7_layer_cnn',
              '5_layer_normalization', '5_layer_with_attention', 
              '5_layer_with_skipping', '5_layer_with_all']
    return notebook_helper(trainer, train_loader, val_loader, model_builder, models, com)