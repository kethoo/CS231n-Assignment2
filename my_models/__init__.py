from .one_layer_cnn import build_SingleCNN
from .three_layer_cnn import build_TripleCNN
from .seven_layer_cnn import build_SevenLayerCNN
from .five_layer_cnn import build_FiveLayerCNN
from .five_layer_normalization import build_FiveLayerBatchNorm
from .five_layer_dropout import build_FiveLayerCNNWithDropout                    
from .five_layer_aug import build_FiveLayerCNNWithAugmentation                  
from .five_layer_norm_dropout import build_FiveLayerBatchNormDropout            
from .five_layer_with_attention import build_FiveLayerAttention
from .five_layer_with_skipping import build_FiveLayerSkipConnections
from .five_layer_with_all import build_FiveLayerWithAll
from .five_layer_batchnorm_attention import build_FiveLayerBatchNormAttention
from .five_layer_batchnorm_skipping import build_FiveLayerBatchNormSkipping
from .five_layer_batchnorm_combo import build_FiveLayerBatchNormCombo

const_n = 7

def model_not_here(curr_model, curr_model_mas):
    if curr_model not in curr_model_mas:
        raise ValueError(f"Current model '{curr_model}' is problematic")

def building_each_one(curr_model, curr_model_mas, num_classes):
    builder = curr_model_mas[curr_model]
    curr_ans = builder(num_classes=num_classes)
    return curr_ans

def conc_model(curr_model='5_layer_cnn', num_classes=const_n):
    curr_model_mas = {
        '1_layer_cnn': build_SingleCNN,
        '3_layer_cnn': build_TripleCNN, 
        '5_layer_cnn': build_FiveLayerCNN,
        '5_layer_normalization': build_FiveLayerBatchNorm,
        '5_layer_dropout': build_FiveLayerCNNWithDropout,                        
        '5_layer_aug': build_FiveLayerCNNWithAugmentation,                   
        '5_layer_norm_dropout': build_FiveLayerBatchNormDropout,                 
        '5_layer_with_attention': build_FiveLayerAttention,
        '5_layer_with_skipping': build_FiveLayerSkipConnections,
        '5_layer_with_all': build_FiveLayerWithAll,
        '5_layer_batchnorm_attention': build_FiveLayerBatchNormAttention,
        '5_layer_batchnorm_skipping': build_FiveLayerBatchNormSkipping,
        '5_layer_batchnorm_combo': build_FiveLayerBatchNormCombo,
        '7_layer_cnn': build_SevenLayerCNN
    }
    model_not_here(curr_model, curr_model_mas)
    ans = building_each_one(curr_model, curr_model_mas, num_classes)
    return ans

__all__ = [
    'conc_model',
    'build_SingleCNN',
    'build_TripleCNN', 
    'build_FiveLayerCNN',
    'build_FiveLayerBatchNorm',
    'build_FiveLayerCNNWithDropout',                                            
    'build_FiveLayerCNNWithAugmentation',                                       
    'build_FiveLayerBatchNormDropout',                                          
    'build_FiveLayerAttention',
    'build_FiveLayerSkipConnections',
    'build_FiveLayerWithAll',
    'build_FiveLayerBatchNormAttention',
    'build_FiveLayerBatchNormSkipping',
    'build_FiveLayerBatchNormCombo',
    'build_SevenLayerCNN'
]