import torch

def generate_default_config():
    configs = {}
    
    # Device
    configs['use_gpu'] = torch.cuda.is_available()
    configs['use_multi_gpu'] = configs['use_gpu'] and torch.cuda.device_count() > 1
    configs['device'] = torch.device('cuda' if torch.cuda.is_available() 
                                     and configs['use_gpu'] else 'cpu')
    
    # Dataset
    configs['dataset'] = None
    
    # Training parameters
    configs['dtype'] = torch.float
    configs['lr'] = 1e-3
    configs['weight_decay'] = 1e-5
    configs['batch_size'] = 1000
    configs['start_epoch'] = 0
    configs['max_epoch'] = 200
    configs['evaluate'] = False
    configs['pre_sigmoid_bias'] = 0.5
    
    # Training information display and log
    configs['display'] = True
    configs['display_freq'] = 10
    configs['save_checkpoint_path'] = 'checkpoint'
    configs['exp'] = 'exp'
    
    # Reproducibility
    configs['rand_seed'] = 0
    
    return configs

