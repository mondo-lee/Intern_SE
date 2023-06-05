import torch, os
import torch.nn as nn
from torch.optim import Adam,SGD
### adaptive momentum, stochastic gradient descent (optionally with momentum)
from util import *
from torch.utils.data import DataLoader
### heart of data-loading capabilities of PyTorch, supporting map-style and iterable-style datasets
from sklearn.model_selection import train_test_split
### facilitates train test split, splitting arrays or matrices into random train and test subjects
from torch.utils.data.dataset import Dataset ### abstract class representing Dataset
import pdb
from tqdm import tqdm 
from joblib  import parallel_backend, Parallel, delayed
### enables lightweight pipelining, through:
###     transparent disk-caching of functions and lazy re-evaluation
###     easy simple parallel computing. optimized to be fast and robust, specific optimizations for numpy arrays
from collections import OrderedDict ### subclass of dictionary that remembers orders in which entries were added
from torch.optim.lr_scheduler import ReduceLROnPlateau ### reduce learning rate when metric has stopped improving

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) ### only counted if gradient is required
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        for name, param in m.named_parameters():
            if param.requires_grad==True:
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)
                    ### also known as glorot initializer, randomly samples from segmented version of normal distribution
                    ### stdev determined using number of input and output units in weight tensor
        
        
def load_checkoutpoint(model,optimizer,checkpoint_path):

    if os.path.isfile(checkpoint_path):
        model.eval()
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        try:
            model.load_state_dict(checkpoint['model'])
        except:
            model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['model'].items()}) ### i see the issue now... but why?
#             model.load_state_dict({k.replace('module.',''):v for k,v in torch.load('checkpoint.pt').items()})

        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        print(f"=> loaded checkpoint '{checkpoint_path}' (epoch {epoch})")
        
        return model,epoch,best_loss,optimizer
    else:
        raise NameError(f"=> no checkpoint found at '{checkpoint_path}'")



def Load_model(args,model,checkpoint_path,model_path):
    
    criterion = {
        'mse'     : nn.MSELoss(),
        'l1'      : nn.L1Loss(),
        'l1smooth': nn.SmoothL1Loss(), ### portion of l1 norm near 0 is replaced with l2 norm
        'cosine'  : nn.CosineEmbeddingLoss()} ### 1 - cosine similarity of the predicted and actual outputs,
                                              ### or max(0, margin-cos(x₁,x₂)), for some positive margin

    device    = torch.device(f'cuda:{args.gpu}') ### context manager that selects device
 
    criterion = criterion[args.loss].to(device) ### torch.tensor.to sends the argument to the
                                                ### desired device as the correct dtype
    
    optimizers = {
        'adam'    : Adam(model.parameters(),lr=args.lr,weight_decay=0),
        'SGD'     : SGD(model.parameters(),lr=args.lr,weight_decay=0)
    }
    
    optimizer = optimizers[args.optim]
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=44, verbose=True, threshold=1e-8, threshold_mode='rel', cooldown=22, min_lr=0, eps=1e-8)
    '''
    mode: min/max.  min - lr reduced when quantity has stopped decreasing
                    max - lr reduced when quantity has stopped increasing
    factor: factor by which learning rate is reduced
    patience: number of samples before reduction
    verbose: if True, prints a message to stdout for each update
    threshold: threshold for optimum, default 1e-4
    threshold_mode: rel/abs. fractional threshold or absolute threshold
    cooldown: number of samples after reduction before resuming normal operation
    min_lr: lower bound on learning rate of all or individual param groups
    eps: minimal decay applied to lr
    '''


    if args.resume: ### ??? How is resume changed?
        model,epoch,best_loss,optimizer = load_checkoutpoint(model,optimizer,checkpoint_path)
    elif args.retrain:
        model,epoch,best_loss,optimizer = load_checkoutpoint(model,optimizer,model_path)
               
    else:
        epoch = 0
        best_loss = 500
        model.apply(weights_init)
        
    para = count_parameters(model)
    print(f'Num of model parameter : {para}')
        
    return model,epoch,best_loss,optimizer,scheduler,criterion,device


def Load_data(args, Train_path):
    
    file_paths = get_filepaths(Train_path['noisy'],'.wav')
    clean_path = Train_path['clean']
    # pdb.set_trace()
    train_paths,val_paths = train_test_split(file_paths,test_size=0.2,random_state=999)

    train_dataset, val_dataset = CustomDataset(train_paths,clean_path), CustomDataset(val_paths,clean_path)
    loader = { 
        'train':DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=False),
        'val'  :DataLoader(val_dataset, batch_size=args.batch_size,
                            num_workers=4, pin_memory=False)
        
    }

    '''
    dataset: dataset from which to load data,
    batch_size: samples per batch to laod
    shuffle: whether or not to reshuffle data at each epoch
    num_workers: number of subprocess to use for data loading
    pin_memory: if true, data loader will copy tensors into device before returning them
    '''

    return loader

def load_torch(path): ### see line 48
    return torch.load(path)


class CustomDataset(Dataset):

    def __init__(self, paths, clean_path):   # initial logic happens like transform
        
        self.n_paths = paths
        self.c_paths = [os.path.join(clean_path, noisy_path.split('/')[-1]) for noisy_path in paths]
    def __getitem__(self, index):
        
        noisy, sr = librosa.load(self.n_paths[index],sr=16000)
        sep_y, y_phase,y_len = make_spectrum(y=noisy,feature_type ='spec') ### from util.py
        # print('sep_y.shape = ', sep_y.shape)
        
        clean, sr = librosa.load(self.c_paths[index],sr=16000)
        sep_c, c_phase,c_len = make_spectrum(y=clean,feature_type ='spec')
        # print('sep_c.shape = ', sep_c.shape)
        
        return sep_y, sep_c
        # return sep_y.transpose(1,2), sep_c.transpose(1,2)

    def __len__(self):  # return count of sample we have
        
        return len(self.n_paths)        
        
    
