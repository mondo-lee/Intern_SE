### importing packages

import os, argparse, torch, random, sys

'''
os: part of stdlib.
    provides a portable way of using operating system dependent functionality.
    OSError is raised by functions in case of invalid/inaccessible file names
    and paths, or arguments not accepted by the operating system
    
argparse: part of stdlib.
    used to write user-friendly CLIs, generating help and usage messages
    and issuing errors when invalid arguments are inputted

torch: part of PyTorch.
    provides tensor computation with strong gpu acceleration, enables DNNs
    built on tape-based autograd system.
    Autograd is a core torch package for automatic differentiation:
    forward phase - autograd remembers all operations it excecuted,
    backward phase - replay operations
    - automatic differentiation: exploits the fact that every computer program
        executes a series of elementary arithmetic operations and functions.
        repeated chain rules can allow automatic computation of partial
        derivatives

random: part of stdlib.
    implements pseudo-random number generators for various distributions

sys: part of stdlib.
    provides access to certain variables used or maintained by the interpreter,
    or to functions that interact strongly with the interpreter


'''


### auxiliary modules
from Trainer import Trainer
from Load_model import Load_model, Load_data
from util import check_folder

from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import pandas as pd
import pdb

'''
tensorboardX: an adaptation of tensorboard that applies to NN frameworks other
    than TensorFlow. tensorboard records the metrics and related images within
    the training process, allowing programmers to observe it.
    SummaryWriter is a class which encapsulates everything in tensorboardX.

torch.backends.cudnn: part of torch.
    torch.backends controls the behaviour of various backends that PyTorch
    supports. a machine learning backend is the part of the software that runs
    the algorithms.
    cudnn is then a GPU-accelerated library of primitives for DNNs, with highly
    optimized implementations for routines including forward/backward
    convolution, pooling, normalization, and activation layers.
    
pandas: python data analysis library. for data analysis and manipulation

pdb: interactive source code debugger, supports setting breakpoints and
    single stepping, inspection of stack frames, source code listing etc...

* CUDA (compute unified device architecture): a parallel computing platform and
    API that allows softwares to use certain types of GPUs for general purpose
    processing
'''

# fix random
SEED = 999
random.seed(SEED)
torch.manual_seed(SEED)
cudnn.deterministic = True 
### Deterministic operations are often slower, but can save time in development
### by facilitating experimentation, debugging & regression testing

def get_args():

    '''
    interacts with run1.sh,
    defining argument name, data type and default values
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='V1')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--epochs', type=int, default=400) ### number of passes
    ### number of samples per batch, sample size/batch size = iterations v
    parser.add_argument('--batch_size', type=int, default=8) 
    parser.add_argument('--lr', type=float, default=0.00005) ### learning rate
    parser.add_argument('--loss', type=str, default='l1') ### l1 norm loss
    parser.add_argument('--optim', type=str, default='adam')

    '''
    grab onto your seats this is a doozy

    0. setup

        the data (n pairs) is divided into m batches of size s. (s*m=n)
        the processes below are all of one epoch. repeat for multiple epochs

    1. typical gradient descent

        forward propagate from batch input array X to batch output array Y
        calculate loss function
        back-propagate to update weight matrix and bias vector of each layer:

        W <- W - α dW
        b <- b - α db

        α: learning rate
        minus sign indicates opposite direction of gradient, as goal is to
            minimize loss function

        1b varying batch size

        s = 1: stochastic gradient descent
        s = n: batch gradient descent
        intermediate values are usually chosen to be powers of 2

    2. momentum gradient descent (exponentially weighted moving average, EWMA)

        while a simple moving average adds up the last n terms and divides by n,
        an EWMA multiplies the last term by 1-β and the current term by β, the
        forgetting factor. this can be intuitively understood by multiplying every
        term before the current one by 1-β, resulting in an exponentially decaying
        weight.

        In equation form:

        let θ[] be the list of values in a time series, and V[] be the respective
        EWMAs at each time:

        V₁ = β V₀ + (1-β) θ₁
        V₂ = β V₁ + (1-β) θ₂ 
           = β(β V₀ + (1-β) θ₁) + (1-β) θ₂
        ... etc. (V_n = β(V_n-1) + (1-β) θ_n )

        V₀ = 0

        2b. error-correcting factor

        As V₀=0, the first few moving averages will be smaller and therefore
        inaccurate. Therefore, a correction factor which diminishes in influence over
        time is introduced:

        Vcorr_t = V_t/(1-β^t)

        this is included in all the algorithms below

        2c. rationale

        the EWMA can be applied recursively, while also valuing recent batches
        (where the parameters would be nearer to optimizing the objective function)

        2d. application

        momentum gradient descent uses the EWMA to simulate what intuitively
        resembles "friction" and "momentum" in gradient descent.

        forward propagate from batch input array X to batch output array Y
        calculate loss function

        *back-propagate to update velocity of each parameter using EWMA*:
            V_dW_t = β(V_dW_(t-1)) + (1-β)dW
            V_db_t = β(V_db_(t-1)) + (1-β)db

        update each parameter:
            W <- W - α V_dW
            b <- b - α V_db

    3. root mean square propagation (RMSprop)

        a different interim variable is calculated, using the weighted root mean
        square instead.

        s_dW_t = β(s_dW_(t-1)) + (1-β)(dW)²   - note that the squaring is element-wise
        s_db_t = β(s_db_(t-1)) + (1-β)(db)²

        W <- W - α dW/(sqrt(s_dW) + ε) - ε is a small number preventing division by 0
        b <- b - α db/(sqrt(s_db) + ε)

    4. Adam optimization (Adaptive Moment Estimation)

        a combination of  momentum gradient descent and RMSprop.

        V_dW_t = β₁(V_dW_(t-1)) + (1-β₁)dW
        V_db_t = β₁(V_db_(t-1)) + (1-β₁)db

        s_dW_t = β₂(s_dW_(t-1)) + (1-β₂)(dW)²
        s_db_t = β₂(s_db_(t-1)) + (1-β₂)(db)²

        W <- W - α dW V_dW/(sqrt(s_dW) + ε)
        b <- b - α db V_db/(sqrt(s_db) + ε)

    P. postscript (hyperparameters)

        note that:
        the forgetting factors β₁ and β₂ are often set to 0.9 and 0.999,
        ε is often set to 10⁻⁸, and
        α usually decays with the increase of epoch number

    '''

    parser.add_argument('--model', type=str, default='BLSTM') ### bidirectional LSTM, account for both forward and backward association of information
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--target', type=str, default='MAP') #'MAP' or 'IRM'
    parser.add_argument('--task', type=str, default='VCTK') ### voice cloning toolkit
    parser.add_argument('--resume' , action='store_true') ### ??? is resume changed anywhere? If not, why keep it as an argument?
    parser.add_argument('--retrain', action='store_true') ### ??? why retrain?
    parser.add_argument('--save_results', type=str, default='False')
    parser.add_argument('--re_epochs', type=int, default=300) 
    parser.add_argument('--checkpoint', type=str, default=None)

    args = parser.parse_args()
    return args

def get_path(args):

    '''
    A guess:
    Produces path names for the checkpoint, the model and the score (PESQ and STOI) from the CLI command.
    Returns them as strings (checkpoint and model in tar format), and the two scores in a dictionary (in csv format)
    '''
    
    checkpoint_path = f'./checkpoint/'\
    f'{args.model}_{args.version}_{args.task}_{args.target}_epochs{args.epochs}_{args.optim}' \
    f'_{args.loss}_batch{args.batch_size}_lr{args.lr}.pth.tar'
    
    model_path = f'./save_model/'\
    f'{args.model}_{args.version}_{args.task}_{args.target}_epochs{args.epochs}_{args.optim}' \
    f'_{args.loss}_batch{args.batch_size}_lr{args.lr}.pth.tar'
    
    score_path = {
    'PESQ':f'./sourc/PESQ/'\
    f'{args.model}_{args.version}_{args.task}_{args.target}_epochs{args.epochs}_{args.optim}' \
    f'_{args.loss}_batch{args.batch_size}_lr{args.lr}.csv',       
    'STOI':f'./sourc/STOI/'\
    f'{args.model}_{args.version}_{args.task}_{args.target}_epochs{args.epochs}_{args.optim}' \
    f'_{args.loss}_batch{args.batch_size}_lr{args.lr}.csv'
    }
    
    return checkpoint_path,model_path,score_path

if __name__ == '__main__': ### if condition used for debugging purposes
    # get current path
    cwd = os.path.dirname(os.path.abspath(__file__))
    print(cwd)
    print(SEED)
        
    # get parameter
    args = get_args() ### calls get_args() from above
    
    print('model name =', args.model)
    print('target mode =', args.target)
    print('version =', args.version)
    print('Lr = ', args.lr)
    
    # data path
    Train_path = {
    'noisy':'/mnt/Nas234/Corpus/VCTK_28spk/noisy_trainset_wav',
    'clean':'/mnt/Nas234/Corpus/VCTK_28spk/clean_trainset_wav'
    } 

    Test_path = {
    'noisy':'/mnt/Nas234/Corpus/VCTK_28spk/noisy_testset_wav',
    'clean':'/mnt/Nas234/Corpus/VCTK_28spk/clean_testset_wav'
    }
        
    Output_path = {
    'audio':f'./result/'\
        f'{args.model}_{args.version}_{args.task}_{args.target}_epochs{args.epochs}_{args.optim}' \
        f'_{args.loss}_batch{args.batch_size}_lr{args.lr}'
    }
    
    # declair path
    checkpoint_path,model_path,score_path = get_path(args) ### calls get_path from above

    # tensorboard
    writer = SummaryWriter(f'./logs/'\
                           f'{args.model}_{args.version}_{args.task}_{args.target}_epochs{args.epochs}_{args.optim}' \
                           f'_{args.loss}_batch{args.batch_size}_lr{args.lr}') ### set up SummaryWriter with the args
    
    # pdb.set_trace() ### breakpoint
    exec (f"from models.{args.model.split('_')[0]} import {args.model} as model") ### default BLSTM, but can test multiple models, delimited with _...?
    model     = model()
    model, epoch, best_loss, optimizer, scheduler, criterion, device = Load_model(args,model,checkpoint_path, model_path) ### load in everything
    
    loader = Load_data(args, Train_path) ### load in data
    if args.retrain: ### if retrain is true, use a different number of epochs, and recover existing work using get_path
        args.epochs = args.re_epochs 
        checkpoint_path, model_path, score_path = get_path(args)
        
    # pdb.set_trace() ### breakpoint
    Trainer = Trainer(model, args.version, args.epochs, epoch, best_loss, optimizer,scheduler, 
                      criterion, device, loader, Test_path, writer, model_path, score_path, args, Output_path, args.save_results, args.target) ### everything loaded in is put into trainer
    try:
        if args.mode == 'train':
            Trainer.train()
        Trainer.test()
        
    except KeyboardInterrupt: ### assuming this is Ctrl-C as well ??? Assuming that loading the model from a pause is dependent on the Load_data module, but leaving this question here regardless
        state_dict = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_loss': best_loss
            }
        check_folder(checkpoint_path)
        torch.save(state_dict, checkpoint_path)
        print('Saved interrupt')
        try: ### uses sys, then os to exit the program ??? What is the coverage of sys and os?
            sys.exit(0)
        except SystemExit:
            os._exit(0)
