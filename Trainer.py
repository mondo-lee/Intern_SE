import torch.nn as nn ### contains the various layers in a NN
import torch
import pandas as pd
import os, sys
from tqdm import tqdm ### module to print a dynamically updating progress bar
import librosa, scipy ### librosa is a python package for music and audio analysis,
### scipy provides algorithms for various classes of problems including optimization, interpolation, DEs etc.
import pdb
import numpy as np ### ??? why import both numpy and scipy if scipy has all the functions of numpy and more detailed formulae in some cases? is it computation speed?
from scipy.io.wavfile import write as audiowrite ### scipy.io requires explicit import, wavefile.write allows the writing of wavfiles using NumPy arrays
from util import * ### auiliary module
# import pyworld as pw ### python wrapper of WORLD Vocoder, parametrizing speech into pitch contour (f0), harmonic spectral envelope (sp) and aperiodic spectral envelope (ap)
from sklearn import preprocessing ### sklearn is a machine learning module itself. preprocessing includes functions to normalize, discretize, and transform non-linearly etc.
# import torchaudio ### library for audio and signal processing with PyTorch

maxv = np.iinfo(np.int16).max ### iinfo - machine limits for integer types
epsilon = np.finfo(float).eps ### finfo - machine limits for float types

class Trainer:
    def __init__(self, model, version, epochs, epoch, best_loss, optimizer,scheduler, 
                 criterion, device, loader, Test_path, writer, model_path, score_path, args, Output_path, save_results, target):
        self.epoch = epoch  
        self.epochs = epochs
        self.best_loss = best_loss
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.version = version
        self.device = device
        self.loader = loader
        self.criterion = criterion
        self.save_results = save_results
        self.target = target
        
        self.Test_path = Test_path
        self.Output_path = Output_path

        self.train_loss = 0
        self.val_loss = 0
        self.writer = writer
        self.model_path = model_path
        self.score_path = score_path
        self.args = args

    def save_checkpoint(self,):
        state_dict = {
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_loss': self.best_loss
            }
        check_folder(self.model_path)
        torch.save(state_dict, self.model_path)
         
    def _train_step(self, in_y, in_c, target):
        
        device = self.device
        # spec_y, spec_c = in_y.to(device), in_c.to(device)  
        spec_y, spec_c = in_y.transpose(1,2).to(device), in_c.transpose(1,2).to(device)           
        log1p_y = torch.log1p(spec_y)
        log1p_c = torch.log1p(spec_c)
        
        if target == 'MAP': ### mapping
            pred = self.model(log1p_y)       
            loss = self.criterion(pred, log1p_c)
            
        elif target == 'IRM': ### ideal ratio mask
            pred_irm = self.model(log1p_y)        
            loss = self.criterion(log1p_y*pred_irm, log1p_c)
        
        self.train_loss += loss.item()
        self.optimizer.zero_grad()
        loss.backward() 
        self.optimizer.step() ### back-propagation? optimizer by default is adam
        # self.scheduler.step(loss) ### ???


    def _train_epoch(self):
        self.train_loss = 0

        progress = tqdm(total=len(self.loader['train']), desc=f'Epoch {self.epoch} / Epoch {self.epochs} | train', unit='step') ### progress bar
        self.model.train()
         
        for spec_y, spec_c in self.loader['train']:
            self._train_step(spec_y, spec_c, self.target)
            progress.update(1) ### update progress bar
            
        progress.close() ### once complete, close progress bar
        self.train_loss /= len(self.loader['train']) ### normalize loss function with respect to number of samples

        print(f'train_loss:{self.train_loss}')

    def _val_step(self, in_y, in_c, target):
        
        device = self.device
        # spec_y, spec_c = in_y.to(device), in_c.to(device) 
        spec_y, spec_c = in_y.transpose(1,2).to(device), in_c.transpose(1,2).to(device)           
        log1p_y = torch.log1p(spec_y)
        log1p_c = torch.log1p(spec_c)
        
        if target == 'MAP':
            pred = self.model(log1p_y)       
            loss = self.criterion(pred, log1p_c)
            
        elif target == 'IRM':      
            pred_irm = self.model(log1p_y)        
            loss = self.criterion(log1p_y*pred_irm, log1p_c)
        
        self.val_loss += loss.item()   
        ### near identical (with some exceptions) to _train_step, but does not include the lines below:
        
        '''
        self.train_loss += loss.item() ### replaced by increment to val_loss instead
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        '''
       
        # self.scheduler.step(loss)

    def _val_epoch(self): 
        self.val_loss = 0
     
        progress = tqdm(total=len(self.loader['val']), desc=f'Epoch {self.epoch} / Epoch {self.epochs} | valid', unit='step')
        self.model.eval()

        for spec_y, spec_c in self.loader['val']:
            self._val_step(spec_y, spec_c, self.target)
            progress.update(1)

        progress.close()

        self.val_loss /= len(self.loader['val'])
        
        print(f'val_loss:{self.val_loss}')

        ### near identical to _train, but includes below part which updates best_loss if val_loss is lower
       
        if self.best_loss > self.val_loss:
            
            print(f"Save model to '{self.model_path}'")
            self.save_checkpoint()
            self.best_loss = self.val_loss

            
    def write_score(self, test_file, clean_path, audio_path, target): ### ??? reasoning behind some of the operations, esp with matrix manipulation
        
        self.model.eval()
        noisy, sr = librosa.load(test_file,sr=16000) ### sample rate 16k
        clean, sr = librosa.load(os.path.join(clean_path, test_file.split('/')[-1]),sr=16000)
        log1p_y, y_phase, y_len = make_spectrum(y=noisy,feature_type ='log1p') ### map onto log1p spectrum, from util.py
        
        log1p_y_1 = torch.from_numpy(log1p_y).cuda().detach().transpose(0,1)

        '''
        torch.from_numpy(log1p_y): creates tensor based on np.ndarray of log1p_y
        torch.cuda(): adds support for CUDA tensor types
        torch.tensor.detach(): returns a new tensor, detached from the original
        transpose(0,1): swaps the two dimensions provided as arguments
        '''

        pred = self.model(log1p_y_1.unsqueeze(0))
        ### unsqueeze inserts singleton dimension at given index, i.e., puts each element of the given index into a higher dimension
        ### e.g. x=torch.tensor([1,2,3,4]), torch.unsqueeze(x,0) returns [[1,2,3,4]], torch.unsqueeze(x,1) returns [[1],[2],[3],[4]]

        pred = pred.cpu().detach().numpy().squeeze(0).T
        ### squeeze removes these singleton dimensions, or the singleton dimension at the given index
        ### e.g. x is of dimension A*1*B, x.squeeze() and x.squeeze(1) both return tensor of dimensions A*B
        ### x.squeeze(0) or x.squeeze(2) leaves tensor unchanged
                        
        if target == 'MAP':
            pred_clean = recons_spec_phase(pred, y_phase, y_len, feature_type='log1p') ### from util.py
        elif target == 'IRM':      
            pred_clean = recons_spec_phase(pred*log1p_y, y_phase, y_len, feature_type='log1p')

        if self.save_results == 'True':
            out_a_path = os.path.join(audio_path,  f"{test_file.split('/')[-1].split('.')[0]+'.wav'}")
            check_folder(out_a_path)
            audiowrite(out_a_path,16000,(pred_clean* maxv).astype(np.int16))


        ### normalize clean, noisy and pred_clean_wav wrt maximum values
        clean = clean/abs(clean).max()
        noisy = noisy/abs(noisy).max() ### ??? cal_score seems to normalise the 2 inputted signals already, is there a different purpose?
        pred_clean_wav = pred_clean/abs(pred_clean).max()
        
        n_pesq, n_stoi = cal_score(clean,noisy)
        s_pesq, s_stoi = cal_score(clean,pred_clean_wav)
        
        wave_name = test_file.split('/')[-1].split('.')[0]
        with open(self.score_path['PESQ'], 'a') as f:
            f.write(f'{wave_name},{n_pesq},{s_pesq}\n')
        with open(self.score_path['STOI'], 'a') as f:
            f.write(f'{wave_name},{n_stoi},{s_stoi}\n')

    def train(self):
        while self.epoch < self.epochs:
            self._train_epoch()
            self._val_epoch()
            
            self.sc_name = f'{self.args.task}/{self.model.__class__.__name__}_{self.args.optim}' \
            f'_{self.args.loss}' ### ??? explicit line joining, does the string join?
            
            self.writer.add_scalars(self.sc_name, {'train': self.train_loss},self.epoch) ### records task, model, optimizer, loss, train loss
            self.writer.add_scalars(self.sc_name, {'val': self.val_loss},self.epoch) ### same but with val loss instead
                                
            self.epoch += 1
            
    def test(self):
        # load model
        self.model.eval()
        checkpoint = torch.load(self.model_path)
        self.model.load_state_dict(checkpoint['model'])        
        
        test_folders = get_filepaths(self.Test_path['noisy'],'.wav')
        test_folders = test_folders
        clean_path = self.Test_path['clean']
        
        audio_path = self.Output_path['audio']        
        print(self.score_path)
        
        ### write headers for PESQ file
        check_folder(self.score_path['PESQ'])
        if os.path.exists(self.score_path['PESQ']): ### remove preexisting PESQ, presumably to remove name confusions
            os.remove(self.score_path['PESQ'])
        with open(self.score_path['PESQ'], 'a') as f: ### read/write capabilities using open(), native Python
            f.write('Filename,Noisy_PESQ,Pred_PESQ\n')
        
        ### repeat with STOI file
        check_folder(self.score_path['STOI'])
        if os.path.exists(self.score_path['STOI']):
            os.remove(self.score_path['STOI'])
        with open(self.score_path['STOI'], 'a') as f:
            f.write('Filename,Noisy_STOI,Pred_STOI\n')   
            
        for test_file in tqdm(test_folders): ### using tqdm as iterator allows recording of progress
            
            self.write_score(test_file, clean_path, audio_path, self.target)
        
        ### calculate record noisy and predicted average PESQ and STOI
        data = pd.read_csv(self.score_path['PESQ'])
        n_pesq_mean = data['Noisy_PESQ'].to_numpy().astype('float').mean()
        s_pesq_mean = data['Pred_PESQ'].to_numpy().astype('float').mean()

        with open(self.score_path['PESQ'], 'a') as f:
            f.write(','.join(('Average',str(n_pesq_mean),str(s_pesq_mean)))+'\n')


        data = pd.read_csv(self.score_path['STOI'])
        n_stoi_mean = data['Noisy_STOI'].to_numpy().astype('float').mean()
        s_stoi_mean = data['Pred_STOI'].to_numpy().astype('float').mean()
        
        with open(self.score_path['STOI'], 'a') as f:
            f.write(','.join(('Average',str(n_stoi_mean),str(s_stoi_mean)))+'\n')
    
    
