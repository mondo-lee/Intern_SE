# Intern_SE

We show two task SE models with BLSTM, one for the Mapping task and the other for the IRM task. The dataset is VCTK.  
The Dataset is [VCTK_sp28](https://drive.google.com/file/d/1sePGXayGyJkqSFaCPwzI8GshZ5OFL-kJ/view?usp=share_link)

you can used run1.sh to training and testing the SE model.  
Or you can used the main.py to get results  

python main.py --model BLSTM_01 --target MAP --batch_size 1 --epochs 50 \  
               --loss mse --version jin_0608 --lr 5e-4 --task VCTK  
               
--model　　　　　　　　which model you used  
--target　　　　　　　　defined the training task MAP or IRM  
--batch_size　　　　　　training batch   
--epochs　　　　　　　　training epochs  
--loss　　　　　　　　　training loss with mse, l1, l1smooth, cosine  
--version　　　　　　　defined your version  
--lr　　　　　　　　　　training learn rate  
--task　　　　　　　　　defined your datasets  

## Edits in Updated

Building on the annotations in the `commented` branch, this branch seeks to update the code with newer library functionalities, most notably replacing librosa with torchaudio. In addition, docstrings will be added wherever relevant.

## Environment Setup (New)
```
python---------3.10.11
torch----------2.0.1+cu117
torchaudio-----2.0.2+cu117
numpy----------1.25.0
tensorboardx---2.6.1
matplotlib-----3.7.1
scikit-learn---1.2.2
tqdm-----------4.65.0
pandas---------2.0.2
pystoi---------0.3.3
pesq-----------0.0.4
```

## Environment Setup (Old)
python-----------3.6.13  
torch------------1.10.0  
librosa----------0.9.1  
numpy------------1.19.5  
scipy------------1.5.4  
tensorboardx-----2.5.1  
tqdm-------------4.64.0  
pandas-----------1.1.5  
pystoi-----------0.3.3  
pesq-------------0.0.4  
