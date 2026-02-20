# GANBase

This is the code of the article <i> Genome-Guided Generative Adversarial Learning enables nanopore adaptive sequencing </i>. We developed the GANBase, which can identify whether base sequences obtained by nanopore sequencing are target genome sequences, enabling adaptive sequencing.

During the training phase of GANBase, the discriminator utilizes a Transformer encoder, while the generator employs an LSTM architecture. The input data is obtained from the reference genome and constructed into a training set using equal-length sliding window truncation and sequence encoding. The discriminator determines the classification result.
For test, load the trained model to classify.

Here, we provided the scripts which are based on Python. User can use pretest_predict.py or prediction.py to test the model on the simulated data. The read_until.py can be used in real adaptive sequencing experiment.
We have supplied the trained models (saved in 'model' folder), and simulated test data sample in 'data' folder.



## Table of Contents

- [GANBase](#GANBase)
  - [Table of Contents](#table-of-contents)
  - [install](#install)
    - [install](#installation)
    - [Dependencies](#dependencies)
  - [Usage](#usage)
    - [Data Preprocessing](#data-preprocessing)
    - [For prediction](#for-prediction)
        - [For pre-test](#pre-test)
        - [For human-test](#human-test)
        - [For real-world test](#real-world-test)
    - [For training](#for-training)



## install
### Dependencies

    torch matplotlib pathlib pandas numpy tqdm sklearn

Please install the packages in requirements.txt

### Installation
If the dependencies are satified, We can install it manually by using the commands below:

```bash
git clone https://github.com/FPPGroup/GANBase.git
cd GANBase
conda create -n GANBase_env python=3.6 -y
conda activate GANBase_env
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```
<!-- or 
```bash
conda env create -f environment.yml -n GANBase_env
``` -->




## Usage


### Data Preprocessing

Details see data/data_readme.md, use jupyter-notebook to run data preprocess.


<b>input : </b>  fasta_file, it defines the target DNA sequence in a one-letter code, containing sequence ID and sequence.

for example:

    >BS.pilon.polished.v3.ST170922
    GTTGTACTTCGTTCAGTTACGTATTGCTGCCCTGCTGTCGGAAGGCCGTCTGACCACCAGCCGTAAGCTGCTACTATAAACCAGAATGGAAGGCTTGC...
<b>output : </b> txt_file or npy_file after preprocess, named as 'Species_datanum' or 'Species1_datanum1_Species2_datanum2'.
for example:

    2 3 3 2 3 0 1 3 3 1  ...  3 2 1 3 2 3 1 2 2 0 0
    0 2 0 0 0 2 1 2 0 3  ...  2 1 0 3 2 1 3 2 0 0 2
    1 3 2 3 0 3 1 0 2 3  ...  2 1 3 1 1 2 0 3 1 0 2
or

    ['2 3 3 2 3 0 1 3 3 1  ...  3 2 1 3 2 3 1 2 2 0 0']
    ['0 2 0 0 0 2 1 2 0 3  ...  2 1 0 3 2 1 3 2 0 0 2']
    ['1 3 2 3 0 3 1 0 2 3  ...  2 1 3 1 1 2 0 3 1 0 2']

or

    ['2 3 3 2 3 0 1 3 3 1  ...  3 2 1 3 2 3 1 2 2 0 0' 'Sta']
    ['0 2 0 0 0 2 1 2 0 3  ...  2 1 0 3 2 1 3 2 0 0 2' 'Sta']
    ['1 3 2 3 0 3 1 0 2 3  ...  2 1 3 1 1 2 0 3 1 0 2' 'Sta']



### For prediction

#### prediction
We use 'prediction' to determine whether the input sequence is a target sequence, and select to deplete or enrich based on the option.

```bash
python script/prediction.py -h ## show the parameters
```
paramaters

```bash
usage: prediction.py [-h] [--data data] [--model_path PATH] [--length N]
                     [--option OPTION]

Prediction

optional arguments:
  -h, --help         show this help message and exit
  --data data        input sigle sequence data
  --model_path PATH  your model files paths (default: "model/Human_model.pt")
  --length N         length (default: 200)
  --option OPTION    choose remove or enrichment target species  (default: "remove")
```

example

```bash
CUDA_VISIBLE_DEVICES=0 python script/prediction.py --model_path "model/Human_model.pt" --data 'ATTGTACTTCGTTCAATCACTTCCGGTATTTGTACTTCGTTCAGTTTTCAAATGAAGGTAGGTGTTTAACCTCGATTCCGTTTGTAGTCGTCTGGTTTTTTCGCATTTATCGTGAAACGCTTTCGCGTTTTTCGTGCGCCGCTTCATTAGTTATATTATTAAATATTAACTAATGTGTGCTCTATATTTATTGAATAGTT'
```


#### pre-test
We use 'pretest_predict' to determine whether the sequences from data path conform to target(from model), output predict accuracy.

```bash
python script/pretest_predict.py -h ## show the parameters
```


paramaters

```bash
usage: pretest_predict.py [-h] [--data_path PATH] [--model_path PATH]
                          [--out_path PATH] [--batch_size N]

PreTest_Predict

optional arguments:
  -h, --help         show this help message and exit
  --data_path PATH   your data files path (default:"data/Bac_2000.npy")
  --model_path PATH  your model files path (default: "model/Sta_model.pt")
  --out_path PATH    path to save out files (default: "out/")
  --batch_size N     batch size (default: 200)
```

example

```bash
CUDA_VISIBLE_DEVICES=0 python script/pretest_predict.py --model_path "model/Sta_model.pt" --data_path "data/Bac_2000.npy"
```


   

#### human-test
We use 'human-test' to determine whether the sequences from data path conform to human, input file names as 'Human_Human quantity_other species_quantity', output predict accuracy.

<!-- 
```bash
## use given model to test human and unhuman

python script/human-test.py -h ## show the parameters

``` -->

example

```bash
CUDA_VISIBLE_DEVICES=0 python script/human-test.py --model_path "model/Human_model.pt" --data_path "data/Human_2000_Sta_2000.npy"
```


<!-- ### predict 


```bash
## pre-test you can by this command

python script/prediction.py --model_path 'model/model_dis.pt' --data_path "data/Bac2000.npy"


```

example

```bash
python script/prediction.py --model_path 'model/model_dis.pt' --data_path "data/Bac2000.npy"

``` -->



#### real-world test
We use 'read_until.py' to the real-world test. By its read_until_api, we use human_model to deplete human sequence as much as possible. 

```bash
python script/read_until.py
``` 






### For training
```bash
## use jupyter-notebook to run train data preprocess ， use train for training.
python train.py  -h  ## show the parameters
```
paramaters:


```bash
START...
usage: train.py [-h] [--project_path PATH] [--data_path PATH] [--rounds N]
                [--g_pretrain_steps N] [--d_pretrain_steps N]
                [--update_rate UR] [--n_rollout N] [--vocab_size N]
                [--batch_size N] [--g_batch_size N] [--gen_lr LR]
                [--dis_lr LR] [--no_cuda] [--seed S] [--seq_len S]

optional arguments:
  -h, --help            show this help message and exit
  --project_path PATH   Project path (default: )
  --data_path PATH      data path to save files (default: )
  --rounds N            rounds of adversarial training (default: 101)
  --g_pretrain_steps N  steps of pre-training of generators (default: 15)
  --d_pretrain_steps N  steps of pre-training of discriminators (default: 1)
  --update_rate UR      update rate of roll-out model (default: 0.8)
  --n_rollout N         number of roll-out (default: 16)
  --vocab_size N        vocabulary size (default: 10)
  --batch_size N        batch size (default: 280)
  --g_batch_size N      batch size (default: 50000)
  --gen_lr LR           learning rate of generator optimizer (default: 1e-3)
  --dis_lr LR           learning rate of discriminator optimizer (default:
                        1e-3)
  --no_cuda             disables CUDA training
  --seed S              random seed (default: 1)
  --seq_len S           sequence length (default: 200)
```


example

```bash
CUDA_VISIBLE_DEVICES=0 python script/train.py --project_path your_project_dir --data_path your_data_path  --g_pretrain_steps 15 --d_pretrain_steps 1 --batch_size 280 --g_batch_size 5000 --rounds 100 
```




