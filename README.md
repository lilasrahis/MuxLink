# MuxLink: Circumventing Learning-Resilient MUX-Locking Using Graph Neural Network-based Link Prediction
Lilas Alrahis, Satwik Patnaik, Muhammad Shafique, and Ozgur Sinanoglu

---
### About

MuxLink is a link prediction-based attack on learning resilient logic locking. This repository contains the python implementation of MuxLink attack in addition to our own implementation of the deceptive logic locking scheme ([DMUX](https://ieeexplore.ieee.org/document/9496607)).

**Contact**
Lilas Alrahis (lma387@nyu.edu)
### Setup
#### Step 1: Install the default GNN model (i.e., DGCNN)
```sh
$ git clone https://github.com/muhanzhang/pytorch_DGCNN
$ cd pytorch_DGCNN/lib
$ make -j4
$ cd ../..
```
#### Step 2: Install Required Packages
1) Install [PyTorch](https://pytorch.org/)
2) Install numpy, scipy, networkx, tqdm, sklearn, gensim

### Usage
#### Attacking DMUX
***1) Lock a design***
- Example, lock the c1908 ISCAS benchmark with key size of 32
```sh
$ cd ./MuxLink/DMUX_Locking
$ python3 convert_DMUX.py c1908 32 ../data/c1908_K32_DMUX
```
- `convert_DMUX.py` is a Python script that reads a circuit in Bench format and locks it using DMUX. It will convert the design into a graph. It assigns unique numerical IDs (0 to N-1) to the nodes (gates). N represents the total number of nodes (gates) in the design.
- It will generate a directory `../data/c1908_K32_DMUX` which includes:
-- The extracted features will be dumped in `feat.txt`. The ith line in feat.txt represent the feature vector of the node ID = the ith line in `count.txt`
-- The existence of an edge i between two vertices u and v is represented by the entry of ith line in `links_train.txt`
-- The `links_test.txt` and `link_test_n.txt` are created to identify the edges exclusive to the testing set. `links_test.txt` includes all the true MUX connections while `link_test_n.txt` includes all the false MUX connections
-- The `cell.txt` file includes the mapping between node IDs and gate instances
-- The `c1908_K32.bench` file represents the locked circuit

***2) Train MuxLink***
```sh
$ cd ../
$ python Main.py --file-name c1908_K32_DMUX --train-name links_train.txt  --test-name links_test.txt --testneg-name link_test_n.txt --hop 3  --save-model > Log_train_c1908_DMUX_K32.txt
```
***3) Get the predictions***
```sh
$ python Main.py  --file-name c1908_DMUX_K32 --train-name links_train.txt  --test-name links_test.txt --hop 3  --only-predict > Log_pos_predict_c1908_DMUX_K32.txt
$ python Main.py  --file-name c1908_DMUX_K256 --train-name links_train.txt  --test-name  link_test_n.txt --hop 3  --only-predict > Log_neg_predict_c1908_DMUX_K32.txt
```
- The likelihoods for the links will be dumped in `links_test_3__pred.txt` and `link_test_n_3__pred.txt`. Here, `3` represents the hop size

***4) Parse the predictions***
 ```sh
 $perl break_DMUX.pl c1908_K32_DMUX 0.01 3
```
- Here, 0.01 is the  threshold value (th) explained in the paper. it can be between 0 and 1.
- 3 represents the hop size.
### Citation & Acknowledgement
If you find the code useful, please cite our paper:
* MuxLink 2022:
```
@INPROCEEDINGS{muxlink,
  author={Alrahis, Lilas and Patnaik, Satwik and Shafique, Muhammad and Sinanoglu, Ozgur},
  booktitle={2022 Design, Automation Test in Europe Conference Exhibition (DATE)}, 
  title={MuxLink: Circumventing Learning-Resilient MUX-Locking Using Graph Neural Network-based Link Prediction}, 
  year={2022},
  pages={702-707},
 }
```
We owe many thanks to Muhan Zhang for making his [SEAL](https://github.com/muhanzhang/SEAL) code available.
