# GCRA_Rec
This is our Tensorflow implementation for our KIS 2026 paper:

>Dawed Omer Ahmed, Venkateswara Rao Kagita ,Vikas Kumar(2026). GCRA-Rec: A Graph Convolutional Recurrent Attention Recommender Model for Dynamic Relevance Weighting of Historical Interactions, [Paper in arXiv]().

Contributors: Dawed Omer Ahmed, Venkateswara Rao Kagita ,Vikas Kumar(2026).



## Introduction
In this work, we designed a Graph-Convolutional Recurrent Attention Recommender (GCRA-Rec) model to capture collaborative signals from the neighborhood and dynamically weight historical interactions from a user’s temporal history for improved recommendation.

## Environment Requirement
The code has been tested running under Python 3.9.12. The required packages are as follows:
* python == 3.9.12
* tensorflow == 2.9.1
* numpy == 1.21.5
* scipy == 1.7.3
* sklearn == 1.0.2
## C++ evaluator
We have implemented C++ code to output metrics during and after training, which is much more efficient than python evaluator. It needs to be compiled first using the following command. 
```
python setup.py build_ext --inplace
```
After compilation, the C++ code will run by default instead of Python code.

## Examples to run a 3-layer GCN
The instruction of commands has been clearly stated in the codes (see the parser function in GCRA-Rec/utility/parser.py).
### MovieLens-100k dataset
* Command
```
python run_GCRA_Rec.py --dataset ml-100kprocessed
```
* Output log :
```
eval_score_matrix_foldout with python
n_users=1212, n_items=3708
n_interactions=53159
n_train=42048, n_test=11111, sparsity=0.01183
...
Building user sequences...
Users with valid sequences: 1092 out of 1212
Average sequence length: 36.56
Sequential weight: 0.001
...
Epoch 1 [9.7s]: train==[0.56935=0.54036 + 0.00010 + 0.02889]
Epoch 2 [7.5s]: train==[0.43097=0.40523 + 0.00044 + 0.02530]
    ...
Epoch 400 [28.1s + 10.8s]: test==[0.46454=0.40009 + 0.03460 + 0.02985], recall=[0.12564], precision=[0.03727], ndcg=[0.08447]
Early stopping is trigger at step: 6 log:0.12564431130886078
Best Iter=[13]@[3745.1s]	recall=[0.13282], precision=[0.03852], hit=[0.03587], ndcg=[0.08886]
```


### Yelp2018 dataset
* Command
```
python run_GCRA_Rec.py --dataset Yelp2018_processeed
```
* Output log :
```
eval_score_matrix_foldout with python
n_users=37450, n_items=25603
n_interactions=963717
n_train=756222, n_test=207495, sparsity=0.00101
...
Building user sequences...
Users with valid sequences: 37450 out of 37450
Average sequence length: 20.19
Sequential weight: 0.005
...
Epoch 1 [95.2s]: train==[0.19785=0.19678 + 0.00007 + 0.00100]
Epoch 2 [94.9s]: train==[0.10738=0.10666 + 0.00017 + 0.00054]

    ...
Epoch 560 [1067.3s + 711.2s]: test==[0.15022=0.13925 + 0.00545 + 0.00552], recall=[0.08532], precision=[0.02004], ndcg=[0.05382]
Early stopping is trigger at step: 10 log:0.08532067388296127
Best Iter=[17]@[120095.3s]	recall=[0.08592], precision=[0.02013], hit=[0.02156], ndcg=[0.05409]
```
### Last.FM dataset
* Command
```
python run_GCRA_Rec.py --dataset LastFM
```
* Output log :
```
eval_score_matrix_foldout with python
n_users=1212, n_items=3708
n_interactions=53159
n_train=42048, n_test=11111, sparsity=0.01183
...
Building user sequences...
Users with valid sequences: 1092 out of 1212
Average sequence length: 36.56
Sequential weight: 0.05
...
Epoch 1 [9.7s]: train==[0.56935=0.54036 + 0.00010 + 0.02889]
Epoch 2 [7.5s]: train==[0.43097=0.40523 + 0.00044 + 0.02530]
    ...
Epoch 400 [28.1s + 10.8s]: test==[0.46454=0.40009 + 0.03460 + 0.02985], recall=[0.12564], precision=[0.03727], ndcg=[0.08447]
Early stopping is trigger at step: 6 log:0.12564431130886078
Best Iter=[13]@[3745.1s]	recall=[0.13282], precision=[0.03852], hit=[0.03587], ndcg=[0.08886]
```
NOTE : the duration of training and testing depends on the running environment.
## Dataset
We provide three processed datasets: MovieLens-100k, Yelp2018 and LastFM.
* `train.txt`
  * Train file.
  * Each line is a user with her/his positive interactions with items: userID\t a list of itemID\n.

* `test.txt`
  * Test file (positive instances).
  * Each line is a user with her/his positive interactions with items: userID\t a list of itemID\n.
  * Note that here we treat all unobserved interactions as the negative instances when reporting performance.
  
* `user_list.txt`
  * User file.
  * Each line is a triplet (org_id, remap_id) for one user, where org_id and remap_id represent the ID of the user in the original and our datasets, respectively.
  
* `item_list.txt`
  * Item file.
  * Each line is a triplet (org_id, remap_id) for one item, where org_id and remap_id represent the ID of the item in the original and our datasets, respectively.

## Efficiency Improvements:
  * Parallelized sampling on CPU
  * C++ evaluation for top-k recommendation

=======
