'''
Created on Jan 14, 2026
Tensorflow Implementation of A Graph Convolutional Recurrent Attention Recommender Model 
for Dynamic Relevance Weighting of Historical Interactions. In KIS 2026.

@author: Dawed Omer Ahmed (do24csr1r07@student.nitw.ac.in)
'''
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run NGCF.")
    parser.add_argument('--weights_path', nargs='?', default='',
                        help='Store model path.')
    parser.add_argument('--data_path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default='',
                        help='Project path.')

    parser.add_argument('--dataset', nargs='?', default='gowalla',
                        help='Choose a dataset from {gowalla, yelp2018, amazon-book, movielens-100k}')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--is_norm', type=int, default=1,
                    help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=1000,
                        help='Number of epoch.')

    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--layer_size', nargs='?', default='[64, 64, 64, 64]',
                        help='Output sizes of every layer')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')

    parser.add_argument('--regs', nargs='?', default='[1e-5,1e-5,1e-2]',
                        help='Regularizations.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')

    parser.add_argument('--model_type', nargs='?', default='lightgcn',
                        help='Specify the name of model (lightgcn).')
    parser.add_argument('--adj_type', nargs='?', default='pre',
                        help='Specify the type of the adjacency (laplacian) matrix from {plain, norm, mean}.')
    # parser.add_argument('--alg_type', nargs='?', default='lightgcn',
    #                     help='Specify the type of the graph convolutional layer from {ngcf, gcn, gcmc}.')
    parser.add_argument('--alg_type', nargs='?', default='lightgcn',
                        help='Specify the type of the graph convolutional layer from '
                             '{ngcf, gcn, gcmc, lightgcn, neumf, gat, duallightgcn, '
                             'sasrec, bert4rec, gru4rec}.')

    # parser.add_argument('--max_seq_len', type=int, default=50,
    #                     help='Maximum sequence length for sequential models.')
    # parser.add_argument('--num_heads', type=int, default=2,
    #                     help='Number of attention heads for SASRec/BERT4Rec.')
    parser.add_argument('--mask_prob', type=float, default=0.2,
                        help='Masking probability for BERT4Rec.')
                        
    parser.add_argument('--ablation', type=str, default='full',
    choices=['full','gcn_only','gcn_gru','gcn_attn','gru_only','attn_only','gru_attn'])
    parser.add_argument('--ablationfusion', type=str, default='full',
                    help='Fusion type: full, concat, fixed_gate, learnable_gate, attention, gcn_only')

    parser.add_argument('--gpu_id', type=int, default=0,
                        help='0 for NAIS_prod, 1 for NAIS_concat')

    parser.add_argument('--node_dropout_flag', type=int, default=0,
                        help='0: Disable node dropout, 1: Activate node dropout')
    parser.add_argument('--node_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

    parser.add_argument('--Ks', nargs='?', default='[20]',
                        help='Top k(s) recommend')

    parser.add_argument('--save_flag', type=int, default=0,
                        help='0: Disable model saver, 1: Activate model saver')

    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    parser.add_argument('--report', type=int, default=0,
                        help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')
    
    
    # Add these arguments to your parse
    # Add these to parse_args() function after existing arguments:
    parser.add_argument('--grid_search', type=int, default=0,
                        help='0: Normal run, 1: Grid search mode')
    parser.add_argument('--grid_regs', nargs='?', default='[1e-5,2e-5,3e-5,5e-5,8e-5]',
                        help='Regularization values for grid search')
    parser.add_argument('--grid_lr', nargs='?', default='[0.005,0.008,0.01,0.012]',
                        help='Learning rate values for grid search')
    parser.add_argument('--grid_seq_weight', nargs='?', default='[0.01,0.015]',
                        help='Sequential weight values for grid search')
    parser.add_argument('--grid_seq_dropout', nargs='?', default='[0.15,0.2,0.25,0.3]',
                        help='Sequential dropout values for grid search')
    # Sequential component parameters
    parser.add_argument('--num_heads', type=int, default=1,
                    help='Number of attention heads.')
    parser.add_argument('--seq_weight', type=float, default=0.0,
                        help='Weight for sequential loss. Set to 0.0 for pure LightGCN.')
    parser.add_argument('--max_seq_len', type=int, default=50,
                        help='Maximum sequence length for user interactions.')
    parser.add_argument('--min_seq_len', type=int, default=3,
                        help='Minimum sequence length for valid users.')
    parser.add_argument('--gru_hidden_size', type=int, default=64,
                        help='Hidden size for GRU in sequential component.')
    parser.add_argument('--seq_dropout', type=float, default=0.1,
                        help='Dropout rate for sequential component.')
    return parser.parse_args()

