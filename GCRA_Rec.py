'''
Created on Jan 14, 2026
Tensorflow Implementation of A Graph Convolutional Recurrent Attention Recommender Model 
for Dynamic Relevance Weighting of Historical Interactions. In KIS 2026.

@author: Dawed Omer Ahmed (do24csr1r07@student.nitw.ac.in)
'''
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=" + os.environ["CONDA_PREFIX"]
import sys
import threading
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.client import device_lib
from utility.helper import *
from utility.batch_test import *
from utility.parser import parse_args
from utility.load_data import *
from time import time
from utility.batch_test import data_generator, args
# from utility.visualize_tsne import visualize_tsne
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Parse arguments and load data at module level
# args = parse_args()
# data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size)

cpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'CPU']

class HybridSeqGCN(object):
    def __init__(self, data_config, pretrain_data):
        # Argument settings
        self.model_type = 'HybridSeqGCN'
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type
        self.pretrain_data = pretrain_data
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_fold = 100
        self.norm_adj = data_config['norm_adj']
        self.n_nonzero_elems = self.norm_adj.count_nonzero()
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        
        # Sequential settings
        self.max_seq_len = getattr(args, 'max_seq_len', 50)
        self.min_seq_len = getattr(args, 'min_seq_len', 3)
        self.hidden_size = getattr(args, 'gru_hidden_size', 64)
        self.num_heads = getattr(args, 'num_heads', 1)
        
        # Check if sequential component should be enabled
        self.seq_weight = getattr(args, 'seq_weight', 0.0)
        self.has_sequential_data = (self.seq_weight > 0)
        
        print(f"Sequential data: {'ENABLED' if self.has_sequential_data else 'DISABLED'}")
        if self.has_sequential_data:
            print(f"Sequential weight: {self.seq_weight}")
        
        self.log_dir = self.create_model_str()
        self.verbose = args.verbose
        self.Ks = eval(args.Ks)
        
        # Create placeholders
        self._create_placeholders()
        
        # Create TensorBoard summaries
        self._create_summaries()

        # Initialize weights
        self.weights = self._init_weights()

        # Create GCN embeddings - EXACTLY like LightGCN
        self._create_gcn_embeddings()

        # Create user embeddings (with or without sequential component)
        self._create_user_embeddings()

        # Create item embeddings for batch
        self._create_item_embeddings()

        # Create inference operation
        self._create_inference()

        # Create loss and optimizer
        self._create_loss_optimizer()

    def _create_placeholders(self):
        """Create TensorFlow placeholders"""
        # Traditional placeholders - EXACTLY like LightGCN
        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))
        
        # Dropout placeholders - EXACTLY like LightGCN
        self.node_dropout_flag = args.node_dropout_flag
        self.node_dropout = tf.placeholder(tf.float32, shape=[None])
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None])
        
        # Sequential placeholders (only when needed)
        if self.has_sequential_data:
            self.user_seq = tf.placeholder(tf.int32, shape=(None, None), name='user_seq')
            self.seq_len = tf.placeholder(tf.int32, shape=(None,), name='seq_len')
            self.target_item = tf.placeholder(tf.int32, shape=(None,), name='target_item')
            self.seq_dropout = tf.placeholder(tf.float32)

    def _create_summaries(self):
        """Create TensorBoard summaries - EXACTLY like LightGCN"""
        with tf.name_scope('TRAIN_LOSS'):
            self.train_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('train_loss', self.train_loss)
            self.train_mf_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('train_mf_loss', self.train_mf_loss)
            self.train_emb_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('train_emb_loss', self.train_emb_loss)
            self.train_reg_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('train_reg_loss', self.train_reg_loss)
            if self.has_sequential_data:
                self.train_seq_loss = tf.placeholder(tf.float32)
                tf.summary.scalar('train_seq_loss', self.train_seq_loss)
        self.merged_train_loss = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, 'TRAIN_LOSS'))
        
        with tf.name_scope('TRAIN_ACC'):
            self.train_rec_first = tf.placeholder(tf.float32)
            tf.summary.scalar('train_rec_first', self.train_rec_first)
            self.train_rec_last = tf.placeholder(tf.float32)
            tf.summary.scalar('train_rec_last', self.train_rec_last)
            self.train_ndcg_first = tf.placeholder(tf.float32)
            tf.summary.scalar('train_ndcg_first', self.train_ndcg_first)
            self.train_ndcg_last = tf.placeholder(tf.float32)
            tf.summary.scalar('train_ndcg_last', self.train_ndcg_last)
        self.merged_train_acc = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, 'TRAIN_ACC'))

        with tf.name_scope('TEST_LOSS'):
            self.test_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('test_loss', self.test_loss)
            self.test_mf_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('test_mf_loss', self.test_mf_loss)
            self.test_emb_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('test_emb_loss', self.test_emb_loss)
            self.test_reg_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('test_reg_loss', self.test_reg_loss)
            if self.has_sequential_data:
                self.test_seq_loss = tf.placeholder(tf.float32)
                tf.summary.scalar('test_seq_loss', self.test_seq_loss)
        self.merged_test_loss = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, 'TEST_LOSS'))

        with tf.name_scope('TEST_ACC'):
            self.test_rec_first = tf.placeholder(tf.float32)
            tf.summary.scalar('test_rec_first', self.test_rec_first)
            self.test_rec_last = tf.placeholder(tf.float32)
            tf.summary.scalar('test_rec_last', self.test_rec_last)
            self.test_ndcg_first = tf.placeholder(tf.float32)
            tf.summary.scalar('test_ndcg_first', self.test_ndcg_first)
            self.test_ndcg_last = tf.placeholder(tf.float32)
            tf.summary.scalar('test_ndcg_last', self.test_ndcg_last)
        self.merged_test_acc = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, 'TEST_ACC'))

    def _create_gcn_embeddings(self):
        """Create GCN embeddings - EXACTLY like LightGCN"""
        if self.alg_type == 'lightgcn':
            self.ua_embeddings, self.ia_embeddings = self._create_lightgcn_embed()
        elif self.alg_type == 'ngcf':
            self.ua_embeddings, self.ia_embeddings = self._create_ngcf_embed()
        elif self.alg_type == 'gcn':
            self.ua_embeddings, self.ia_embeddings = self._create_gcn_embed()
        elif self.alg_type == 'gcmc':
            self.ua_embeddings, self.ia_embeddings = self._create_gcmc_embed()

    def _create_user_embeddings(self):
        """Create final user embeddings - PURE LIGHTGCN when seq_weight=0"""
        """Create final user embeddings with advanced fusion when sequential data is available"""
        # Get GCN user embeddings for batch
        self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
        
        if self.has_sequential_data:
            # Step 1: Process sequences through GRU + Self-attention
            self.seq_embeddings = self._create_sequential_embed()
            
            # Step 2: ADVANCED FUSION - Append GCN user embedding to sequence outputs and reapply attention
            seq_outputs = self.seq_embeddings  # [batch, seq_len, emb_dim]
            gcn_outputs = self.u_g_embeddings  # [batch, emb_dim]
            
            # Expand GCN for sequence dimension
            gcn_exp = tf.expand_dims(gcn_outputs, axis=1)  # [batch, 1, emb_dim]
            
            # Concatenate: [batch, seq_len+1, emb_dim]
            combined = tf.concat([seq_outputs, gcn_exp], axis=1)
            
            # Apply attention pooling over combined
            with tf.variable_scope('fusion_attention_pooling', reuse=tf.AUTO_REUSE):
                # Initialize attention weights if they don't exist
                if not hasattr(self, 'fusion_weights_initialized'):
                    with tf.variable_scope('fusion_weights'):
                        self.W_att = tf.get_variable('W_att', [self.emb_dim, self.emb_dim], 
                                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
                        self.b_att = tf.get_variable('b_att', [self.emb_dim], 
                                                    initializer=tf.zeros_initializer())
                        self.w_att = tf.get_variable('w_att', [self.emb_dim, 1], 
                                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
                    self.fusion_weights_initialized = True
                
                logits = tf.nn.tanh(tf.tensordot(combined, self.W_att, axes=1) + self.b_att)
                logits = tf.tensordot(logits, self.w_att, axes=1)  # [batch, seq_len+1]
                logits = tf.squeeze(logits, axis=-1)  # Remove last dimension
                seq_mask = tf.sequence_mask(self.seq_len, maxlen=tf.shape(seq_outputs)[1], dtype=tf.float32)
                gcn_mask = tf.ones_like(seq_mask[:, :1])  # GCN token always valid
                full_mask = tf.concat([seq_mask, gcn_mask], axis=1)  # [batch, seq_len+1]
                logits = logits + (1.0 - full_mask) * (-1e9)
                weights = tf.nn.softmax(logits, axis=-1)  # [batch, seq_len+1]
                self.final_user_embeddings = tf.reduce_sum(combined * tf.expand_dims(weights, -1), axis=1)  # [batch, emb_dim]
            
            # L2 normalize final embedding
            self.final_user_embeddings = tf.nn.l2_normalize(self.final_user_embeddings, axis=1)
        else:
            # PURE LIGHTGCN - no sequential components
            self.final_user_embeddings = self.u_g_embeddings

    def _create_item_embeddings(self):
        """Create item embeddings for batch - EXACTLY like LightGCN"""
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_items)
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.neg_items)
        self.u_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users)
        self.pos_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.pos_items)
        self.neg_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.neg_items)

    def _create_inference(self):
        """Create inference operation - EXACTLY like LightGCN"""
        self.batch_ratings = tf.matmul(self.final_user_embeddings, self.pos_i_g_embeddings, 
                                     transpose_a=False, transpose_b=True)
        # FIX - add a separate eval placeholder:
        self.eval_items = tf.placeholder(tf.int32, shape=(None,))
        eval_item_emb = tf.nn.embedding_lookup(self.ia_embeddings, self.eval_items)
        self.batch_ratings = tf.matmul(self.final_user_embeddings, eval_item_emb, transpose_b=True)

    def _create_loss_optimizer(self):
        """Create loss functions and optimizer - EXACTLY like LightGCN when seq_weight=0"""
        # BPR loss - EXACTLY like LightGCN
        self.mf_loss, self.emb_loss, self.reg_loss = self.create_bpr_loss(
            self.final_user_embeddings, self.pos_i_g_embeddings, self.neg_i_g_embeddings)
        
        if self.has_sequential_data:
            self.seq_loss_raw = self.create_sequential_loss()
            # Use sequential weight directly
            self.seq_loss = self.seq_weight * self.seq_loss_raw
            self.loss = self.mf_loss + self.emb_loss + self.seq_weight * self.seq_loss_raw
        else:
            # PURE LIGHTGCN - no sequential loss
            self.seq_loss_raw = tf.constant(0.0, dtype=tf.float32)
            self.seq_loss = tf.constant(0.0, dtype=tf.float32)
            self.loss = self.mf_loss + self.emb_loss

        # Standard Adam optimizer - EXACTLY like LightGCN
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        # global_step = tf.Variable(0, trainable=False)
        # steps_per_epoch = 1524  # Yelp2018 with batch_size=1024
        # decay_steps = steps_per_epoch * 100  # decay every 100 epochs
        # decay_rate = 0.96  # % drop

        # decayed_lr = tf.train.exponential_decay(
        #     learning_rate=self.lr,    # 0.005
        #     global_step=global_step,
        #     decay_steps=decay_steps,
        #     decay_rate=decay_rate,
        #     staircase=True
        # )

        # self.opt = tf.train.AdamOptimizer(learning_rate=decayed_lr).minimize(
        #     self.loss, global_step=global_step
        # )

    def create_model_str(self):
        """Create model string for logging"""
        if self.has_sequential_data:
            log_dir = f'/{self.model_type}/layers_{self.n_layers}/dim_{self.emb_dim}'
            log_dir += f'/seq_{self.max_seq_len}/hidden_{self.hidden_size}'
        else:
            log_dir = f'/{self.model_type}_GCNOnly/layers_{self.n_layers}/dim_{self.emb_dim}'
        log_dir += f'/{args.dataset}/lr_{self.lr}/reg_{self.decay}'
        return log_dir

    def _init_weights(self):
        """Initialize model weights - EXACTLY like LightGCN"""
        all_weights = dict()
        initializer = tf.random_normal_initializer(stddev=0.01)
        
        if self.pretrain_data is None:
            all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embedding')
            all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name='item_embedding')
            print('Using random initialization')
        else:
            all_weights['user_embedding'] = tf.Variable(initial_value=self.pretrain_data['user_embed'], trainable=True,
                                                        name='user_embedding', dtype=tf.float32)
            all_weights['item_embedding'] = tf.Variable(initial_value=self.pretrain_data['item_embed'], trainable=True,
                                                        name='item_embedding', dtype=tf.float32)
            print('Using pretrained initialization')
            
        self.weight_size_list = [self.emb_dim] + self.weight_size
        
        # GCN layer weights (only for NGCF, GCN, GCMC - not LightGCN)
        if self.alg_type in ['ngcf', 'gcn', 'gcmc']:
            for k in range(self.n_layers):
                all_weights[f'W_gc_{k}'] = tf.Variable(
                    initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name=f'W_gc_{k}')
                all_weights[f'b_gc_{k}'] = tf.Variable(
                    initializer([1, self.weight_size_list[k+1]]), name=f'b_gc_{k}')

                if self.alg_type == 'ngcf':
                    all_weights[f'W_bi_{k}'] = tf.Variable(
                        initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name=f'W_bi_{k}')
                    all_weights[f'b_bi_{k}'] = tf.Variable(
                        initializer([1, self.weight_size_list[k + 1]]), name=f'b_bi_{k}')

                if self.alg_type == 'gcmc':
                    all_weights[f'W_mlp_{k}'] = tf.Variable(
                        initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name=f'W_mlp_{k}')
                    all_weights[f'b_mlp_{k}'] = tf.Variable(
                        initializer([1, self.weight_size_list[k+1]]), name=f'b_mlp_{k}')

        return all_weights

    def _split_A_hat(self, X):
        """Split adjacency matrix - EXACTLY like LightGCN"""
        A_fold_hat = []
        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            end = self.n_users + self.n_items if i_fold == self.n_fold - 1 else (i_fold + 1) * fold_len
            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _split_A_hat_node_dropout(self, X):
        """Split adjacency matrix with node dropout - EXACTLY like LightGCN"""
        A_fold_hat = []
        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            end = self.n_users + self.n_items if i_fold == self.n_fold - 1 else (i_fold + 1) * fold_len
            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))
        return A_fold_hat

    def _create_lightgcn_embed(self):
        
        if self.node_dropout_flag:
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)
        
        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
        all_embeddings = [ego_embeddings]
        
        for k in range(0, self.n_layers):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            side_embeddings = tf.concat(temp_embed, 0)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings=tf.stack(all_embeddings,1)
        all_embeddings=tf.reduce_mean(all_embeddings,axis=1,keepdims=False)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def _create_sequential_embed(self):
        """Create sequential embeddings with GRU + Self-attention"""
        """
        Sequential Encoder:
        - Embeds user sequences using GRU + Self-Attention.
        - Returns contextualized embeddings for each timestep: [batch, seq_len, emb_dim]
        """
        # Use GCN item embeddings for sequences
        seq_item_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.user_seq)
        with tf.variable_scope('position_encoding', reuse=tf.AUTO_REUSE):
            pos_embeddings = tf.get_variable('pos_emb', [self.max_seq_len, self.emb_dim],
                                initializer=tf.truncated_normal_initializer(stddev=0.01))
            seq_len_dynamic = tf.shape(self.user_seq)[1]
            seq_item_embeddings = seq_item_embeddings + pos_embeddings[:seq_len_dynamic, :]
        
        # Create sequence mask
        mask = tf.sequence_mask(self.seq_len, maxlen=tf.shape(self.user_seq)[1], dtype=tf.float32)
        
        # GRU Layer
        with tf.variable_scope('gru_encoder', reuse=tf.AUTO_REUSE):
            gru_cell = tf.nn.rnn_cell.GRUCell(self.emb_dim)
            gru_outputs, _ = tf.nn.dynamic_rnn(
                gru_cell,
                seq_item_embeddings,
                sequence_length=self.seq_len,
                dtype=tf.float32
            )
        gru_outputs = tf.nn.dropout(gru_outputs, keep_prob=1 - self.seq_dropout)
        
        # # Self-Attention Layer
        # with tf.variable_scope('self_attention', reuse=tf.AUTO_REUSE):
        #     d_k = self.emb_dim
        #     Q = tf.layers.dense(gru_outputs, d_k, use_bias=False, name='query_transform')
        #     K = tf.layers.dense(gru_outputs, d_k, use_bias=False, name='key_transform')
        #     V = tf.layers.dense(gru_outputs, d_k, use_bias=False, name='value_transform')
            
        #     # Compute attention scores
        #     scale = tf.sqrt(tf.cast(d_k, tf.float32))
        #     attention_scores = tf.matmul(Q, K, transpose_b=True) / scale  # [batch, seq_len, seq_len]
            
        #     # Apply mask for valid positions
        #     mask_expanded = tf.expand_dims(mask, 1)  # [batch, 1, seq_len]
        #     mask_tiled = tf.tile(mask_expanded, [1, tf.shape(attention_scores)[1], 1])  # [batch, seq_len, seq_len]
            
        #     attention_scores = tf.where(
        #         tf.equal(mask_tiled, 0),
        #         tf.ones_like(attention_scores) * (-1e9),
        #         attention_scores
        #     )
            
        #     attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        #     attended_outputs = tf.matmul(attention_weights, V)  # [batch, seq_len, emb_dim]
            
        #     # Residual connection
        #     attended_outputs = attended_outputs + gru_outputs
        
        # return attended_outputs  # shape: [batch, seq_len, emb_dim]
        # REPLACE the self_attention block WITH:
        num_heads = self.num_heads
        head_dim = self.emb_dim // num_heads
        with tf.variable_scope('self_attention', reuse=tf.AUTO_REUSE):
            Q = tf.layers.dense(gru_outputs, self.emb_dim, use_bias=False, name='query_transform')
            K = tf.layers.dense(gru_outputs, self.emb_dim, use_bias=False, name='key_transform')
            V = tf.layers.dense(gru_outputs, self.emb_dim, use_bias=False, name='value_transform')

            # Split into heads: [batch, heads, seq_len, head_dim]
            def split_heads(x):
                batch = tf.shape(x)[0]
                seq = tf.shape(x)[1]
                x = tf.reshape(x, [batch, seq, num_heads, head_dim])
                return tf.transpose(x, [0, 2, 1, 3])

            Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

            scale = tf.sqrt(tf.cast(head_dim, tf.float32))
            attention_scores = tf.matmul(Q, K, transpose_b=True) / scale  # [batch, heads, seq, seq]

            mask_expanded = tf.expand_dims(tf.expand_dims(mask, 1), 1)  # [batch, 1, 1, seq]
            attention_scores = tf.where(
                tf.equal(tf.tile(mask_expanded, [1, num_heads, tf.shape(gru_outputs)[1], 1]), 0),
                tf.ones_like(attention_scores) * (-1e9),
                attention_scores
            )

            attention_weights = tf.nn.softmax(attention_scores, axis=-1)
            attended = tf.matmul(attention_weights, V)  # [batch, heads, seq, head_dim]

            # Merge heads back: [batch, seq, emb_dim]
            batch = tf.shape(attended)[0]
            seq = tf.shape(attended)[2]
            attended_outputs = tf.transpose(attended, [0, 2, 1, 3])
            attended_outputs = tf.reshape(attended_outputs, [batch, seq, self.emb_dim])

            # Residual connection
            attended_outputs = attended_outputs + gru_outputs
        return attended_outputs

    def create_sequential_loss(self):
        """Create simplified sequential loss"""
        if not self.has_sequential_data:
            return tf.constant(0.0, dtype=tf.float32)
            
        # Use final_user_embeddings for sequential prediction
        target_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.target_item)
        pos_scores = tf.reduce_sum(tf.multiply(self.final_user_embeddings, target_embeddings), axis=1)
        
        # Simple negative sampling
        num_negs = 1
        neg_items = tf.random.uniform([tf.shape(self.target_item)[0], num_negs], 0, self.n_items, dtype=tf.int32)
        neg_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, neg_items)
        
        # Compute negative scores
        user_expanded = tf.expand_dims(self.final_user_embeddings, 1)
        neg_scores = tf.reduce_sum(tf.multiply(user_expanded, neg_embeddings), axis=2)
        neg_scores = tf.squeeze(neg_scores, axis=1)
        
        # BPR loss
        seq_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))
        
        return seq_loss

    def create_bpr_loss(self, users, pos_items, neg_items):
        """Create BPR loss - EXACTLY like LightGCN"""
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        
        # EXACTLY like LightGCN regularization
        regularizer = (tf.nn.l2_loss(self.u_g_embeddings_pre) + 
                      tf.nn.l2_loss(self.pos_i_g_embeddings_pre) + 
                      tf.nn.l2_loss(self.neg_i_g_embeddings_pre)) / self.batch_size
        
        mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))
        emb_loss = self.decay * regularizer
        reg_loss = tf.constant(0.0, tf.float32, [1])

        return mf_loss, emb_loss, reg_loss

    def _create_ngcf_embed(self):
        """Create NGCF embeddings"""
        A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj) if self.node_dropout_flag else self._split_A_hat(self.norm_adj)
        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
        all_embeddings = [ego_embeddings]

        for k in range(self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            side_embeddings = tf.concat(temp_embed, 0)
            sum_embeddings = tf.nn.leaky_relu(tf.matmul(side_embeddings, self.weights[f'W_gc_{k}']) + self.weights[f'b_gc_{k}'])
            bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
            bi_embeddings = tf.nn.leaky_relu(tf.matmul(bi_embeddings, self.weights[f'W_bi_{k}']) + self.weights[f'b_bi_{k}'])
            ego_embeddings = sum_embeddings + bi_embeddings
            norm_embeddings = tf.nn.l2_normalize(ego_embeddings, axis=1)
            all_embeddings.append(norm_embeddings)

        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def _create_gcn_embed(self):
        """Create GCN embeddings"""
        A_fold_hat = self._split_A_hat(self.norm_adj)
        embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
        all_embeddings = [embeddings]

        for k in range(self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))
            embeddings = tf.concat(temp_embed, 0)
            embeddings = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights[f'W_gc_{k}']) + self.weights[f'b_gc_{k}'])
            all_embeddings.append(embeddings)

        all_embeddings = tf.concat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def _create_gcmc_embed(self):
        """Create GCMC embeddings"""
        A_fold_hat = self._split_A_hat(self.norm_adj)
        embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
        all_embeddings = []

        for k in range(self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))
            embeddings = tf.concat(temp_embed, 0)
            embeddings = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights[f'W_gc_{k}']) + self.weights[f'b_gc_{k}'])
            mlp_embeddings = tf.matmul(embeddings, self.weights[f'W_mlp_{k}']) + self.weights[f'b_mlp_{k}']
            all_embeddings.append(mlp_embeddings)
        all_embeddings = tf.concat(all_embeddings, 1)

        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        """Convert sparse matrix to sparse tensor"""
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)
        
    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """Dropout for sparse tensors"""
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob + tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(X, dropout_mask)
        return pre_out * tf.div(1., keep_prob)


def load_pretrained_data():
    """Load pretrained embeddings"""
    pretrain_path = f'{args.proj_path}pretrain/{args.dataset}/embedding.npz'
    try:
        pretrain_data = np.load(pretrain_path)
        print('Loaded pretrained embeddings.')
        return pretrain_data
    except Exception:
        return None


def add_sequential_methods_to_data_generator(data_gen, max_seq_len=50, min_seq_len=3):
    """Add sequential data generation methods to data generator"""
    """Add sequential data generation methods to data generator"""
    if not getattr(args, 'seq_weight', 0.0) > 0:
        print("Sequential component disabled - skipping sequence building")
        return data_gen
        
    print("Building user sequences...")
    user_sequences = {}
    
    # Build sequences for each user
    for user_id in range(data_gen.n_users):
        if user_id in data_gen.train_items:
            items = list(data_gen.train_items[user_id])
            # CRITICAL CHANGE: PRESERVE temporal order instead of shuffling
            # The data appears to be in chronological order already
            user_sequences[user_id] = items if len(items) >= min_seq_len else []
        else:
            user_sequences[user_id] = []
    
    
    # valid_users = [u for u, seq in user_sequences.items() if len(seq) >= min_seq_len]
    valid_users = [u for u, seq in user_sequences.items() if len(seq) > min_seq_len]
    print(f"Users with valid sequences: {len(valid_users)} out of {data_gen.n_users}")
    
    if valid_users:
        avg_len = np.mean([len(seq) for seq in user_sequences.values() if len(seq) > 0])
        print(f"Average sequence length: {avg_len:.2f}")
        print("IMPORTANT: Preserving temporal order from data (no random shuffle)")
    
    def sample_sequences(users=None, batch_size=1024):
        if not valid_users:
            return (np.zeros((batch_size, max_seq_len), dtype=np.int32),
                    np.ones(batch_size, dtype=np.int32),
                    np.zeros(batch_size, dtype=np.int32))

        if users is not None:
            users = [u for u in users
                    if u in user_sequences and len(user_sequences[u]) > min_seq_len]
            if len(users) == 0:
                users = np.random.choice(valid_users, size=batch_size, replace=True).tolist()
        else:
            users = np.random.choice(valid_users, size=batch_size, replace=True).tolist()

        sequences, seq_lengths, targets = [], [], []

        for user_id in users:
            seq = user_sequences.get(user_id, [])

            if len(seq) < 2:
                sequences.append([0] * max_seq_len)
                seq_lengths.append(1)
                targets.append(0)
                continue
            elif len(seq) == 2:
                input_seq = [seq[0]]
                target = seq[1]
            elif len(seq) <= min_seq_len + 1:
                input_seq = seq[:-1]
                target = seq[-1]
            else:
                target_pos = np.random.randint(min_seq_len, len(seq))
                input_seq = seq[max(0, target_pos - max_seq_len):target_pos]
                target = seq[target_pos]

            if len(input_seq) < max_seq_len:
                padded = input_seq + [0] * (max_seq_len - len(input_seq))
            else:
                padded = input_seq[:max_seq_len]

            sequences.append(padded)
            seq_lengths.append(min(len(input_seq), max_seq_len))
            targets.append(target)

        return (np.array(sequences, dtype=np.int32),
                np.array(seq_lengths, dtype=np.int32),
                np.array(targets, dtype=np.int32))
    
    def sample_test_sequences(batch_size=1024):
        """Sample test sequences"""
        test_valid_users = [u for u in valid_users if hasattr(data_gen, 'test_set') and u in data_gen.test_set]
        
        if not test_valid_users:
            return (np.zeros((batch_size, max_seq_len), dtype=np.int32), 
                   np.ones(batch_size, dtype=np.int32),
                   np.zeros(batch_size, dtype=np.int32))
        
        sampled_users = np.random.choice(test_valid_users, size=batch_size, replace=True)
        sequences, seq_lengths, targets = [], [], []
        
        for user_id in sampled_users:
            seq = user_sequences[user_id]
            # Use the most recent items as context (preserving order)
            input_seq = seq[-max_seq_len:] if len(seq) > max_seq_len else seq
            
            # Pad sequence
            padded = input_seq + [0] * (max_seq_len - len(input_seq)) if len(input_seq) < max_seq_len else input_seq
            sequences.append(padded)
            seq_lengths.append(min(len(input_seq), max_seq_len))
            targets.append(np.random.choice(list(data_gen.test_set[user_id])))
        
        return (np.array(sequences, dtype=np.int32),
                np.array(seq_lengths, dtype=np.int32),
                np.array(targets, dtype=np.int32))
    
    # Add methods to data generator
    data_gen.sample_sequences = sample_sequences
    data_gen.sample_test_sequences = sample_test_sequences
    data_gen.user_sequences = user_sequences
    
    return data_gen


# Threading classes for parallel processing
class sample_thread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.data = None
        self.seq_data = None
        
    def run(self):
        with tf.device(cpus[0]):
            try:
                self.data = data_generator.sample()
                if hasattr(data_generator, 'sample_sequences') and getattr(args, 'seq_weight', 0.0) > 0:
                    # self.seq_data = data_generator.sample_sequences()
                    self.seq_data = data_generator.sample_sequences(users=self.data[0])
                else:
                    self.seq_data = None
            except Exception as e:
                print(f"Sampling error: {e}")
                batch_size = getattr(args, 'batch_size', 1024)
                self.data = (np.zeros(batch_size, dtype=np.int32),) * 3
                self.seq_data = None


class sample_thread_test(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.data = None
        self.seq_data = None
        
    def run(self):
        with tf.device(cpus[0]):
            try:
                self.data = data_generator.sample_test()
                if hasattr(data_generator, 'sample_test_sequences') and getattr(args, 'seq_weight', 0.0) > 0:
                    self.seq_data = data_generator.sample_test_sequences()
                else:
                    self.seq_data = None
            except Exception as e:
                print(f"Test sampling error: {e}")
                batch_size = getattr(args, 'batch_size', 1024)
                self.data = (np.zeros(batch_size, dtype=np.int32),) * 3
                self.seq_data = None


class train_thread(threading.Thread):
    def __init__(self, model, sess, sample):
        threading.Thread.__init__(self)
        self.model = model
        self.sess = sess
        self.sample = sample
        self.data = None
        
    def run(self):
        try:
            users, pos_items, neg_items = self.sample.data
            seq_data = self.sample.seq_data
            
            # Base feed dict - EXACTLY like LightGCN
            feed_dict = {
                self.model.users: users,
                self.model.pos_items: pos_items,
                self.model.neg_items: neg_items,
                self.model.node_dropout: eval(args.node_dropout),
                self.model.mess_dropout: eval(args.mess_dropout)
            }
        
            # Add sequential data if needed
            if self.model.has_sequential_data:
                feed_dict[self.model.seq_dropout] = getattr(args, 'seq_dropout', 0.1)
                
                if seq_data is not None:
                    user_seqs, seq_lens, target_items = seq_data
                    
                    min_batch_size = min(len(users), len(user_seqs))
                    feed_dict.update({
                        self.model.user_seq: user_seqs[:min_batch_size],
                        self.model.seq_len: seq_lens[:min_batch_size],
                        self.model.target_item: target_items[:min_batch_size],
                        self.model.users: users[:min_batch_size],
                        self.model.pos_items: pos_items[:min_batch_size],
                        self.model.neg_items: neg_items[:min_batch_size]
                    })
                else:
                    # Dummy sequential data
                    batch_size = len(users)
                    max_seq_len = getattr(args, 'max_seq_len', 50)
                    feed_dict.update({
                        self.model.user_seq: np.zeros((batch_size, max_seq_len), dtype=np.int32),
                        self.model.seq_len: np.ones(batch_size, dtype=np.int32),
                        self.model.target_item: np.zeros(batch_size, dtype=np.int32)
                    })
                
                # Run with sequential loss
                _, batch_loss, batch_mf_loss, batch_emb_loss, batch_seq_loss = self.sess.run(
                    [self.model.opt, self.model.loss, self.model.mf_loss, self.model.emb_loss, self.model.seq_loss],
                    feed_dict=feed_dict
                )
                self.data = [None, batch_loss, batch_mf_loss, batch_emb_loss, batch_seq_loss]
            else:
                # Pure LightGCN mode - no sequential components
                _, batch_loss, batch_mf_loss, batch_emb_loss = self.sess.run(
                    [self.model.opt, self.model.loss, self.model.mf_loss, self.model.emb_loss],
                    feed_dict=feed_dict
                )
                self.data = [None, batch_loss, batch_mf_loss, batch_emb_loss, 0.0]
            
        except Exception as e:
            print(f"Training thread error: {e}")
            self.data = [None, 0.0, 0.0, 0.0, 0.0]


class train_thread_test(threading.Thread):
    def __init__(self, model, sess, sample):
        threading.Thread.__init__(self)
        self.model = model
        self.sess = sess
        self.sample = sample
        
    def run(self):
        try:
            users, pos_items, neg_items = self.sample.data
            seq_data = self.sample.seq_data
            
            # Base feed dict - EXACTLY like LightGCN
            feed_dict = {
                self.model.users: users, 
                self.model.pos_items: pos_items,
                self.model.neg_items: neg_items,
                self.model.node_dropout: [0.] * len(eval(args.node_dropout)),
                self.model.mess_dropout: [0.] * len(eval(args.mess_dropout))
            }
            
            # Add sequential data if needed
            if self.model.has_sequential_data:
                feed_dict[self.model.seq_dropout] = 0.0
                
                if seq_data is not None:
                    user_seqs, seq_lens, target_items = seq_data
                    min_batch_size = min(len(users), len(user_seqs))
                    feed_dict.update({
                        self.model.user_seq: user_seqs[:min_batch_size],
                        self.model.seq_len: seq_lens[:min_batch_size],
                        self.model.target_item: target_items[:min_batch_size],
                        self.model.users: users[:min_batch_size],
                        self.model.pos_items: pos_items[:min_batch_size],
                        self.model.neg_items: neg_items[:min_batch_size]
                    })
                else:
                    # Dummy sequential data
                    batch_size = len(users)
                    max_seq_len = getattr(args, 'max_seq_len', 50)
                    feed_dict.update({
                        self.model.user_seq: np.zeros((batch_size, max_seq_len), dtype=np.int32),
                        self.model.seq_len: np.ones(batch_size, dtype=np.int32),
                        self.model.target_item: np.zeros(batch_size, dtype=np.int32)
                    })
                
                # Run with sequential loss
                self.data = self.sess.run([self.model.loss, self.model.mf_loss, self.model.emb_loss, self.model.seq_loss], 
                                        feed_dict=feed_dict)
            else:
                # Pure LightGCN mode
                batch_loss, batch_mf_loss, batch_emb_loss = self.sess.run(
                    [self.model.loss, self.model.mf_loss, self.model.emb_loss],
                    feed_dict=feed_dict
                )
                self.data = [batch_loss, batch_mf_loss, batch_emb_loss, 0.0]
            
        except Exception as e:
            print(f"Test thread error: {e}")
            self.data = [0.0, 0.0, 0.0, 0.0]


def main():
    """Main training function"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Load data configuration
    plain_adj, norm_adj, mean_adj, pre_adj = data_generator.get_adj_mat()
    
    if args.adj_type == 'plain':
        config_adj = plain_adj
        print('use the plain adjacency matrix')
    elif args.adj_type == 'norm':
        config_adj = norm_adj
        print('use the normalized adjacency matrix')
    elif args.adj_type == 'gcmc':
        config_adj = mean_adj
        print('use the gcmc adjacency matrix')
    elif args.adj_type == 'pre':
        config_adj = pre_adj
        print('use the pre adjacency matrix')
    else:
        config_adj = mean_adj + sp.eye(mean_adj.shape[0])
        print('use the mean adjacency matrix')
    
    config = {
        'n_users': data_generator.n_users,
        'n_items': data_generator.n_items,
        'norm_adj': config_adj
    }

    print('Loading pretrained data...')
    pretrain_data = load_pretrained_data()

    # Add sequential methods to data generator if needed
    if getattr(args, 'seq_weight', 0.0) > 0:
        data_generator_updated = add_sequential_methods_to_data_generator(
            data_generator, 
            max_seq_len=getattr(args, 'max_seq_len', 50),
            min_seq_len=getattr(args, 'min_seq_len', 3)
        )

    # Create model
    model = HybridSeqGCN(data_config=config, pretrain_data=pretrain_data)

    # Initialize session
    sess.run(tf.global_variables_initializer())
    print('Model initialized without pretraining.')

    # Setup TensorBoard
    tensorboard_model_path = 'tensorboard/'
    os.makedirs(tensorboard_model_path, exist_ok=True)
    run_time = 1
    while os.path.exists(f'{tensorboard_model_path}{model.log_dir}/run_{run_time}'):
        run_time += 1
    train_writer = tf.summary.FileWriter(f'{tensorboard_model_path}{model.log_dir}/run_{run_time}', sess.graph)

    # Training variables
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step, should_stop, cur_best_pre_0 = 0, False, 0.0
    t0 = time()

    # Training loop
    for epoch in range(1, args.epoch + 1):
        t1 = time()
        loss, mf_loss, emb_loss, seq_loss = 0., 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1

        # Parallelized sampling and training like LightGCN
        sample_last = sample_thread()
        sample_last.start()
        sample_last.join()
        
        for idx in range(n_batch):
            train_cur = train_thread(model, sess, sample_last)
            sample_next = sample_thread()
            
            train_cur.start()
            sample_next.start()
            
            sample_next.join()
            train_cur.join()
            
            sample_last = sample_next
            
            loss += train_cur.data[1] / n_batch
            mf_loss += train_cur.data[2] / n_batch
            emb_loss += train_cur.data[3] / n_batch
            seq_loss += train_cur.data[4] / n_batch

        # TensorBoard logging
        if model.has_sequential_data:
            summary_train_loss = sess.run(model.merged_train_loss, feed_dict={
                model.train_loss: loss, model.train_mf_loss: mf_loss,
                model.train_emb_loss: emb_loss, model.train_reg_loss: 0,
                model.train_seq_loss: seq_loss
            })
        else:
            summary_train_loss = sess.run(model.merged_train_loss, feed_dict={
                model.train_loss: loss, model.train_mf_loss: mf_loss,
                model.train_emb_loss: emb_loss, model.train_reg_loss: 0
            })
        train_writer.add_summary(summary_train_loss, epoch)

        if np.isnan(loss):
            print('ERROR: loss is nan.')
            sys.exit()

        # Print progress - show seq_loss only if sequential is enabled
        if (epoch % 20) != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                if model.has_sequential_data:
                    perf_str = f'Epoch {epoch} [{time() - t1:.1f}s]: train==[{loss:.5f}={mf_loss:.5f} + {emb_loss:.5f} + {seq_loss:.5f}]'
                else:
                    perf_str = f'Epoch {epoch} [{time() - t1:.1f}s]: train==[{loss:.5f}={mf_loss:.5f} + {emb_loss:.5f}]'
                print(perf_str)
            continue

        # Evaluation every 20 epochs
        users_to_test = list(data_generator.train_items.keys())
        ret_train = test(sess, model, users_to_test, drop_flag=True, train_set_flag=1)
        
        if model.has_sequential_data:
            print(f'Epoch {epoch}: train==[{loss:.5f}={mf_loss:.5f} + {emb_loss:.5f} + {seq_loss:.5f}], '
                  f'recall=[{", ".join([f"{r:.5f}" for r in ret_train["recall"]])}], '
                  f'precision=[{", ".join([f"{r:.5f}" for r in ret_train["precision"]])}], '
                  f'ndcg=[{", ".join([f"{r:.5f}" for r in ret_train["ndcg"]])}]')
        else:
            print(f'Epoch {epoch}: train==[{loss:.5f}={mf_loss:.5f} + {emb_loss:.5f}], '
                  f'recall=[{", ".join([f"{r:.5f}" for r in ret_train["recall"]])}], '
                  f'precision=[{", ".join([f"{r:.5f}" for r in ret_train["precision"]])}], '
                  f'ndcg=[{", ".join([f"{r:.5f}" for r in ret_train["ndcg"]])}]')
        
        # Calculate test loss
        loss_test, mf_loss_test, emb_loss_test, seq_loss_test = 0., 0., 0., 0.
        sample_last = sample_thread_test()
        sample_last.start()
        sample_last.join()
        
        for idx in range(n_batch):
            test_cur = train_thread_test(model, sess, sample_last)
            sample_next = sample_thread_test()
            
            test_cur.start()
            sample_next.start()
            
            sample_next.join()
            test_cur.join()
            
            sample_last = sample_next
            
            loss_test += test_cur.data[0] / n_batch
            mf_loss_test += test_cur.data[1] / n_batch
            emb_loss_test += test_cur.data[2] / n_batch
            seq_loss_test += test_cur.data[3] / n_batch

        # Test on test set
        # t2 = time()
        # users_to_test = list(data_generator.test_set.keys())
        # ret = test(sess, model, users_to_test, drop_flag=True)
        # t3 = time()
        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        t_infer_start = time()
        ret = test(sess, model, users_to_test, drop_flag=True)
        t_infer_end = time()
        infer_time_ms = ((t_infer_end - t_infer_start) / len(users_to_test)) * 1000
        print(f'Inference time per user: {infer_time_ms:.4f} ms')
        t3 = time()
        ##############################################################################
        # ADD THIS right after t3 = time():
        if epoch % 20 == 0:
            groups = {'cold': [], 'low': [], 'medium': [], 'warm': []}
            for user in data_generator.test_set.keys():
                n = len(data_generator.train_items.get(user, []))
                if n <= 5:
                    groups['cold'].append(user)
                elif n <= 20:
                    groups['low'].append(user)
                elif n <= 50:
                    groups['medium'].append(user)
                else:
                    groups['warm'].append(user)
            
            for group_name, group_users in groups.items():
                if len(group_users) == 0:
                    continue
                ret_group = test(sess, model, group_users, drop_flag=True)
                print(f'[Sparsity] Epoch {epoch} | Group={group_name} | '
                    f'n_users={len(group_users)} | '
                    f'recall={[f"{r:.5f}" for r in ret_group["recall"]]} | '
                    f'ndcg={[f"{r:.5f}" for r in ret_group["ndcg"]]}')
        #########################################################################################

        if model.has_sequential_data:
            print(f'Epoch {epoch} [{t2 - t1:.1f}s + {t3 - t2:.1f}s]: '
                  f'test==[{loss_test:.5f}={mf_loss_test:.5f} + {emb_loss_test:.5f} + {seq_loss_test:.5f}], '
                  f'recall=[{", ".join([f"{r:.5f}" for r in ret["recall"]])}], '
                  f'precision=[{", ".join([f"{r:.5f}" for r in ret["precision"]])}], '
                  f'ndcg=[{", ".join([f"{r:.5f}" for r in ret["ndcg"]])}]')
        else:
            print(f'Epoch {epoch} [{t2 - t1:.1f}s + {t3 - t2:.1f}s]: '
                  f'test==[{loss_test:.5f}={mf_loss_test:.5f} + {emb_loss_test:.5f}], '
                  f'recall=[{", ".join([f"{r:.5f}" for r in ret["recall"]])}], '
                  f'precision=[{", ".join([f"{r:.5f}" for r in ret["precision"]])}], '
                  f'ndcg=[{", ".join([f"{r:.5f}" for r in ret["ndcg"]])}]')

        # Store results
        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        # Early stopping
        cur_best_pre_0, stopping_step, should_stop = early_stopping(
            ret['recall'][0], cur_best_pre_0, stopping_step, expected_order='acc', flag_step=10)

        if should_stop:
            break

    # Final results
    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    # visualize_tsne(
    #     sess, model, data_generator, args,
    #     epoch=args.epoch,
    #     save_dir=f'tsne_plots/{args.dataset}'
    # )

    final_perf = (f"Best Iter=[{idx}]@[{time() - t0:.1f}s]\t"
                  f"recall=[{chr(9).join([f'{r:.5f}' for r in recs[idx]])}], "
                  f"precision=[{chr(9).join([f'{r:.5f}' for r in pres[idx]])}], "
                  f"hit=[{chr(9).join([f'{r:.5f}' for r in hit[idx]])}], "
                  f"ndcg=[{chr(9).join([f'{r:.5f}' for r in ndcgs[idx]])}]")
    print(final_perf)

    # Save results
    save_path = f'{args.proj_path}output/{args.dataset}/{model.model_type}.result'
    ensureDir(save_path)
    with open(save_path, 'a') as f:
        f.write(f'embed_size={args.embed_size}, lr={args.lr:.4f}, layer_size={args.layer_size}, '
                f'node_dropout={args.node_dropout}, mess_dropout={args.mess_dropout}, '
                f'regs={args.regs}, adj_type={args.adj_type}, '
                f'max_seq_len={args.max_seq_len}, min_seq_len={args.min_seq_len}, '
                f'gru_hidden_size={args.gru_hidden_size}, seq_dropout={args.seq_dropout}, '
                f'seq_weight={getattr(args, "seq_weight", 0.0):.4f}\n\t{final_perf}\n')


if __name__ == '__main__':
    main()