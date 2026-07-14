'''
Created on Jan 14, 2026
Tensorflow Implementation of A Graph Convolutional Recurrent Attention Recommender Model 
for Dynamic Relevance Weighting of Historical Interactions. In KIS 2026.

@author: Dawed Omer Ahmed (do24csr1r07@student.nitw.ac.in)
'''
import subprocess
import datetime
import os

seq_weight_values = [0.0005,0.001,0.005,0.01,0.05,0.1]
max_seq_len_values = [20]
lr_values = [0.005]
batch_size_values = [1024]
seq_dropout_values = [0.2]
reg_values = ["1e-5"]
num_heads_values = [1]

for seq_weight in seq_weight_values:
    for max_seq_len in max_seq_len_values:
        for lr in lr_values:
            for batch_size in batch_size_values:
                for seq_dropout in seq_dropout_values:
                    for reg in reg_values:
                        for num_heads in num_heads_values:
                            ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                            log = f"Yelp2018_processeed_sequence_weight vary_maxlen{max_seq_len}_regs{reg}_lr{lr}_seq{seq_dropout}_heads{num_heads}_{ts}.txt"

                            print(f">>> Running seq_weight={seq_weight} | max_seq_len={max_seq_len} | lr={lr} | batch_size={batch_size} | seq_dropout={seq_dropout} | regs={reg} | num_heads={num_heads} | log: {log}")

                            cmd = [
                                "python", "GCRA_Rec.py",
                                "--dataset", "Yelp2018_processeed",
                                "--embed_size", "64",
                                "--layer_size", "[64,64,64]",
                                "--lr", str(lr),
                                "--regs", f"[{reg}]",
                                "--max_seq_len", str(max_seq_len),
                                "--min_seq_len", "3",
                                "--gru_hidden_size", "64",
                                "--batch_size", str(batch_size),
                                "--seq_dropout", str(seq_dropout),
                                "--seq_weight", str(seq_weight),
                                "--num_heads", str(num_heads),
                                "--epoch", "1000"
                            ]

                            with open(log, "w") as f:
                                process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
                                process.wait()