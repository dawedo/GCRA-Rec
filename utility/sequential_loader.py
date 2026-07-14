import numpy as np
import random
from collections import defaultdict

class SequentialDataLoader:
    def __init__(self, train_data, max_seq_len=10, min_seq_len=3):
        self.train_data = train_data
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.user_sequences = self._build_sequences()
    
    def _build_sequences(self):
        """Build sequential data from user-item interactions"""
        user_sequences = {}
        
        for user_id, items in self.train_data.items():
            if len(items) >= self.min_seq_len:
                # Sort items by timestamp if available, otherwise use original order
                item_sequence = list(items)
                
                # Create sequences of different lengths
                sequences = []
                for i in range(self.min_seq_len, min(len(item_sequence) + 1, self.max_seq_len + 1)):
                    for start_idx in range(len(item_sequence) - i + 1):
                        seq = item_sequence[start_idx:start_idx + i]
                        sequences.append(seq)
                
                user_sequences[user_id] = sequences
        
        return user_sequences
    
    def get_batch_sequences(self, user_batch):
        """Get sequences for a batch of users"""
        batch_sequences = []
        batch_lengths = []
        
        for user_id in user_batch:
            if user_id in self.user_sequences and self.user_sequences[user_id]:
                # Randomly select one sequence for this user
                seq = random.choice(self.user_sequences[user_id])
                
                # Pad or truncate to max_seq_len
                if len(seq) < self.max_seq_len:
                    padded_seq = seq + [0] * (self.max_seq_len - len(seq))
                else:
                    padded_seq = seq[:self.max_seq_len]
                
                batch_sequences.append(padded_seq)
                batch_lengths.append(min(len(seq), self.max_seq_len))
            else:
                # User has no valid sequences, use zero padding
                batch_sequences.append([0] * self.max_seq_len)
                batch_lengths.append(1)  # At least length 1 to avoid issues
        
        return np.array(batch_sequences), np.array(batch_lengths)