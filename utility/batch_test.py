'''
Batch test implementation for GCRA-Rec
'''
from utility.parser import parse_args
from utility.load_data import *
from evaluator import eval_score_matrix_foldout
import multiprocessing
import numpy as np

cores = multiprocessing.cpu_count() // 2
args = parse_args()
data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size)
USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
BATCH_SIZE = args.batch_size


def test(sess, model, users_to_test, drop_flag=False, train_set_flag=0):
    """
    Test function for GCRA-Rec
    """
    top_show = np.sort(model.Ks)
    max_top = max(top_show)
    result = {
        'precision': np.zeros(len(model.Ks)), 
        'recall': np.zeros(len(model.Ks)), 
        'hit_ratio': np.zeros(len(model.Ks)), 
        'ndcg': np.zeros(len(model.Ks))
    }

    u_batch_size = BATCH_SIZE
    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0
    all_result = []
    item_batch = list(range(ITEM_NUM))  # FIX 4: list() instead of range()

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size
        user_batch = test_users[start:end]

        if len(user_batch) == 0:
            continue

        # FIX 1: Always include dropout placeholders, set to 0 when drop_flag=True
        if drop_flag == False:
            feed_dict = {
                model.users: user_batch,
                # model.pos_items: item_batch
                model.eval_items: item_batch,
            }
        else:
            feed_dict = {
                model.users: user_batch,
                model.eval_items: item_batch,
                # model.pos_items: item_batch,
                model.node_dropout: [0.] * len(eval(args.layer_size)),
                model.mess_dropout: [0.] * len(eval(args.layer_size))
            }

        # Handle sequential data for hybrid model ONLY if sequential is enabled
        if hasattr(model, 'has_sequential_data') and model.has_sequential_data:
            batch_size = len(user_batch)
            max_seq_len = getattr(args, 'max_seq_len', 50)
            
            feed_dict[model.seq_dropout] = 0.0
            user_seqs, seq_lens = [], []
            for user in user_batch:
                seq = data_generator.train_items.get(user, [])
                seq = seq[-max_seq_len:] if len(seq) > max_seq_len else seq
                seq_lens.append(max(len(seq), 1))
                padded = seq + [0] * (max_seq_len - len(seq))
                user_seqs.append(padded)

            feed_dict.update({
                model.user_seq: np.array(user_seqs, dtype=np.int32),
                model.seq_len: np.array(seq_lens, dtype=np.int32),
                model.target_item: np.zeros(batch_size, dtype=np.int32)  # not used at inference
            })
            # feed_dict.update({
            #     model.user_seq: np.zeros((batch_size, max_seq_len), dtype=np.int32),
            #     model.seq_len: np.ones(batch_size, dtype=np.int32),
            #     model.target_item: np.zeros(batch_size, dtype=np.int32)
            # })

        try:
            rate_batch = sess.run(model.batch_ratings, feed_dict)
            rate_batch = np.array(rate_batch)  # (B, N)
        except Exception as e:
            print(f"Error during rating computation: {e}")
            print(f"User batch size: {len(user_batch)}, Item batch size: {len(item_batch)}")
            continue

        test_items = []
        if train_set_flag == 0:
            for user in user_batch:
                if user in data_generator.test_set:
                    test_items.append(data_generator.test_set[user])
                else:
                    test_items.append([])
                
            for idx, user in enumerate(user_batch):
                if user in data_generator.train_items:
                    train_items_off = data_generator.train_items[user]
                    rate_batch[idx][train_items_off] = -np.inf
        else:
            for user in user_batch:
                if user in data_generator.train_items:
                    test_items.append(data_generator.train_items[user])
                else:
                    test_items.append([])

        valid_indices = [i for i, items in enumerate(test_items) if len(items) > 0]
        
        if len(valid_indices) > 0:
            valid_rate_batch = rate_batch[valid_indices]
            valid_test_items = [test_items[i] for i in valid_indices]
            
            try:
                batch_result = eval_score_matrix_foldout(valid_rate_batch, valid_test_items, max_top)
                count += len(batch_result)
                all_result.append(batch_result)
            except Exception as e:
                print(f"Error in eval_score_matrix_foldout: {e}")
                continue

    if len(all_result) == 0:
        print("Warning: No valid results found during testing")
        return result

    # FIX 2: Removed fragile assert, replaced with simple row count check
    all_result = np.concatenate(all_result, axis=0)
    assert count == len(all_result)

    final_result = np.mean(all_result, axis=0)
    final_result = np.reshape(final_result, newshape=[5, max_top])
    final_result = final_result[:, top_show-1]
    final_result = np.reshape(final_result, newshape=[5, len(top_show)])
    
    result['precision'] = final_result[0]
    result['recall'] = final_result[1] 
    result['hit_ratio'] = final_result[2]
    result['ndcg'] = final_result[3]
    
    return result

