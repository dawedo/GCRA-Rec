import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import os

def visualize_tsne(sess, model, data_generator, args, epoch, save_dir='tsne_plots'):
    """
    t-SNE visualization of learned user and item embeddings.
    Plots are saved to save_dir.
    """
    os.makedirs(save_dir, exist_ok=True)

    # ── 1. Extract embeddings from the TF session ──────────────────────────
    user_emb, item_emb = sess.run([model.ua_embeddings, model.ia_embeddings])
    # ua_embeddings / ia_embeddings are the final GCN embeddings (all users/items)

    # ── 2. Sample for speed (t-SNE is O(n²)) ──────────────────────────────
    max_users = 2000
    max_items = 2000

    n_users = user_emb.shape[0]
    n_items = item_emb.shape[0]

    user_idx = np.random.choice(n_users, size=min(max_users, n_users), replace=False)
    item_idx = np.random.choice(n_items, size=min(max_items, n_items), replace=False)

    user_emb_sample = user_emb[user_idx]
    item_emb_sample = item_emb[item_idx]

    # ── 3. Build sparsity labels for users (cold/low/medium/warm) ──────────
    def get_sparsity_label(user_id):
        n = len(data_generator.train_items.get(user_id, []))
        if n <= 5:   return 'cold'
        if n <= 20:  return 'low'
        if n <= 50:  return 'medium'
        return 'warm'

    user_labels = [get_sparsity_label(u) for u in user_idx]

    # Item labels: use interaction frequency buckets
    item_freq = {
        i: sum(1 for items in data_generator.train_items.values() if i in items)
        for i in item_idx
    }
    def get_item_label(i):
        f = item_freq.get(i, 0)
        if f <= 5:   return 'rare'
        if f <= 20:  return 'occasional'
        if f <= 50:  return 'frequent'
        return 'popular'

    item_labels = [get_item_label(i) for i in item_idx]

    # ── 4. Run t-SNE ───────────────────────────────────────────────────────
    print(f"Running t-SNE on {len(user_idx)} users and {len(item_idx)} items ...")
    tsne = TSNE(
        n_components=2,
        perplexity=40,
        n_iter=1000,
        random_state=42,
        init='pca',        # PCA init is more stable than random
        learning_rate='auto'
    )

    # Fit users and items separately for cleaner plots
    user_2d = tsne.fit_transform(user_emb_sample)
    item_2d = tsne.fit_transform(item_emb_sample)

    # ── 5. Plot: User embeddings coloured by sparsity group ───────────────
    sparsity_colors = {
        'cold':   '#E24B4A',   # red
        'low':    '#EF9F27',   # amber
        'medium': '#1D9E75',   # teal
        'warm':   '#378ADD',   # blue
    }
    sparsity_order = ['cold', 'low', 'medium', 'warm']

    fig, ax = plt.subplots(figsize=(8, 7))
    for group in sparsity_order:
        mask = np.array(user_labels) == group
        if mask.sum() == 0:
            continue
        ax.scatter(
            user_2d[mask, 0], user_2d[mask, 1],
            c=sparsity_colors[group],
            label=group, s=12, alpha=0.65, linewidths=0
        )
    ax.set_title(f't-SNE: User embeddings by sparsity group (epoch {epoch})', fontsize=12)
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')
    ax.legend(title='Interaction count', markerscale=2, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    path = os.path.join(save_dir, f'tsne_users_epoch{epoch}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved user t-SNE → {path}")

    # ── 6. Plot: Item embeddings coloured by popularity ───────────────────
    item_colors = {
        'rare':       '#E24B4A',
        'occasional': '#EF9F27',
        'frequent':   '#1D9E75',
        'popular':    '#378ADD',
    }
    item_order = ['rare', 'occasional', 'frequent', 'popular']

    fig, ax = plt.subplots(figsize=(8, 7))
    for group in item_order:
        mask = np.array(item_labels) == group
        if mask.sum() == 0:
            continue
        ax.scatter(
            item_2d[mask, 0], item_2d[mask, 1],
            c=item_colors[group],
            label=group, s=12, alpha=0.65, linewidths=0
        )
    ax.set_title(f't-SNE: Item embeddings by popularity (epoch {epoch})', fontsize=12)
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')
    ax.legend(title='Interaction frequency', markerscale=2, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    path = os.path.join(save_dir, f'tsne_items_epoch{epoch}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved item t-SNE → {path}")

    # ── 7. Plot: Joint user+item space ────────────────────────────────────
    # Stack both, run one joint t-SNE, then split by type
    joint_emb = np.vstack([user_emb_sample, item_emb_sample])
    joint_2d  = TSNE(
        n_components=2, perplexity=40, n_iter=1000,
        random_state=42, init='pca', learning_rate='auto'
    ).fit_transform(joint_emb)

    u2d = joint_2d[:len(user_idx)]
    i2d = joint_2d[len(user_idx):]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(u2d[:, 0], u2d[:, 1], c='#378ADD', s=10,
               alpha=0.5, linewidths=0, label='Users')
    ax.scatter(i2d[:, 0], i2d[:, 1], c='#E24B4A', s=10,
               alpha=0.5, linewidths=0, label='Items')
    ax.set_title(f't-SNE: Joint user–item embedding space (epoch {epoch})', fontsize=12)
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')
    ax.legend(markerscale=2, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    path = os.path.join(save_dir, f'tsne_joint_epoch{epoch}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved joint t-SNE → {path}")