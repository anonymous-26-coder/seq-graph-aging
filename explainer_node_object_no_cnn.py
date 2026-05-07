import os
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import PNAConv
from torch_geometric.explain import Explainer, GNNExplainer

BASE_DIR        = os.environ.get('GRAPHAGE_BASE_DIR', '/path/to/your/graphage_seq')
CHECKPOINT_PATH = os.path.join(BASE_DIR, 'checkpoint_seq_no_CNN/cv_fold2_best_test_model.pth')
OUTPUT_DIR      = os.path.join(BASE_DIR, 'checkpoint_seq_no_CNN/explainer_node')
CPG_INFO_PATH   = os.path.join(BASE_DIR, 'cpgsite-info/GPL8490_HumanMethylation27_270596_v.1.2.csv')
DATA_DIR        = os.path.join(BASE_DIR, 'all-organs4/all_organs')
ALTUMAGE_CPGS   = os.path.join(BASE_DIR, 'graph-age/example_dependencies/multi_platform_cpgs.pkl')

os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 0
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

THRESHOLD_CORR            = 0.70
SECONDARY_THRESHOLD       = THRESHOLD_CORR - 0.02
TERTIARY_THRESHOLD        = THRESHOLD_CORR - 0.04
THRESHOLD_DIST            = 1e5
EDGE_DIM                  = 3
K_FOLDS                   = 5
DESIRED_FOLD              = 2
EDGE_IMPORTANCE_THRESHOLD = 0.1
MIN_SUBNETWORK_SIZE       = 10

AGE_GROUP_LABELS = ['0', '0-20', '20-45', '45-55', '55-65', '65-75', '75-80', '80+']

def age_group(age):
    if age <= 0:   return 0
    if age <= 20:  return 1
    if age <= 45:  return 2
    if age <= 55:  return 3
    if age <= 65:  return 4
    if age <= 75:  return 5
    if age <= 80:  return 6
    return 7

print('Loading CpG site information...')
information = pd.read_csv(CPG_INFO_PATH, skiprows=7, low_memory=False)
information.dropna(subset=['Chr'], inplace=True)
information[['start', 'end']] = (
    information.CPG_ISLAND_LOCATIONS
    .fillna('0:0-0').str.split(':').str[1]
    .str.split('-', expand=True).astype(int)
)
information['CPG_ISLAND']     = information['CPG_ISLAND'].astype(int)
information['CPG_ISLAND_LEN'] = information.end - information.start
information.MapInfo            = information.MapInfo.astype(int)

scaler = MinMaxScaler()
for col in ['MapInfo', 'TSS_Coordinate']:
    information[f'Normalized_{col}'] = information.groupby('Chr')[col].transform(
        lambda x: scaler.fit_transform(x.values.reshape(-1,1)).flatten()
    )
information = pd.get_dummies(information, columns=['Gene_Strand'])
for col in ['start', 'end', 'CPG_ISLAND_LEN']:
    information[col] = information.groupby('Chr')[col].transform(
        lambda x: scaler.fit_transform(x.values.reshape(-1,1)).flatten()
    )
information = pd.get_dummies(information, columns=['Next_Base'])
information['Distance_to_TSS'] = information.groupby('Chr')['Distance_to_TSS'].transform(
    lambda x: scaler.fit_transform(x.values.reshape(-1,1)).flatten()
)
information['Distance_to_TSS'] = information['Distance_to_TSS'].fillna(1)
information = pd.get_dummies(information, columns=['SourceStrand'])
Chrom = information.Chr.tolist()
information = pd.get_dummies(information, columns=['Chr'])
information['Chr'] = Chrom
information.index  = information.IlmnID

NODE_FEATURE_DICT = {
    1: 'CPG_ISLAND', 2: 'CPG_ISLAND_LEN', 3: 'Distance_to_TSS',
    4: ['Next_Base_A', 'Next_Base_C', 'Next_Base_T'],
    5: 'start', 6: 'end', 7: 'Normalized_TSS_Coordinate',
    8: 'Normalized_MapInfo', 9: [f'Chr_{x}' for x in range(1,23)]
}
user_selected_features = []
for key in [1,2,3,4,5,6,8]:
    feat = NODE_FEATURE_DICT[key]
    if isinstance(feat, list):
        user_selected_features.extend(feat)
    else:
        user_selected_features.append(feat)
INPUT_DIM = len(user_selected_features) + 1
print(f'INPUT_DIM: {INPUT_DIM}')

AltumAge_cpgs  = np.array(pd.read_pickle(ALTUMAGE_CPGS)).tolist()
number_of_cpgs = len(AltumAge_cpgs)

node2cpg = {}
def make_cpg2node(cpg):
    cpg2node = {}
    for i, c in enumerate(cpg):
        cpg2node[c] = i
        node2cpg[i] = c
    return cpg2node

cpg2node             = make_cpg2node(AltumAge_cpgs)
filtered_information = information[information.IlmnID.isin(AltumAge_cpgs)].copy()


class DNASequenceProcessor:
    def __init__(self, max_seq_len=122):
        self.max_seq_len = max_seq_len

    def extract_statistical_features(self, seq):
        if pd.isna(seq) or not isinstance(seq, str) or len(seq) == 0:
            return np.zeros(8, dtype=np.float32)
        seq      = seq.replace('[CG]', 'CG').upper()
        n        = len(seq)
        gc       = (seq.count('G') + seq.count('C')) / max(n, 1)
        cpg_dens = seq.count('CG') / max(n-1, 1)
        upstream = seq[:60]
        gc_up    = (upstream.count('G') + upstream.count('C')) / max(len(upstream), 1)
        downstream = seq[62:]
        gc_down  = (downstream.count('G') + downstream.count('C')) / max(len(downstream), 1)
        context  = seq[55:65] if len(seq) >= 65 else seq
        a_freq   = context.count('A') / max(len(context), 1)
        t_freq   = context.count('T') / max(len(context), 1)
        c_freq   = context.count('C') / max(len(context), 1)
        g_freq   = context.count('G') / max(len(context), 1)
        return np.array([gc, cpg_dens, gc_up, gc_down, a_freq, t_freq, c_freq, g_freq], dtype=np.float32)

    def extract_all_statistical_features(self, sequences):
        return np.array([self.extract_statistical_features(s) for s in
                         tqdm(sequences, desc='Extracting seq features')], dtype=np.float32)


def make_graph(information, threshold_corr, threshold_dist, meth):
    chromosomes_arr = np.array(information.Chr.values)
    genes           = np.array(information.Symbol.values)
    chromosome_adj  = (chromosomes_arr[:, None] == chromosomes_arr).astype(np.float32)
    base_pair       = information.MapInfo.values
    distance_matrix = squareform(pdist(base_pair.reshape(-1,1)))
    genes_adj       = (genes[:, None] == genes).astype(np.float32)
    print('  Computing co-methylation...')
    adj             = np.corrcoef(meth.to_numpy(), rowvar=False)
    src, dst        = np.where(
        (((chromosome_adj == 1) &
          ((np.abs(adj) > SECONDARY_THRESHOLD) |
           ((np.abs(distance_matrix) < threshold_dist) & (np.abs(adj) > TERTIARY_THRESHOLD))))
         | (np.abs(adj) > threshold_corr))
        & (np.arange(adj.shape[0])[:, None] != np.arange(adj.shape[1]))
    )
    weights    = np.column_stack([adj[src,dst], chromosome_adj[src,dst], genes_adj[src,dst]])
    edge_index = torch.stack([torch.tensor(src, dtype=torch.int64),
                               torch.tensor(dst, dtype=torch.int64)], dim=0)
    edge_attr  = torch.tensor(weights, dtype=torch.float32).reshape(-1, EDGE_DIM)
    print(f'  Edges: {edge_index.shape[1]}')
    return edge_index, edge_attr, adj


class HierarchicalNet(nn.Module):
    def __init__(self, deg, original_dim=10, num_cpgs=20318):
        super().__init__()
        self.original_dim    = original_dim
        self.importance_gate = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid()
        )
        self.seq_proj = nn.Sequential(
            nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2)
        )
        self.fused_dim = original_dim + 2
        aggregators = ['mean', 'max', 'std', 'min']
        scalers     = ['identity', 'amplification', 'attenuation']
        self.LastLayer = PNAConv(
            in_channels=self.fused_dim, out_channels=1,
            aggregators=aggregators, scalers=scalers, deg=deg,
            edge_dim=3, towers=1, pre_layers=1, post_layers=1, divide_input=False
        )
        self.mlp = nn.Sequential(
            nn.Linear(num_cpgs, 1024), nn.ReLU(),
            nn.Linear(1024, 656),      nn.SELU(),
            nn.Linear(656, 256),       nn.ReLU(),  nn.Dropout(0.2),
            nn.Linear(256, 124),       nn.ReLU(),  nn.Dropout(0.2),
            nn.Linear(124, 64),        nn.SELU(),
            nn.Linear(64, 32),         nn.ReLU(),
            nn.Linear(32, 8),          nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x_original, x_seq, edge_index, edge_attr, batch):
        importance     = self.importance_gate(x_seq)
        meth_modulated = x_original[:, 0:1] * importance
        seq_extra      = self.seq_proj(x_seq)
        x = torch.cat([meth_modulated, x_original[:, 1:], seq_extra], dim=-1)
        x = self.LastLayer(x, edge_index, edge_attr)
        x = F.relu(x)
        return self.mlp(x.T).flatten()


class WrapperOriginal(nn.Module):
    def __init__(self, model, x_seq_fixed):
        super().__init__()
        self.model = model
        self.register_buffer('x_seq_fixed', x_seq_fixed)

    def forward(self, x, edge_index, edge_attr, batch=None):
        return self.model(x, self.x_seq_fixed, edge_index, edge_attr, batch)


def compute_deg(loader):
    max_degree = -1
    for data in loader:
        max_degree = max(max_degree, int(data.edge_index[1].max()))
        break
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in loader.dataset:
        d    = torch.bincount(data.edge_index[1], minlength=max_degree+1)
        deg += d
    return deg


print('\nLoading methylation data...')

def select(d):
    a = d[d.columns[d.columns.isin(AltumAge_cpgs)].tolist() +
          ['age', 'gender', 'dataset', 'tissue_type']]
    return a[a['tissue_type'].str.lower().str.contains('blood')].dropna()

train_frames, test_frames = [], []
for filename in os.listdir(DATA_DIR):
    if not filename.endswith('.pkl'):
        continue
    df = select(pd.read_pickle(os.path.join(DATA_DIR, filename)))
    if len(df) == 0:
        continue
    tr, te = train_test_split(df, test_size=0.2, random_state=42)
    train_frames.append(tr)
    test_frames.append(te)

train_combined = pd.concat(train_frames)
test_combined  = pd.concat(test_frames)

kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kf.split(train_combined)):
    if fold != DESIRED_FOLD:
        continue
    fold_train = train_combined.iloc[train_idx, :].sample(frac=1, random_state=42)
    break

X_fold_train = fold_train.drop(columns=['age','gender','dataset','tissue_type']).astype('float')
X_fold_test  = test_combined.drop(columns=['age','gender','dataset','tissue_type']).astype('float')
y_fold_test  = test_combined.age.reset_index(drop=True)
X_fold_test  = X_fold_test.reset_index(drop=True)
meth_test_values = X_fold_test.copy()
print(f'Train: {len(X_fold_train)}, Test: {len(X_fold_test)}')

print('\nBuilding co-methylation graph...')
edge_index, edge_attr, adj_matrix = make_graph(
    filtered_information, THRESHOLD_CORR, THRESHOLD_DIST, X_fold_train
)
np.save(os.path.join(OUTPUT_DIR, 'edge_index.npy'), edge_index.numpy())
np.save(os.path.join(OUTPUT_DIR, 'edge_attr.npy'),  edge_attr.numpy())

print('\nExtracting sequence features...')
processor         = DNASequenceProcessor()
sequences         = filtered_information['TopGenomicSeq'].values
seq_data          = processor.extract_all_statistical_features(sequences)
seq_tensor_global = torch.tensor(seq_data, dtype=torch.float32)
print(f'seq_data shape: {seq_data.shape}')


def make_test_loader(X_, y_, seq_data):
    seq_tensor = torch.tensor(seq_data, dtype=torch.float32)
    graphs = []
    for row in range(len(X_)):
        x = pd.concat([X_.iloc[row, :], filtered_information[user_selected_features]], axis=1, join='inner')
        x_original = torch.tensor(x.to_numpy().astype('float'), dtype=torch.float32)
        y          = torch.tensor(y_.iloc[row], dtype=torch.float32)
        graphs.append(Data(x=x_original, x_seq=seq_tensor, y=y,
                           edge_index=edge_index, edge_attr=edge_attr))
    return DataLoader(graphs, batch_size=1)

print('\nBuilding test DataLoader...')
test_loader = make_test_loader(X_fold_test, y_fold_test, seq_data)

print('\nLoading model...')
deg  = compute_deg(test_loader)
ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
    state_dict = ckpt['model_state_dict']
    saved_deg  = ckpt.get('deg', deg)
else:
    state_dict = ckpt
    saved_deg  = deg

model = HierarchicalNet(deg=saved_deg.to('cpu'), original_dim=INPUT_DIM, num_cpgs=number_of_cpgs).to(device)
model.load_state_dict(state_dict)
model.eval()

wrapper = WrapperOriginal(model, seq_tensor_global.to(device)).to(device)
wrapper.eval()

explainer3 = Explainer(
    model=wrapper,
    algorithm=GNNExplainer(epochs=120),
    explanation_type='phenomenon',
    node_mask_type='object',
    edge_mask_type='object',
    model_config=dict(mode='regression', task_level='graph', return_type='raw'),
)

print(f'\nRunning node explainer on {len(test_loader)} test samples...')

n_age_groups        = 8
age_group_sum_node  = np.zeros((n_age_groups, number_of_cpgs))
age_group_sum_edge  = np.zeros((n_age_groups, edge_index.shape[1]))
age_group_count     = np.zeros(n_age_groups)
all_node_importance = []
all_edge_importance = []
all_ages            = []

SAVE_EVERY = 10

for cnt, data in enumerate(tqdm(test_loader, desc='Node explainer')):
    data = data.to(device)
    age  = data.y.cpu().item()
    ag   = age_group(age)
    explanation = explainer3(
        x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
        batch=data.batch_size if hasattr(data, 'batch_size') else None,
        target=data.y
    )
    node_imp = explanation.node_mask.cpu().detach().flatten().numpy()
    edge_imp = explanation.edge_mask.cpu().detach().flatten().numpy()
    age_group_sum_node[ag] += node_imp
    age_group_sum_edge[ag] += edge_imp
    age_group_count[ag]    += 1
    all_node_importance.append(node_imp)
    all_edge_importance.append(edge_imp)
    all_ages.append(age)
    if (cnt + 1) % SAVE_EVERY == 0 or cnt == len(test_loader) - 1:
        np.save(os.path.join(OUTPUT_DIR, 'age_group_sum_node.npy'),  age_group_sum_node)
        np.save(os.path.join(OUTPUT_DIR, 'age_group_sum_edge.npy'),  age_group_sum_edge)
        np.save(os.path.join(OUTPUT_DIR, 'age_group_count.npy'),     age_group_count)
        np.save(os.path.join(OUTPUT_DIR, 'node_importance.npy'),     np.array(all_node_importance))
        np.save(os.path.join(OUTPUT_DIR, 'edge_importance.npy'),     np.array(all_edge_importance))
        np.save(os.path.join(OUTPUT_DIR, 'all_ages.npy'),            np.array(all_ages))

safe_count          = np.maximum(age_group_count[:, None], 1)
age_group_mean_node = age_group_sum_node / safe_count
age_group_mean_edge = age_group_sum_edge / safe_count
mean_node_overall   = np.array(all_node_importance).mean(axis=0)

np.save(os.path.join(OUTPUT_DIR, 'age_group_mean_node.npy'), age_group_mean_node)
np.save(os.path.join(OUTPUT_DIR, 'age_group_mean_edge.npy'), age_group_mean_edge)
np.save(os.path.join(OUTPUT_DIR, 'mean_node_overall.npy'),   mean_node_overall)

print('\nComputing temporal slopes...')
ages_arr = np.array(all_ages)
node_arr = np.array(all_node_importance)
slopes   = np.zeros(number_of_cpgs)
for i in range(number_of_cpgs):
    slope, _, _, _, _ = stats.linregress(ages_arr, node_arr[:, i])
    slopes[i] = slope
np.save(os.path.join(OUTPUT_DIR, 'node_importance_slopes.npy'), slopes)

top10_up_idx   = np.argsort(slopes)[::-1][:10]
top10_down_idx = np.argsort(slopes)[:10]

def node_to_gene(idx):
    cpg_name = node2cpg.get(idx, f'node_{idx}')
    if cpg_name in filtered_information.index:
        gene = filtered_information.loc[cpg_name, 'Symbol'] if 'Symbol' in filtered_information.columns else 'N/A'
    else:
        gene = 'N/A'
    return cpg_name, gene

up_records   = [(node_to_gene(i)[0], node_to_gene(i)[1], slopes[i], mean_node_overall[i])
                for i in top10_up_idx]
down_records = [(node_to_gene(i)[0], node_to_gene(i)[1], slopes[i], mean_node_overall[i])
                for i in top10_down_idx]

df_up   = pd.DataFrame(up_records,   columns=['CpG_site', 'Gene', 'Slope', 'Mean_Importance'])
df_down = pd.DataFrame(down_records, columns=['CpG_site', 'Gene', 'Slope', 'Mean_Importance'])
df_up.to_csv(os.path.join(OUTPUT_DIR,   'top10_upward_cpg.csv'),   index=False)
df_down.to_csv(os.path.join(OUTPUT_DIR, 'top10_downward_cpg.csv'), index=False)
print('Top 10 upward-trending CpG sites:')
print(df_up.to_string(index=False))
print('\nTop 10 downward-trending CpG sites:')
print(df_down.to_string(index=False))

print('\nClassifying hypo/hypermethylation per CpG site...')
meth_slopes = np.zeros(number_of_cpgs)
for i in range(number_of_cpgs):
    slope, _, _, _, _ = stats.linregress(ages_arr, meth_test_values.iloc[:, i].values)
    meth_slopes[i] = slope
is_hyper = meth_slopes > 0
np.save(os.path.join(OUTPUT_DIR, 'meth_slopes.npy'), meth_slopes)
np.save(os.path.join(OUTPUT_DIR, 'is_hyper.npy'),    is_hyper.astype(int))

print('\nGenerating plots...')
sorted_by_age = np.argsort(ages_arr)
ages_sorted   = ages_arr[sorted_by_age]
node_sorted   = node_arr[sorted_by_age]

fig, ax = plt.subplots(figsize=(11, 5))
colors_up = plt.cm.tab10(np.linspace(0, 1, 10))
for rank, idx in enumerate(top10_up_idx):
    cpg_name, gene = node_to_gene(idx)
    ax.scatter(ages_sorted, node_sorted[:, idx], s=5, alpha=0.4,
               color=colors_up[rank], label=f'{cpg_name} ({gene})')
ax.set_xlabel('Chronological Age (years)', fontsize=12)
ax.set_ylabel('Node Importance Score', fontsize=12)
ax.set_title('Top 10 Upward-Trending CpG Sites (Importance Increases with Age)', fontsize=12)
ax.legend(fontsize=6.5, ncol=2, loc='upper left', markerscale=2, framealpha=0.7)
ax.grid(True, alpha=0.2); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'top10_upward_trend.png'), dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(11, 5))
colors_down = plt.cm.Set2(np.linspace(0, 1, 10))
for rank, idx in enumerate(top10_down_idx):
    cpg_name, gene = node_to_gene(idx)
    ax.scatter(ages_sorted, node_sorted[:, idx], s=5, alpha=0.4,
               color=colors_down[rank], label=f'{cpg_name} ({gene})')
ax.set_xlabel('Chronological Age (years)', fontsize=12)
ax.set_ylabel('Node Importance Score', fontsize=12)
ax.set_title('Top 10 Downward-Trending CpG Sites (Importance Decreases with Age)', fontsize=12)
ax.legend(fontsize=6.5, ncol=2, loc='upper right', markerscale=2, framealpha=0.7)
ax.grid(True, alpha=0.2); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'top10_downward_trend.png'), dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(mean_node_overall, bins=100, color='#5B8DB8', edgecolor='none', alpha=0.8)
ax.set_xlabel('Mean Node Importance Score', fontsize=12)
ax.set_ylabel('Number of CpG Sites', fontsize=12)
ax.set_title('Distribution of Mean Node Importance across 20,318 CpG Sites', fontsize=12)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'node_importance_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()

top50_idx   = np.argsort(mean_node_overall)[::-1][:50]
top50_names = [f"{node_to_gene(i)[0]}\n({node_to_gene(i)[1]})" for i in top50_idx]
top50_vals  = mean_node_overall[top50_idx]
fig, ax = plt.subplots(figsize=(18, 6))
colors_bar = ['#D95F5F' if is_hyper[i] else '#5B8DB8' for i in top50_idx]
ax.bar(range(50), top50_vals, color=colors_bar, edgecolor='none')
ax.set_xticks(range(50)); ax.set_xticklabels(top50_names, rotation=90, fontsize=6)
ax.set_ylabel('Mean Importance Score', fontsize=11)
ax.set_title('Top 50 CpG Sites by Mean Importance\n(Red = Hypermethylating, Blue = Hypomethylating)', fontsize=11)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'top50_node_importance.png'), dpi=300, bbox_inches='tight')
plt.close()

try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    print('graphviz not found. Skipping MRN visualisation. Install with: pip install graphviz')

if GRAPHVIZ_AVAILABLE:
    edge_index_np = edge_index.numpy()
    edge_attr_np  = edge_attr.numpy()
    VIS_AGE_GROUPS = [2, 3, 4]

    for ag in VIS_AGE_GROUPS:
        ag_label = AGE_GROUP_LABELS[ag]
        mean_node_ag = age_group_mean_node[ag]
        mean_edge_ag = age_group_mean_edge[ag]
        nonzero_node_mask = mean_node_ag > 0
        src_arr = edge_index_np[0]; dst_arr = edge_index_np[1]
        edge_mask = (
            nonzero_node_mask[src_arr] & nonzero_node_mask[dst_arr] &
            (mean_edge_ag > EDGE_IMPORTANCE_THRESHOLD)
        )
        kept_src       = src_arr[edge_mask]; kept_dst = dst_arr[edge_mask]
        kept_edge_imp  = mean_edge_ag[edge_mask]
        kept_edge_attr = edge_attr_np[edge_mask]

        from collections import defaultdict, deque
        adj_dict = defaultdict(set)
        for s, d in zip(kept_src, kept_dst):
            adj_dict[int(s)].add(int(d)); adj_dict[int(d)].add(int(s))

        visited = set(); components = []
        all_involved_nodes = set(kept_src.tolist() + kept_dst.tolist())
        for start_node in all_involved_nodes:
            if start_node in visited:
                continue
            component = []; queue = deque([start_node])
            while queue:
                node = queue.popleft()
                if node in visited: continue
                visited.add(node); component.append(node)
                for nb in adj_dict[node]:
                    if nb not in visited: queue.append(nb)
            components.append(component)

        large_components = [c for c in components if len(c) >= MIN_SUBNETWORK_SIZE]
        mrn_dir = os.path.join(OUTPUT_DIR, f'mrn_age_{ag_label.replace("-", "_")}')
        os.makedirs(mrn_dir, exist_ok=True)

        for sub_idx, component in enumerate(large_components):
            component_set = set(component)
            comp_edges = [(int(s), int(d), float(ei), e_a)
                          for s, d, ei, e_a in zip(kept_src, kept_dst, kept_edge_imp, kept_edge_attr)
                          if int(s) in component_set and int(d) in component_set]
            seen_edges = set(); comp_edges_dedup = []
            for s, d, ei, e_a in comp_edges:
                pair = (min(s,d), max(s,d))
                if pair not in seen_edges:
                    seen_edges.add(pair); comp_edges_dedup.append((s, d, ei, e_a))

            dot = graphviz.Graph(
                name=f'MRN_age{ag_label}_sub{sub_idx+1}', engine='neato',
                graph_attr={'overlap': 'false', 'splines': 'true', 'sep': '+5', 'fontsize': '10'}
            )
            for node_id in component:
                cpg_name, gene = node_to_gene(node_id)
                node_imp_val   = mean_node_ag[node_id]
                hyper_val      = meth_slopes[node_id]
                slope_val      = slopes[node_id]
                fill_color     = '#D95F5F' if is_hyper[node_id] else '#5B8DB8'
                if cpg_name in filtered_information.index:
                    dist_tss = filtered_information.loc[cpg_name, 'Distance_to_TSS'] \
                               if 'Distance_to_TSS' in filtered_information.columns else 0.0
                    chrom    = filtered_information.loc[cpg_name, 'Chr'] \
                               if 'Chr' in filtered_information.columns else 'N/A'
                else:
                    dist_tss = 0.0; chrom = 'N/A'
                label = (f'{cpg_name}\n{gene}\ndist_tss:{dist_tss:.0f}\n'
                         f'imprtnce:{node_imp_val:.2f}\n'
                         f'{"hyper" if is_hyper[node_id] else "hypo"}:{abs(hyper_val):.2f}\n'
                         f'Chrom:{chrom}')
                if abs(slope_val) < 1e-6:
                    pen_color = 'gray'; pen_width = '1'
                elif slope_val > 0:
                    pen_color = 'green'; pen_width = '3'
                else:
                    pen_color = 'gold'; pen_width = '3'
                node_size = max(0.4, node_imp_val * 3.0)
                dot.node(str(node_id), label=label, shape='circle', style='filled',
                         fillcolor=fill_color, color=pen_color, penwidth=pen_width,
                         fontsize='7', fontcolor='white', width=str(node_size), fixedsize='false')
            for s, d, ei, e_a in comp_edges_dedup:
                cometh_val = float(e_a[0])
                dot.edge(str(s), str(d),
                         label=f'cometh:{cometh_val:.2f}\nimprtnce:{ei:.2f}',
                         fontsize='6', penwidth=str(max(0.5, ei * 5.0)), color='#555555')
            out_path = os.path.join(mrn_dir, f'mrn_sub{sub_idx+1}_n{len(component)}')
            try:
                dot.render(out_path, format='png', cleanup=True)
            except Exception as e:
                dot.save(out_path + '.gv')
                print(f'  Render failed ({e}). Saved .gv: {out_path}.gv')

top100_idx = np.argsort(mean_node_overall)[::-1][:100]
summary_records = []
for rank, idx in enumerate(top100_idx):
    cpg_name, gene = node_to_gene(idx)
    summary_records.append({
        'Rank': rank + 1, 'CpG_site': cpg_name, 'Gene': gene,
        'Mean_Importance': mean_node_overall[idx], 'Slope': slopes[idx],
        'Trend': 'up' if slopes[idx] > 0 else 'down',
        'Methylation': 'hyper' if is_hyper[idx] else 'hypo',
        'Meth_Slope': meth_slopes[idx],
        'Age_Group_0_Imp':  age_group_mean_node[0, idx],
        'Age_Group_20_45':  age_group_mean_node[2, idx],
        'Age_Group_65_75':  age_group_mean_node[5, idx],
        'Age_Group_80plus': age_group_mean_node[7, idx],
    })
pd.DataFrame(summary_records).to_csv(os.path.join(OUTPUT_DIR, 'top100_node_summary.csv'), index=False)

print(f'\nAll done. Results saved to: {OUTPUT_DIR}')
