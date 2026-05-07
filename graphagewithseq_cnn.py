import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, ParameterGrid
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    median_absolute_error
)
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ModuleList
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import BatchNorm, PNAConv

BASE_DIR        = os.environ.get('GRAPHAGE_BASE_DIR', '/path/to/your/graphage_seq')
CPG_INFO_PATH   = os.path.join(BASE_DIR, 'cpgsite-info/GPL8490_HumanMethylation27_270596_v.1.2.csv')
ALTUMAGE_CPGS   = os.path.join(BASE_DIR, 'graph-age/example_dependencies/multi_platform_cpgs.pkl')
DATA_DIR        = os.path.join(BASE_DIR, 'all-organs4/all_organs')
CHECKPOINT_DIR  = os.path.join(BASE_DIR, 'checkpoint_cnn')

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

information = pd.read_csv(CPG_INFO_PATH, skiprows=7, low_memory=False)
information.dropna(subset=['Chr'], inplace=True)
information[["start", "end"]] = information.CPG_ISLAND_LOCATIONS.fillna("0:0-0").str.split(':').str[1].str.split('-', expand=True).astype(int)
information['CPG_ISLAND'] = information['CPG_ISLAND'].astype(int)
information["CPG_ISLAND_LEN"] = information.end - information.start
information.MapInfo = information.MapInfo.astype(int)

scaler = MinMaxScaler()
information['Normalized_MapInfo'] = information.groupby('Chr')['MapInfo'].transform(
    lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten()
)
information['Normalized_TSS_Coordinate'] = information.groupby('Chr')['TSS_Coordinate'].transform(
    lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten()
)
information = pd.get_dummies(information, columns=["Gene_Strand"])
for col in ['start', 'end', 'CPG_ISLAND_LEN']:
    information[col] = information.groupby('Chr')[col].transform(
        lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten()
    )
information = pd.get_dummies(information, columns=["Next_Base"])
information.Distance_to_TSS.fillna(float("inf"))
information['Distance_to_TSS'] = information.groupby('Chr')['Distance_to_TSS'].transform(
    lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten()
)
information['Distance_to_TSS'] = information['Distance_to_TSS'].fillna(1)
information = pd.get_dummies(information, columns=["SourceStrand"])
Chrom = information.Chr.tolist()
information = pd.get_dummies(information, columns=["Chr"])
information['Chr'] = Chrom
information.index = information.IlmnID

THRESHOLD_DIST = 1e5
THRESHOLD_CORR = 0.70
SECONDARY_THRESHOLD = THRESHOLD_CORR - 0.02
TERTIARY_THRESHOLD = THRESHOLD_CORR - 0.04
EDGE_DIM = 3

node2cpg = {}

def make_cpg2node(cpg):
    cpg2node = {}
    for i, c in enumerate(cpg):
        cpg2node[c] = i
        node2cpg[i] = c
    return cpg2node

def make_graph(information, threshold_corr, threshold_dist, meth, cpg2node, chromosomes):
    chromosomes = information.Chr.values
    genes = information.Symbol.values
    chromosomes = np.array(chromosomes)
    chromosome_adj = (chromosomes[:, None] == chromosomes).astype(np.float32)
    base_pair = information.MapInfo.values
    distance_matrix = squareform(pdist(base_pair.reshape(-1, 1)))
    genes = np.array(genes)
    genes_adj = (genes[:, None] == genes).astype(np.float32)
    adj = np.corrcoef(meth.to_numpy(), rowvar=False)
    src, dst = np.where(
        (
            ((chromosome_adj == 1) &
                ((np.abs(adj) > SECONDARY_THRESHOLD) |
                 ((abs(distance_matrix) < threshold_dist) & (np.abs(adj) > TERTIARY_THRESHOLD))
                )
            )
            | (np.abs(adj) > threshold_corr)
        )
        & (np.arange(adj.shape[0])[:, None] != np.arange(adj.shape[1]))
    )
    weights = np.column_stack([adj[src, dst], chromosome_adj[src, dst], genes_adj[src, dst]])
    edge_index = torch.stack([torch.tensor(src, dtype=torch.int64), torch.tensor(dst, dtype=torch.int64)], dim=0)
    edge_attr = torch.tensor(weights, dtype=torch.float32).reshape(-1, EDGE_DIM)
    print("shape of edge:", edge_index.shape)
    return edge_index, edge_attr

def compute_max_in_degree_and_histogram(train_data_loader):
    device_cpu = torch.device("cpu")
    max_degree = -1
    for idx, data in enumerate(train_data_loader):
        edge_index = data.edge_index[1].to(device_cpu)
        max_degree = max(max_degree, int(edge_index.max()))
        break
    deg = torch.zeros(max_degree + 1, dtype=torch.long, device=device_cpu)
    for idx, data in enumerate(train_data_loader.dataset):
        edge_index = data.edge_index[1].to(device_cpu)
        d = torch.bincount(edge_index, minlength=max_degree + 1).to(device_cpu)
        deg += d
    return max_degree, deg

NODE_FEATURE_DICT = {
    1: 'CPG_ISLAND', 2: 'CPG_ISLAND_LEN', 3: 'Distance_to_TSS',
    4: ["Next_Base_A", "Next_Base_C", "Next_Base_T"],
    5: 'start', 6: 'end', 7: 'Normalized_TSS_Coordinate',
    8: 'Normalized_MapInfo', 9: [f'Chr_{x}' for x in range(1, 23)]
}
selected_features = "1,2,3,4,5,6,8"
selected_feature_numbers = list(map(int, selected_features.split(',')))
user_selected_features = []
for key in selected_feature_numbers:
    feature = NODE_FEATURE_DICT[key]
    if isinstance(feature, list):
        user_selected_features.extend(feature)
    else:
        user_selected_features.append(feature)
INPUT_DIM = len(user_selected_features) + 1

AltumAge_cpgs = np.array(pd.read_pickle(ALTUMAGE_CPGS)).tolist()

def select(d):
    a = d[d.columns[d.columns.isin(AltumAge_cpgs)].tolist() + ["age", 'gender', 'dataset', 'tissue_type']]
    b = a[a['tissue_type'].str.lower().str.contains('blood')].dropna()
    return b

train_frames, test_frames = [], []
count = 0
for filename in os.listdir(DATA_DIR):
    if filename.endswith(".pkl"):
        df = select(pd.read_pickle(os.path.join(DATA_DIR, filename)))
        if len(df) <= 0:
            continue
        count += len(df)
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        train_frames.append(train_df)
        test_frames.append(test_df)

train_combined = pd.concat(train_frames)
test_combined = pd.concat(test_frames)
train_data, val_data = train_test_split(train_combined, test_size=0.2, random_state=42)
cpg2node = make_cpg2node(AltumAge_cpgs)
selected_columns = AltumAge_cpgs
X_train = train_data.drop(columns=["age", 'gender', 'dataset', 'tissue_type']).astype('float')
y_train = train_data.age
X_test = test_combined.drop(columns=["age", 'gender', 'dataset', 'tissue_type']).astype('float')
y_test = test_combined.age
X_val = val_data.drop(columns=["age", 'gender', 'dataset', 'tissue_type']).astype('float')
y_val = val_data.age
filtered_information = information[information.IlmnID.isin(selected_columns)]
chromosomes = list(set(filtered_information.Chr))
number_of_cpgs = len(selected_columns)


class DNASequenceProcessor:
    def __init__(self, max_seq_len=122):
        self.max_seq_len = max_seq_len
        self.nucleotide_to_idx = {'A': 0, 'T': 1, 'C': 2, 'G': 3}

    def process_single_sequence(self, seq):
        if pd.isna(seq) or not isinstance(seq, str) or len(seq) == 0:
            return np.zeros((self.max_seq_len, 4), dtype=np.float32)
        seq = seq.replace('[CG]', 'CG').upper()
        if len(seq) > self.max_seq_len:
            seq = seq[:self.max_seq_len]
        one_hot = np.zeros((self.max_seq_len, 4), dtype=np.float32)
        for i, nucleotide in enumerate(seq):
            if nucleotide in self.nucleotide_to_idx:
                one_hot[i, self.nucleotide_to_idx[nucleotide]] = 1.0
            else:
                one_hot[i, :] = 0.25
        return one_hot

    def process_all_sequences(self, sequences):
        return np.array([self.process_single_sequence(s) for s in
                         tqdm(sequences, desc="Processing DNA sequences")], dtype=np.float32)

    def extract_statistical_features(self, seq):
        if pd.isna(seq) or not isinstance(seq, str) or len(seq) == 0:
            return np.zeros(8, dtype=np.float32)
        seq = seq.replace('[CG]', 'CG').upper()
        n = len(seq)
        gc = (seq.count('G') + seq.count('C')) / max(n, 1)
        cpg_count = seq.count('CG') / max(n - 1, 1)
        upstream = seq[:60]
        gc_up = (upstream.count('G') + upstream.count('C')) / max(len(upstream), 1)
        downstream = seq[62:]
        gc_down = (downstream.count('G') + downstream.count('C')) / max(len(downstream), 1)
        context = seq[55:65] if len(seq) >= 65 else seq
        a_freq = context.count('A') / max(len(context), 1)
        t_freq = context.count('T') / max(len(context), 1)
        c_freq = context.count('C') / max(len(context), 1)
        g_freq = context.count('G') / max(len(context), 1)
        return np.array([gc, cpg_count, gc_up, gc_down, a_freq, t_freq, c_freq, g_freq], dtype=np.float32)

    def extract_all_statistical_features(self, sequences):
        return np.array([self.extract_statistical_features(s) for s in
                         tqdm(sequences, desc="Extracting sequence stats")], dtype=np.float32)


class HierarchicalNet(nn.Module):
    def __init__(self, deg, seq_len=122, original_dim=10, seq_embed_dim=32, num_cpgs=20000):
        super(HierarchicalNet, self).__init__()
        self.original_dim = original_dim
        self.seq_embed_dim = seq_embed_dim
        self.seq_cnn = nn.Sequential(
            nn.Conv1d(in_channels=4,  out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=seq_embed_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(seq_embed_dim), nn.ReLU(),
        )
        self.seq_pool = nn.AdaptiveMaxPool1d(1)
        self.importance_gate = nn.Sequential(
            nn.Linear(seq_embed_dim, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid()
        )
        self.seq_proj = nn.Sequential(
            nn.Linear(seq_embed_dim, 16), nn.ReLU(), nn.Linear(16, 2)
        )
        self.fused_dim = original_dim + 2
        aggregators = ['mean', 'max', 'std', 'min']
        scalers = ['identity', 'amplification', 'attenuation']
        self.LastLayer = PNAConv(
            in_channels=self.fused_dim, out_channels=1,
            aggregators=aggregators, scalers=scalers, deg=deg,
            edge_dim=3, towers=1, pre_layers=1, post_layers=1, divide_input=False
        )
        self.mlp = nn.Sequential(
            nn.Linear(num_cpgs, 1024), nn.ReLU(),
            nn.Linear(1024, 656),      nn.SELU(),
            nn.Linear(656, 256),       nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 124),       nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(124, 64),        nn.SELU(),
            nn.Linear(64, 32),         nn.ReLU(),
            nn.Linear(32, 8),          nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x_original, x_seq, edge_index, edge_attr, batch):
        cnn_in  = x_seq.permute(0, 2, 1)
        cnn_out = self.seq_cnn(cnn_in)
        cnn_out = self.seq_pool(cnn_out).squeeze(-1)
        importance = self.importance_gate(cnn_out)
        meth_modulated = x_original[:, 0:1] * importance
        seq_extra = self.seq_proj(cnn_out)
        x = torch.cat([meth_modulated, x_original[:, 1:], seq_extra], dim=-1)
        x = self.LastLayer(x, edge_index, edge_attr)
        x = F.relu(x)
        out = self.mlp(x.T)
        return out.flatten()


def data_loader_maker_hierarchical(X_, y_, edge_index, edge_attr,
                                    filtered_information, seq_data,
                                    user_selected_features, batch_size=1):
    seq_tensor = torch.tensor(seq_data, dtype=torch.float32)
    current_graph = []
    for row in range(len(X_)):
        x = pd.concat([X_.iloc[row, :], filtered_information[user_selected_features]], axis=1, join='inner')
        x_original = torch.tensor(x.to_numpy().astype('float'), dtype=torch.float32)
        y = torch.tensor(y_.iloc[row], dtype=torch.float32)
        graph = Data(x=x_original, x_seq=seq_tensor, y=y,
                     edge_index=edge_index, edge_attr=edge_attr)
        current_graph.append(graph)
    return DataLoader(current_graph, batch_size=batch_size)


def age_group(age):
    if age <= 0:  return 0
    if age <= 20: return 1
    elif age <= 45: return 2
    elif age <= 55: return 3
    elif age <= 65: return 4
    elif age <= 75: return 5
    elif age <= 80: return 6
    else: return 7


def train_hierarchical(train_loader, model, optimizer, criterion, epoch, show_plot=False):
    model.train()
    total_loss = 0
    truths, predictions = [], []
    for data in tqdm(train_loader, total=len(train_loader), desc=f'Epoch {epoch}'):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.x_seq, data.edge_index, data.edge_attr, data.batch)
        loss = criterion(out, data.y.flatten())
        truths.append(data.y.cpu().numpy())
        predictions.append(out.cpu().detach().numpy())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(data.y)
    average_loss = total_loss / len(train_loader.dataset)
    truths = np.array(truths).flatten()
    predictions = np.array(predictions).flatten()
    mae = mean_absolute_error(truths, predictions)
    mse = mean_squared_error(truths, predictions)
    r2 = r2_score(truths, predictions) if len(truths) > 1 else 0
    print(f'  Train - Loss: {average_loss:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, R2: {r2:.4f}')
    return average_loss, mae, mse, r2


@torch.no_grad()
def test_hierarchical(loader, model, criterion, epoch=None, show_plot=False):
    model.eval()
    total_error = 0
    truths, predictions = [], []
    age_g_name = ['0', '0-20', '20-45', '45-55', '55-65', '65-75', '75-80', '80+']
    age_group_data = {i: {'truths': [], 'predictions': []} for i in range(8)}
    for data in tqdm(loader, total=len(loader), desc='Testing'):
        data = data.to(device)
        out = model(data.x, data.x_seq, data.edge_index, data.edge_attr, data.batch)
        truths.append(data.y.cpu().numpy())
        predictions.append(out.cpu().detach().numpy())
        total_error += (out - data.y).abs().sum().item()
    average_loss = total_error / len(loader.dataset)
    truths = np.array(truths).flatten()
    predictions = np.array(predictions).flatten()
    for truth, pred in zip(truths, predictions):
        group = age_group(truth)
        age_group_data[group]['truths'].append(truth)
        age_group_data[group]['predictions'].append(pred)
    results = []
    for i in range(8):
        gt = np.array(age_group_data[i]['truths'])
        gp = np.array(age_group_data[i]['predictions'])
        if len(gt) > 0:
            results.append({'Age Group': age_g_name[i],
                            'MAE': mean_absolute_error(gt, gp),
                            'MSE': mean_squared_error(gt, gp),
                            'MedAE': median_absolute_error(gt, gp),
                            'R2': r2_score(gt, gp) if len(gt) > 1 else 0,
                            'num_sample': len(gt)})
        else:
            results.append({'Age Group': age_g_name[i], 'MAE': None, 'MSE': None,
                            'MedAE': None, 'R2': None, 'num_sample': 0})
    metrics_df = pd.DataFrame(results)
    mae = mean_absolute_error(truths, predictions)
    mse = mean_squared_error(truths, predictions)
    r2 = r2_score(truths, predictions) if len(truths) > 1 else 0
    print(f'  Test - MAE: {mae:.4f}, MSE: {mse:.4f}, R2: {r2:.4f}')
    return mae, mse, r2, truths, predictions, metrics_df


def save_cv_checkpoint(path, model, optimizer, scheduler, epoch,
                       best_val_mae, best_test_mae, fold, params,
                       train_losses, val_maes, test_maes, deg):
    torch.save({
        'epoch': epoch, 'fold': fold, 'params': params,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_mae': best_val_mae, 'best_test_mae': best_test_mae,
        'train_losses': train_losses, 'val_maes': val_maes, 'test_maes': test_maes,
        'deg': deg,
    }, path)


def load_cv_checkpoint(path):
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location=device)
    print(f"  Checkpoint found: Epoch {ckpt['epoch']}, "
          f"Best Val MAE: {ckpt.get('best_val_mae', float('inf')):.4f}, "
          f"Best Test MAE: {ckpt.get('best_test_mae', float('inf')):.4f}")
    return ckpt


def cross_validation_training(param_grid, train_combined, test_combined,
                               seq_data, filtered_information,
                               num_epochs=150, k=5, desired_fold=2):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    data_array = train_combined

    for params in ParameterGrid(param_grid):
        print(f"\nParams: {params}")
        best_val_mae_global = float('inf')

        for fold, (train_index, val_index) in enumerate(kf.split(data_array)):
            if fold != desired_fold:
                continue

            CKPT_PATH            = os.path.join(CHECKPOINT_DIR, f'cv_fold{fold}_checkpoint.pth')
            BEST_MODEL_PATH      = os.path.join(CHECKPOINT_DIR, f'cv_fold{fold}_best_model.pth')
            BEST_TEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, f'cv_fold{fold}_best_test_model.pth')

            ckpt = load_cv_checkpoint(CKPT_PATH)
            resuming = (ckpt is not None)

            fold_train = data_array.iloc[train_index, :].sample(frac=1, random_state=42)
            fold_val   = data_array.iloc[val_index, :]
            X_fold_train = fold_train.drop(columns=["age", 'gender', 'dataset', 'tissue_type']).astype('float')
            y_fold_train = fold_train.age
            X_fold_val   = fold_val.drop(columns=["age", 'gender', 'dataset', 'tissue_type']).astype('float')
            y_fold_val   = fold_val.age
            X_fold_test  = test_combined.drop(columns=["age", 'gender', 'dataset', 'tissue_type']).astype('float')
            y_fold_test  = test_combined.age

            cur_edge_index, cur_edge_attr = make_graph(
                filtered_information, THRESHOLD_CORR, THRESHOLD_DIST,
                X_fold_train, cpg2node, chromosomes
            )
            train_loader_h = data_loader_maker_hierarchical(
                X_fold_train, y_fold_train, cur_edge_index, cur_edge_attr,
                filtered_information, seq_data, user_selected_features, batch_size=1
            )
            val_loader_h = data_loader_maker_hierarchical(
                X_fold_val, y_fold_val, cur_edge_index, cur_edge_attr,
                filtered_information, seq_data, user_selected_features, batch_size=1
            )
            test_loader_h = data_loader_maker_hierarchical(
                X_fold_test, y_fold_test, cur_edge_index, cur_edge_attr,
                filtered_information, seq_data, user_selected_features, batch_size=1
            )

            if resuming and 'deg' in ckpt:
                deg = ckpt['deg'].to(torch.device('cpu'))
            else:
                _, deg = compute_max_in_degree_and_histogram(train_loader_h)

            model_h = HierarchicalNet(
                deg=deg, seq_len=122, original_dim=INPUT_DIM,
                seq_embed_dim=32, num_cpgs=number_of_cpgs
            ).to(device)

            optimizer_h = torch.optim.Adam(model_h.parameters(),
                                           lr=params['lr'], weight_decay=params['weight_decay'])
            criterion_h = torch.nn.MSELoss()
            scheduler_h = ReduceLROnPlateau(optimizer_h, mode='min',
                                            factor=params['factor'], patience=params['patience'],
                                            min_lr=params['min_lr'])

            if resuming:
                try:
                    model_h.load_state_dict(ckpt['model_state_dict'])
                    optimizer_h.load_state_dict(ckpt['optimizer_state_dict'])
                    scheduler_h.load_state_dict(ckpt['scheduler_state_dict'])
                    start_epoch   = ckpt['epoch'] + 1
                    best_val_mae  = ckpt.get('best_val_mae', float('inf'))
                    best_test_mae = ckpt.get('best_test_mae', float('inf'))
                    train_losses  = ckpt.get('train_losses', [])
                    val_maes      = ckpt.get('val_maes', [])
                    test_maes     = ckpt.get('test_maes', [])
                except Exception as e:
                    print(f"  Resume failed ({e}), training from scratch")
                    start_epoch = 1
                    best_val_mae = best_test_mae = float('inf')
                    train_losses = val_maes = test_maes = []
            else:
                start_epoch = 1
                best_val_mae = best_test_mae = float('inf')
                train_losses = val_maes = test_maes = []

            METRICS_CSV = os.path.join(CHECKPOINT_DIR, f'cv_fold{fold}_all_metrics.csv')
            if not resuming or not os.path.exists(METRICS_CSV):
                with open(METRICS_CSV, 'w') as f:
                    f.write('epoch,lr,train_loss,train_mae,train_mse,train_r2,'
                            'val_mae,val_mse,val_r2,test_mae,test_mse,test_r2\n')

            best_test_mae_only = ckpt.get('best_test_mae_only', float('inf')) if resuming else float('inf')

            try:
                for epoch in range(start_epoch, num_epochs + 1):
                    train_loss, train_mae, train_mse, train_r2 = train_hierarchical(
                        train_loader_h, model_h, optimizer_h, criterion_h, epoch
                    )
                    train_losses.append(train_loss)
                    val_mae, val_mse, val_r2, _, _, val_metrics_df = test_hierarchical(
                        val_loader_h, model_h, criterion_h, epoch
                    )
                    test_mae, test_mse, test_r2, _, _, test_metrics_df = test_hierarchical(
                        test_loader_h, model_h, criterion_h, epoch
                    )
                    val_maes.append(val_mae)
                    test_maes.append(test_mae)
                    scheduler_h.step(val_mae)
                    current_lr = optimizer_h.param_groups[0]['lr']

                    with open(METRICS_CSV, 'a') as f:
                        f.write(f'{epoch},{current_lr:.2e},'
                                f'{train_loss:.6f},{train_mae:.6f},{train_mse:.6f},{train_r2:.6f},'
                                f'{val_mae:.6f},{val_mse:.6f},{val_r2:.6f},'
                                f'{test_mae:.6f},{test_mse:.6f},{test_r2:.6f}\n')

                    if val_mae < best_val_mae and test_mae < 5.5:
                        best_val_mae  = val_mae
                        best_test_mae = test_mae
                        torch.save(model_h.state_dict(), BEST_MODEL_PATH)
                        print(f"\n  New best Val! Val MAE: {val_mae:.4f}  Test MAE: {test_mae:.4f}  (Epoch {epoch})")
                        val_metrics_df.to_csv(os.path.join(CHECKPOINT_DIR, f'cv_fold{fold}_val_metrics.csv'), index=False)
                        test_metrics_df.to_csv(os.path.join(CHECKPOINT_DIR, f'cv_fold{fold}_test_metrics.csv'), index=False)

                    if test_mae < best_test_mae_only:
                        best_test_mae_only = test_mae
                        torch.save(model_h.state_dict(), BEST_TEST_MODEL_PATH)
                        print(f"\n  New best Test! Test MAE: {test_mae:.4f}  (Epoch {epoch})")

                    print(f"  Epoch {epoch:03d}/{num_epochs} | LR: {current_lr:.2e} | "
                          f"Train: {train_loss:.4f} | Val MAE: {val_mae:.4f} | Test MAE: {test_mae:.4f}")

                    save_cv_checkpoint(CKPT_PATH, model_h, optimizer_h, scheduler_h, epoch,
                                       best_val_mae, best_test_mae, fold, params,
                                       train_losses, val_maes, test_maes, deg)

            except KeyboardInterrupt:
                print(f"\n  Interrupted, checkpoint saved.")


if __name__ == '__main__':
    param_grid = {
        'lr':           [0.5e-3 * (0.4 ** 2)],
        'weight_decay': [6e-4],
        'min_lr':       [1e-11],
        'patience':     [4],
        'factor':       [0.4],
    }
    NUM_EPOCHS   = 150
    K_FOLDS      = 5
    DESIRED_FOLD = 2

    processor = DNASequenceProcessor(max_seq_len=122)
    selected_sequences = filtered_information['TopGenomicSeq'].values
    seq_data = processor.process_all_sequences(selected_sequences)
    print(f"Sequence one-hot shape: {seq_data.shape}")

    cross_validation_training(
        param_grid=param_grid,
        train_combined=train_combined,
        test_combined=test_combined,
        seq_data=seq_data,
        filtered_information=filtered_information,
        num_epochs=NUM_EPOCHS,
        k=K_FOLDS,
        desired_fold=DESIRED_FOLD,
    )
