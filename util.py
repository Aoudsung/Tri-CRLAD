import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def gaussian_kernel(a, b):
    """
    Compute Gaussian kernel between samples a and b.
    """
    squared_distance = ((a - b) ** 2).sum(-1)
    return torch.exp(-squared_distance / 2)

def mmd(x,y):
    xx = gaussian_kernel(x,x)
    yy = gaussian_kernel(y,y)
    xy = gaussian_kernel(x,y)
    mmd = xx.mean() + yy.mean() - 2 * xy.mean()
    return mmd

def asymmetric_l2_loss(u, tau):
    positive_errors = torch.relu(u)
    negative_errors = torch.relu(-u)
    loss = torch.mean(0.5 * tau * positive_errors ** 2 + 0.5 * (1 - tau) * negative_errors ** 2)
    return loss


def jaccard_similarity(a, b):
    a_binary = (a > 0.5).float()
    b_binary = (b > 0.5).float()

    intersection = torch.sum(a_binary * b_binary)
    union = torch.sum(torch.clamp(a_binary + b_binary, 0, 1))

    return intersection / union

def get_latent(x):
    latent = nn.Sequential(nn.Linear(x.shape[-1], 32), nn.ReLU()).to(device)
    return latent(x.to(device))

def SAC_iforest(x,policy_net,device):
    # iforest function on the penuli-layer space of SAC
    x = x.to(device)
    x = F.leaky_relu(policy_net.fc1(x)).cpu().detach().numpy()
    # calculate anomaly scores in the latent space
    iforest=IsolationForest().fit(x)
    scores = iforest.decision_function(x)
    # normalize the scores
    norm_scores = np.array([-1*s+0.5 for s in scores])
    return norm_scores

def get_total_reward(reward_e,intrinsic_rewards,s_t,write_rew=False):
    reward_i = intrinsic_rewards[s_t]
    if write_rew:
        write_reward('./results/rewards.csv',reward_i,reward_e)
    return reward_e + reward_i

def write_reward(path,r_i,r_e):
    with open(path, 'a') as f:
        f.write(f'{r_i},{r_e},')


def test_model(test_set, policy_net, device):
    policy_net.eval()

    test_X, test_y=test_set[:,:-1], test_set[:,-1]
    test_X = torch.tensor(test_X).float().to(device)
    pred_y= policy_net.sample_action(test_X,deterministic=True).detach().cpu().numpy()
    roc = roc_auc_score(test_y, pred_y)
    pr = average_precision_score(test_y, pred_y)

    policy_net.train()
    return roc,pr

def pca_feature_with_mean_replacement(data,mask_type = 'mean'):

    mean = torch.mean(data, dim=0)
    centered_data = data - mean

    u, s, v = torch.pca_lowrank(centered_data)

    explained_var_ratio = s**2 / torch.sum(s**2)
    cumulative_var_ratio = torch.cumsum(explained_var_ratio, dim=0)
    k = torch.argmax((cumulative_var_ratio >= 0.95).int()) + 1

    selected_v = v[:, :k]
    projected_data = torch.mm(centered_data, selected_v)
    reconstructed_data = torch.mm(projected_data, selected_v.t())

    data_with_removed_features = data - (centered_data - reconstructed_data)

    if mask_type == 'mean':
        mean_of_selected_dims = torch.mean(data_with_removed_features, dim=0, keepdim=True)
        mask = (reconstructed_data.abs() > 1e-7).float()
        data_replacement = mean_of_selected_dims * mask
    elif mask_type == 'zero':
        mask = (reconstructed_data == 0).float()
        data_replacement = data_with_removed_features * mask
    else:
        raise ValueError("mask_type must be 'mean' or 'zero'")
    return data_replacement


def compute_mmd(x, y):
    """
    Compute maximum mean discrepancy (MMD) between samples x and y.
    """
    xx = gaussian_kernel_matrix(x, x)
    yy = gaussian_kernel_matrix(y, y)
    xy = gaussian_kernel_matrix(x, y)
    mmd = xx.mean() + yy.mean() - 2 * xy.mean()
    return mmd

def contains_nan_or_inf(tensor):
    return torch.isnan(tensor).any() or torch.isinf(tensor).any()

def gaussian_kernel_matrix(point, buffer_points, sigma=1.0):
    """Compute the Gaussian kernel between point and multiple buffer_points"""
    differences = buffer_points - point
    squared_norms = torch.sum(differences ** 2, dim=1)
    return torch.exp(-squared_norms / (2 * (sigma ** 2)))

def compute_avg_similarity(point, buffer_points):
    """Compute the average similarity between point and all points in buffer_points"""
    similarities = gaussian_kernel_matrix(point, buffer_points)
    return torch.mean(similarities)

def inject_noise(seed, n_out, random_seed):
    '''
    add anomalies to training data to replicate anomaly contaminated data sets.
    we randomly swape 5% features of anomalies to avoid duplicate contaminated anomalies.
    this is for dense data
    '''
    rng = np.random.RandomState(random_seed)
    n_sample, dim = seed.shape
    swap_ratio = 0.05
    n_swap_feat = int(swap_ratio * dim)
    noise = np.empty((n_out, dim))
    for i in np.arange(n_out):
        outlier_idx = rng.choice(n_sample, 2, replace = False)
        o1 = seed[outlier_idx[0]]
        o2 = seed[outlier_idx[1]]
        swap_feats = rng.choice(dim, n_swap_feat, replace = False)
        noise[i] = o1.copy()
        noise[i, swap_feats] = o2[swap_feats]
    return noise
