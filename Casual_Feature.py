
import torch.nn as nn
import torch.nn.functional as F
from Point_loader import *
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.metrics import roc_curve, auc
from sklearn.cluster import KMeans
import warnings
from util import jaccard_similarity,pca_feature_with_mean_replacement

warnings.filterwarnings("ignore", category=UserWarning)


def asymmetric_l2_loss(u, tau, alpha=0.5):
    positive_errors = torch.relu(u)
    negative_errors = torch.relu(-u)
    loss = torch.mean(alpha * tau * positive_errors ** 2 + (1 - alpha) * (1 - tau) * negative_errors ** 2)
    return loss  # version 5


# BatchMultihead 用于计算每个节点的表示，GAT的工作原理是，它采用节点的输入特征，并通过一系列的图注意力层产生新的节点表示，这些表示可以用于后续的任务，如节点分类、图分类等
class BatchMultiHeadGraphAttention(nn.Module):
    def __init__(self, n_head, f_in, f_out, attn_dropout, bias=True):
        super(BatchMultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.f_in = f_in
        self.f_out = f_out
        self.w = nn.Parameter(torch.Tensor(n_head, f_in, f_out))

        self.a_src = nn.Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_dst = nn.Parameter(torch.Tensor(n_head, f_out, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter("bias", None)

        nn.init.xavier_uniform_(self.w, gain=1.414)
        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)

    def forward(self, h):
        bs, n = h.size()[:2]

        h_prime = torch.matmul(h.unsqueeze(1), self.w)
        attn_src = torch.matmul(h_prime, self.a_src)
        attn_dst = torch.matmul(h_prime, self.a_dst)
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(
            0, 1, 3, 2
        )
        attn = self.leaky_relu(attn)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, h_prime)
        if self.bias is not None:
            return output + self.bias, attn
        else:
            return output, attn

    def __repr__(self):
        return (
                self.__class__.__name__
                + " ("
                + str(self.n_head)
                + " -> "
                + str(self.f_in)
                + " -> "
                + str(self.f_out)
                + ")"
        )


class GAT(nn.Module):
    def __init__(self, n_units, n_heads, dropout=0.2, alpha=0.2):
        super(GAT, self).__init__()
        self.n_layer = len(n_units) - 1
        self.dropout = dropout
        self.layer_stack = nn.ModuleList()

        for i in range(self.n_layer):
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            self.layer_stack.append(
                BatchMultiHeadGraphAttention(
                    n_heads[i], f_in=f_in, f_out=n_units[i + 1], attn_dropout=dropout
                )
            )

        self.norm_list = [
            torch.nn.InstanceNorm1d(32,eps=1e-4),
            torch.nn.InstanceNorm1d(64,eps=1e-4),
        ]

    def forward(self, x):
        x = x.unsqueeze(1) if x.dim() == 2 else x  # 如果输入数据是2D，增加一个维度

        bs, n = x.size()[:2]

        for i, gat_layer in enumerate(self.layer_stack):

            if bs > 1:
                x = self.norm_list[i](x.permute(0, 2, 1)).permute(0, 2, 1)
            x, attn = gat_layer(x)
            if i + 1 == self.n_layer:
                x = x.squeeze(dim=2)
            else:
                x = F.elu(x.transpose(1, 2).contiguous().view(bs, n, -1))
                x = F.dropout(x, self.dropout, training=self.training)

        return x


class GATEncoder(nn.Module):
    def __init__(self, n_units, n_heads, dropout, alpha):
        super(GATEncoder, self).__init__()
        self.gat_net = GAT(n_units, n_heads, dropout, alpha)

    def forward(self, obs_traj_embedding):
        graph_embeded_data = self.gat_net(obs_traj_embedding)
        return graph_embeded_data


class CausalFeatureExtractor(nn.Module):
    def __init__(self, input_dim, gat_output_dim, mlp_hidden_dim, device, n_heads=2):
        super(CausalFeatureExtractor, self).__init__()
        self.gat_out_dim = gat_output_dim
        self.device = device
        self.gat_encoder = GATEncoder(
            n_units=[input_dim, gat_output_dim],
            n_heads=[n_heads, 1],
            dropout=0.1,
            alpha=0.2
        )

        self.Decoder = Decoder(
            input_dim=input_dim,
            hidden_dims=mlp_hidden_dim
        ).to(self.device)

        self.cross_attention = CrossAttention(
            input_dim=gat_output_dim,
            out_dim=input_dim,
            embed_dim=128
        ).to(self.device)

    def forward(self, anomaly_data, use_casual=True):

        gat_Origin_F = self.gat_encoder(anomaly_data.to(self.device))

        batch_size, n_heads, _ = gat_Origin_F.shape

        if use_casual:
            Origin_F_reshaped = pca_feature_with_mean_replacement(gat_Origin_F.view(batch_size, -1),mask_type='mean')
            gat_C_F = torch.tensor(Origin_F_reshaped, device=self.device).view(batch_size, n_heads, self.gat_out_dim)

            C_F = self.cross_attention(gat_Origin_F, gat_C_F).view(batch_size, -1)
        else:
            C_F = self.cross_attention(gat_Origin_F, gat_Origin_F).view(batch_size, -1)

        return C_F

    def get_pseudo_label(self,feature):
        pseudo_label = self.Decoder(feature)
        return pseudo_label


class CrossAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, out_dim, dropout=0.2):
        super(CrossAttention, self).__init__()

        self.embed_dim = embed_dim
        self.linear_layer = nn.Linear(input_dim, self.embed_dim)
        self.out_layer = nn.Linear(input_dim, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)
        self.scale = nn.Parameter(torch.ones(out_dim))
        self.fc_out = nn.Linear(embed_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, T, S):
        # Compute Q, K, V
        KV = self.linear_layer(S.mean(dim=1))
        Q = T.mean(dim=1)
        Q = self.linear_layer(Q)

        # Compute attention scores
        attn_scores = torch.matmul(Q, KV.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Compute weighted sum of V
        output = torch.matmul(attn_weights, KV)

        # Apply the rest of the operations
        output = self.dropout(self.fc_out(output))
        output = self.layer_norm(output) * self.scale + self.out_layer(T.mean(dim=1))

        return output


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(Decoder, self).__init__()
        self.sigmoid = nn.Sigmoid()
        layers = []
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = dim
        layers.extend([nn.Linear(hidden_dims[-1],1)])
        self.decoder = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        score = self.sigmoid(self.decoder(x))
        return score

def compute_mmd(x, y, kernel_fn):
    """
    Compute maximum mean discrepancy (MMD) between samples x and y.
    """
    xx = kernel_fn(x, x)
    yy = kernel_fn(y, y)
    xy = kernel_fn(x, y)
    mmd = xx.mean() + yy.mean() - 2 * xy.mean()
    return mmd


def gaussian_kernel(a, b):
    """
    Compute Gaussian kernel between samples a and b.
    """
    squared_distance = ((a - b) ** 2).sum(-1)
    return torch.exp(-squared_distance / 2)


def plot_radar_chart(data, labels, title="Radar Chart"):
    """
    Plot a radar chart for the given data.
    :param data: List of lists containing data. Each inner list represents a data point in 6D.
    :param labels: List of dimension names.
    :param title: Title for the chart.
    """
    num_vars = len(data[0])
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    data = np.array(data)

    # Repeat the first value to close the circle
    data = np.concatenate((data, data[:, [0]]), axis=1)
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable
    labels += labels[:1]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_rlabel_position(30)

    # Plot data
    for i, d in enumerate(data):
        ax.plot(angles, d, label=f'Sample {i + 1}')
        ax.fill(angles, d, alpha=0.25)

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title(title)
    plt.show()


if __name__ == "__main__":

    anomaly_data = CustomDataset('./datasets/annthyroid/processed/ar0.1_cr0.1/Da.csv')
    a_dataloader = DataLoader(anomaly_data, batch_size=8, shuffle=True, drop_last=True)
    unlabeled_data = CustomDataset('./datasets/annthyroid/processed/ar0.1_cr0.1/Du.csv')
    u_dataloader = DataLoader(unlabeled_data, batch_size=32, shuffle=True)
    train_data = CustomDataset('./datasets/annthyroid/processed/ar0.1_cr0.1/train_data.csv')
    training_data = DataLoader(train_data, batch_size=32, shuffle=True)

    gat_out_dim = 64
    hidden_dims = [64, 32, 6]  # 64,32,6
    model = CausalFeatureExtractor(6, gat_out_dim, hidden_dims,device='cpu')

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):  # 预训练
        for batch in u_dataloader:
            data = batch[:, :6]
            extracted_feature = model(data, use_casual=True)
            extracted_feature2 = model(data, use_casual=False)
            mmd_value = compute_mmd(extracted_feature2, extracted_feature, gaussian_kernel)
            loss =mmd_value

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    test_data = pd.read_csv('/datasets/annthyroid/processed/ar0.1_cr0.1_test/test_data.csv')
    # for epoch in range(5):  # 微调
    #     for batch in training_data:
    #         data = batch[:, :6]
    #         labels = batch[:, -1:]
    #         Feature, _ = model(data, use_casual=False)
    #         C_Feature, C_reconstructed_data = model(data, use_casual=True)
    #         CF_simi = jaccard_similarity(Feature, C_Feature)
    #         mmd_value = compute_mmd(data, C_reconstructed_data, gaussian_kernel)
    #         loss = asymmetric_l2_loss((data - C_reconstructed_data), tau=0.3, alpha=0.3)
    #
    #         weight = (labels * 9 + 1)  # 假设异常数据的权重是10
    #         loss = (loss * weight*weight).sum()  + mmd_value
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    # # test
    #
    # # test_data = pd.read_csv('/datasets/annthyroid/processed/ar0.1_cr0.1_test/test_data.csv')
    #
    # labeled_anomalies = [batch[:, :6] for batch in a_dataloader]
    # causal_features = [model(anomaly_data)[0] for anomaly_data in labeled_anomalies]
    #
    # average_causal_feature = torch.mean(torch.stack(causal_features), dim=0)
    #
    # unlabeled_features = [model(data.unsqueeze(0))[0] for batch in training_data for data in batch[:, :6]]
    # anomaly_scores = [torch.dist(feature, average_causal_feature).item() for feature in unlabeled_features]
    #
    # threshold = sum(anomaly_scores) / len(anomaly_scores) + 2 * (
    #         sum([(score - sum(anomaly_scores) / len(anomaly_scores)) ** 2 for score in anomaly_scores]) / len(
    #     anomaly_scores)) ** 0.5
    #
    # # 根据阈值分配异常标签
    # predicted_labels = [1 if score >= threshold else 0 for score in anomaly_scores]
    # print(predicted_labels.count(1))
    #
    # origin_data = pd.read_csv('./datasets/annthyroid/processed/ar0.1_cr0.1/original_train_data.csv')
    # a_data = pd.read_csv('./datasets/annthyroid/processed/ar0.1_cr0.1/Da.csv')
    # all_labels = np.concatenate((origin_data['target'].values, a_data['label'].values))
    #
    # fpr, tpr, thresholds = roc_curve(all_labels[:len(anomaly_scores)], anomaly_scores)
    # roc_auc = auc(fpr, tpr)
    # print(roc_auc)
