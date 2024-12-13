import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from einops import rearrange, repeat

# GAT Layer Implementation
class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, dropout=0.6, alpha=0.2):
        super(GATLayer, self).__init__()
        self.attn = GATConv(in_channels, out_channels, heads=heads, dropout=dropout, negative_slope=alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.attn(x, edge_index)  # Apply attention to the graph structure
        x = F.elu(x)  # Activation function (ELU)
        x = self.dropout(x)  # Dropout for regularization
        return x

# Graph Encoder with GAT Layer
class GraphEncoder(nn.Module):
    def __init__(self, attn_layers, gnn_layers, gl_layer, node_embs, cls_len, norm_layer=None):
        super(GraphEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.graph_layers = nn.ModuleList(gnn_layers)  # This will now contain GAT layers
        self.graph_learning = gl_layer
        self.norm = norm_layer
        self.cls_len = cls_len
        self.node_embs = node_embs

    def forward(self, x, edge_index, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]  (batch_size, seq_len, feature_dim)
        attns = []
        gcls_len = self.cls_len
        adj = self.graph_learning(self.node_embs)  # Adjacency matrix computation

        for i, attn_layer in enumerate(self.attn_layers):
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)

            if i < len(self.graph_layers):
                g = x[:, :gcls_len]
                g = rearrange(g, '(b n) p d -> (b p) n d', n=self.node_embs.shape[0])
                g = self.graph_layers[i](g, edge_index) + g  # Apply GAT layer
                g = rearrange(g, '(b p) n d -> (b n) p d', p=gcls_len)
                x[:, :gcls_len] = g

            if self.norm is not None:
                x = self.norm(x)

        return x, attns


# Flatten Head Class for Prediction
class Flatten_Head(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


# Graph Constructor for Adjacency Matrix
class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, alpha=1, static_feat=None):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes

        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)

        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, node_emb):
        nodevec1 = F.gelu(self.alpha * self.lin1(node_emb))
        nodevec2 = F.gelu(self.alpha * self.lin2(node_emb))
        adj = F.relu(torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0)))
        if self.k < node_emb.shape[0]:
            n_nodes = node_emb.shape[0]
            mask = torch.zeros(n_nodes, n_nodes).to(node_emb.device)
            mask.fill_(float('0'))
            s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(self.k, 1)
            mask.scatter_(1, t1, s1.fill_(1))
            adj = adj * mask
        return adj


# Main Model with GAT Layer Integration
class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8, gc_alpha=1):
        """
        patch_len: int, patch length for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride
        cls_len = configs.cls_len
        gdep = configs.graph_depth
        knn = configs.knn
        embed_dim = configs.embed_dim

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)
        # global tokens
        self.cls_token = nn.Parameter(torch.randn(1, cls_len, configs.d_model))
        
        # Encoder (using Graph Attention Layer)
        self.encoder = GraphEncoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            [GATLayer(configs.d_model, configs.d_model, heads=3, dropout=configs.dropout) for _ in range(configs.e_layers-1)],  # Use GAT instead of mixprop
            graph_constructor(configs.enc_in, knn, embed_dim, alpha=gc_alpha),
            nn.Parameter(torch.randn(configs.enc_in, embed_dim), requires_grad=True), cls_len,
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        # Prediction Head
        self.head_nf = configs.d_model * \
            int((configs.seq_len - patch_len) / stride + 2)
        if 'forecast' in self.task_name.lower():
            self.head = Flatten_Head(configs.enc_in, self.head_nf, configs.pred_len,
                                     head_dropout=configs.dropout)
        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.head = Flatten_Head(configs.enc_in, self.head_nf, configs.seq_len,
                                     head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                self.head_nf * configs.enc_in, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)
        # cls token
        patch_len = enc_out.shape[1]
        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=enc_out.shape[0])
        enc_out = torch.cat([cls_tokens, enc_out], dim=1)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        enc_out = enc_out[:, -patch_len:, :]
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
            (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
            (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if 'forecast' in self.task_name.lower():
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
