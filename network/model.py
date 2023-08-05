import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from collections import OrderedDict
from network.utils import OrthLoss, CMD

class Fusion(nn.Module):
    def __init__(self, dropout_rate, input_size, hidden_size, n_head, k):
        super(Fusion, self).__init__()
        self.loss_orth = OrthLoss()
        self.loss_cmd = CMD()

        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.activation = nn.ReLU()
        self.n_head = n_head
        self.k = k

        ##########################################
        # mapping modalities to same sized space
        ##########################################
        self.project_s = nn.Sequential()
        self.project_s.add_module('project_s',
                                  nn.Linear(in_features=input_size, out_features=input_size//4))
        self.project_s.add_module('project_s_activation', self.activation)
        self.project_s.add_module('project_s_layer_norm', nn.LayerNorm(input_size//4))

        self.project_p = nn.Sequential()
        self.project_p.add_module('project_p',
                                  nn.Linear(in_features=input_size, out_features=input_size//4))
        self.project_p.add_module('project_p_activation', self.activation)
        self.project_p.add_module('project_p_layer_norm', nn.LayerNorm(input_size//4))

        ##########################################
        # private encoders
        ##########################################
        self.private_encoder_s = nn.Sequential()
        self.private_encoder_s.add_module('private_s_1',
                                  nn.Linear(in_features=input_size//4, out_features=hidden_size))
        self.private_encoder_s.add_module('private_s_activation_1', nn.Sigmoid())

        self.private_encoder_p = nn.Sequential()
        self.private_encoder_p.add_module('private_p_1',
                                  nn.Linear(in_features=input_size//4, out_features=hidden_size))
        self.private_encoder_p.add_module('private_p_activation_1', nn.Sigmoid())

        ##########################################
        # shared encoder
        ##########################################
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=input_size//4, out_features=hidden_size))
        self.shared.add_module('shared_activation_1', nn.Sigmoid())

        ##########################################
        # shared-private collaborative discriminator
        ##########################################
        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=hidden_size * 4,
                                                           out_features=hidden_size))
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(dropout_rate))
        self.fusion.add_module('fusion_layer_1_activation', self.activation)
        self.fusion.add_module('fusion_layer_2',
                               nn.Linear(in_features=hidden_size, out_features=1))
        self.fusion.add_module('fusion_layer_2_activation', nn.Sigmoid())

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=n_head, dropout=dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.attention = GatingIntraSelfAttention(n_head, hidden_size, hidden_size, dropout=dropout_rate, gating=0)

    def get_cmd_loss(self, ):
        # losses between shared states
        loss = self.loss_cmd(self.shared_s, self.shared_p, 5)

        return loss

    def get_orth_loss(self):
        # Between private and shared
        loss = self.loss_orth(self.private_s, self.shared_s)
        loss += self.loss_orth(self.private_p, self.shared_p)

        # Across privates
        loss += self.loss_orth(self.private_s, self.private_p)

        return loss

    def shared_private(self, embed1, embed2):
        # Projecting to same sized space
        self.s_orig = s_emb = self.project_s(embed1)
        self.p_orig = p_emb = self.project_p(embed2)

        # Private-shared components
        self.private_s = self.private_encoder_s(s_emb)
        self.private_p = self.private_encoder_p(p_emb)

        self.shared_s = self.shared(s_emb)
        self.shared_p = self.shared(p_emb)


    def forward(self, embed1, embed2):
        # Shared-private encoders
        self.shared_private(embed1, embed2)

        h = torch.stack((self.private_s, self.private_p, self.shared_s, self.shared_p), dim=1)
        h = self.attention(h, h)
        h = h.view((h.shape[0], -1))
        o = self.fusion(h)

        orth_loss = self.get_orth_loss()
        cmd_loss = self.get_cmd_loss()
        repre_loss = self.k * (cmd_loss + orth_loss)

        return o, repre_loss


class GatingIntraSelfAttention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, input_size, hidden_size, dropout=0.1, gating=False):
        """
            n_head: head_num,
            input_size: input_dim,

        """
        super().__init__()

        self.n_head = n_head
        self.hidden_size = hidden_size
        self.gating = gating

        self.w_qs = nn.Linear(input_size, n_head * hidden_size)
        self.w_ks = nn.Linear(input_size, n_head * hidden_size)
        self.w_vs = nn.Linear(input_size, n_head * hidden_size)
        self.fc = nn.Linear(n_head * hidden_size, input_size)

        self.attention = ScaledDotProductAttention(temperature=hidden_size ** 0.5, gating=gating, hidden_size=hidden_size, n_head=n_head)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)


    def forward(self, g, x, mask=None):
        hidden_size, n_head = self.hidden_size, self.n_head
        batch_size, length = x.shape[0], x.shape[1]

        residual = x

        q = self.w_qs(x).view(batch_size, length, n_head, hidden_size)
        k = self.w_ks(x).view(batch_size, length, n_head, hidden_size)
        v = self.w_vs(x).view(batch_size, length, n_head, hidden_size)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        q, attn = self.attention(q, k, v, mask=mask, g=g)

        q = q.transpose(1, 2).contiguous().view(batch_size, length, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, temperature, attn_dropout=0.1, gating=False, hidden_size=None, n_head=None):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.gating = gating
        if self.gating:
            self.gating_bias = nn.parameter.Parameter(data=torch.ones((hidden_size*n_head,)))
            self.gating_linear = nn.Linear(hidden_size, n_head * hidden_size)

    def forward(self, q, k, v, mask=None, g=None):
        if self.gating:
            g_avg = g.mean(1)
            gate_values = self.gating_linear(g_avg)

            bs, n_head, q_len, q_dim = q.shape
            q = q.transpose(1, 2).contiguous().view(bs, q_len, -1)
            k = k.transpose(1, 2).contiguous().view(bs, q_len, -1)
            q = (1 + torch.unsqueeze(torch.sigmoid(gate_values + self.gating_bias), 1)) * q
            k = (1 + torch.unsqueeze(torch.sigmoid(gate_values + self.gating_bias), 1)) * k
            q = q.view(bs, q_len, -1, q_dim).permute(0, 2, 1, 3)
            k = k.view(bs, q_len, -1, q_dim).permute(0, 2, 1, 3)
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MIX_TPI(nn.Module):
    def __init__(
        self,
        tcr_padding_len,
        peptide_padding_len,
        embedding_dim,
        map_num,
        dropout_prob,
        n_head,
        gating,
        hidden_channel,
        k,
        blosum_dim=20,
    ):
        super(MIX_TPI, self).__init__()
        self.tcr_padding_len = tcr_padding_len
        self.peptide_padding_len = peptide_padding_len
        self.embedding_dim = embedding_dim
        self.map_num = map_num
        self.dropout_prob = dropout_prob
        self.n_head = n_head
        self.gating = gating
        self.hidden_channel = hidden_channel
        self.blosum_dim = blosum_dim
        self.seq_in_channels = [blosum_dim, blosum_dim, blosum_dim, blosum_dim, blosum_dim]
        self.seq_out_channels = [embedding_dim, embedding_dim, embedding_dim, embedding_dim, embedding_dim]
        self.kernel_size = [3, 5, 9, 11]
        self.aa_in_channels = [map_num, map_num, map_num, map_num]
        self.aa_out_channels = [hidden_channel, hidden_channel, hidden_channel, hidden_channel]
        self.aa_kernel_size = [[3, 3], [5, 5], [9, 9], [11, 11]]

        self.tcrs_encoder = nn.Sequential(
            OrderedDict(
                [
                    (
                        f"tcr_convolutional_{index}",
                        self.convolutional_layer1d(
                            in_channels=self.seq_in_channels[index],
                            out_channels=self.seq_out_channels[index],
                            kernel_size=kernel_size,
                            dropout=self.dropout_prob,
                        ),
                    )
                    for index, kernel_size in enumerate(self.kernel_size)
                ]
            )
        )

        self.peps_encoder = nn.Sequential(
            OrderedDict(
                [
                    (
                        f"pep_convolutional_{index}",
                        self.convolutional_layer1d(
                            in_channels=self.seq_in_channels[index],
                            out_channels=self.seq_out_channels[index],
                            kernel_size=kernel_size,
                            dropout=self.dropout_prob,
                        ),
                    )
                    for index, kernel_size in enumerate(self.kernel_size)
                ]
            )
        )

        self.tcr_intra_attn = GatingIntraSelfAttention(n_head, embedding_dim, embedding_dim, dropout=dropout_prob,
                                                       gating=self.gating)
        self.pep_intra_attn = GatingIntraSelfAttention(n_head, embedding_dim, embedding_dim, dropout=dropout_prob,
                                                       gating=self.gating)

        self.w1 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU()
        )

        self.batch_norm = nn.BatchNorm1d(tcr_padding_len * peptide_padding_len)
        self.layer_norm = nn.LayerNorm(embedding_dim)

        self.encoder = nn.Sequential(
            OrderedDict(
                [
                    (
                        f"tcr_convolutional_{index}",
                        nn.Sequential(
                            self.convolutional_layer2d(
                                input_channel=self.aa_in_channels[index],
                                output_channel=self.aa_out_channels[index],
                                kernel_size=kernel_size,
                                act_fn=nn.ReLU(),
                                batch_norm=True,
                                dropout2d=self.dropout_prob,
                            ),
                            self.convolutional_layer2d(
                                input_channel=hidden_channel,
                                output_channel=hidden_channel // 2,
                                kernel_size=kernel_size,
                                act_fn=nn.ReLU(),
                                batch_norm=True,
                                # max_pooling=False,
                                dropout2d=dropout_prob,
                            ),
                            self.convolutional_layer2d(
                                input_channel=hidden_channel // 2,
                                output_channel=1,
                                kernel_size=kernel_size,
                                act_fn=nn.ReLU(),
                                batch_norm=True,
                                # max_pooling=True,
                                dropout2d=dropout_prob,
                            ),
                        )
                    )
                    for index, kernel_size in enumerate(self.aa_kernel_size)
                ]
            )
        )

        self.Fusion = Fusion(dropout_prob, tcr_padding_len*peptide_padding_len, embedding_dim, n_head, k)

    def forward(self, data):
        tcr_embeds = data["beta_chains"]
        pep_embeds = data["peptides"]

        for index, layer in enumerate(self.tcrs_encoder):
            if index == 0:
                encoded_tcrs = layer(tcr_embeds.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                encoded_tcrs += layer(tcr_embeds.permute(0, 2, 1)).permute(0, 2, 1)

        for index, layer in enumerate(self.peps_encoder):
            if index == 0:
                encoded_peptides = layer(pep_embeds.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                encoded_peptides += layer(pep_embeds.permute(0, 2, 1)).permute(0, 2, 1)

        for index, layer in enumerate(self.encoder):
            if index == 0:
                interaction_maps_aa1 = layer(data["pp_feat"].permute(0, 3, 1, 2))
            else:
                interaction_maps_aa1 += layer(data["pp_feat"].permute(0, 3, 1, 2))
        interaction_maps_aa1 = torch.squeeze(interaction_maps_aa1, axis=1)

        encoded_tcrs = self.layer_norm(encoded_tcrs)
        encoded_peptides = self.layer_norm(encoded_peptides)

        tcr_embed = self.tcr_intra_attn(encoded_peptides, encoded_tcrs)
        pep_embed = self.pep_intra_attn(encoded_tcrs, encoded_peptides)

        interaction_maps_seq = torch.matmul(self.w1(tcr_embed), pep_embed.permute(0, 2, 1))

        merge_maps = interaction_maps_seq

        if merge_maps.shape[0] != 1:
            embed1 = self.batch_norm(merge_maps.reshape(merge_maps.shape[0], -1))
        else:
            embed1 = merge_maps.reshape(merge_maps.shape[0], -1)

        merge_maps = interaction_maps_aa1
        if merge_maps.shape[0] != 1:
            embed2 = self.batch_norm(merge_maps.reshape(merge_maps.shape[0], -1))
        else:
            embed2 = merge_maps.reshape(merge_maps.shape[0], -1)

        output, repre_loss = self.Fusion(embed1, embed2)

        return output, repre_loss

    def convolutional_layer2d(
            self,
            input_channel,
            output_channel,
            kernel_size,
            act_fn=nn.ReLU(),
            batch_norm=False,
            max_pooling=False,
            dropout2d=0.0,
    ):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        "convolve",
                        torch.nn.Conv2d(
                            input_channel,
                            output_channel,
                            kernel_size,
                            padding=[kernel_size[0] // 2, kernel_size[1] // 2],  # pad for valid conv.
                        ),
                    ),
                    ("act_fn", act_fn),
                    ("dropout2d", nn.Dropout2d(p=dropout2d)),
                    ("maxpool", nn.MaxPool2d(kernel_size=(2, 2)) if max_pooling else nn.Identity()),
                    (
                        "batch_norm",
                        nn.BatchNorm2d(output_channel) if batch_norm else nn.Identity(),
                    ),
                ]
            )
        )

    def convolutional_layer1d(
            self,
            in_channels,
            out_channels,
            kernel_size,
            act_fn=nn.ReLU(),
            dropout=0.0,
    ):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        "convolve",
                        torch.nn.Conv1d(
                            in_channels=in_channels,  # channel_in
                            out_channels=out_channels,  # channel_out
                            kernel_size=kernel_size,  # kernel_size
                            padding=kernel_size // 2,  # pad for valid conv.
                        ),
                    ),
                    ("act_fn", act_fn),
                    ("dropout", nn.Dropout(p=dropout)),
                ]
            )
        )
