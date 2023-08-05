from typing import Dict
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
import torch
import sys
from torch.utils.data import Sampler

from physicochemical_properties.feature_builder import CombinedPeptideFeatureBuilder
from physicochemical_properties.peptide_feature import parse_features, parse_operator
from physicochemical_properties.padded_dataset_generator import padded_dataset_generator

blosum62 = {
    "A": np.array(
        (4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0)
    ),
    "R": np.array(
        (-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3)
    ),
    "N": np.array(
        (-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3)
    ),
    "D": np.array(
        (-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3)
    ),
    "C": np.array(
        ( 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1)
    ),
    "Q": np.array(
        (-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2)
    ),
    "E": np.array(
        (-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2)
    ),
    "G": np.array(
        ( 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2 ,-3 ,-3)
    ),
    "H": np.array(
        (-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3)
    ),
    "I": np.array(
        (-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3 , 1 , 0 ,-3, -2, -1, -3, -1,  3)
    ),
    "L": np.array(
        (-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1)
    ),
    "K": np.array(
        (-1,  2, 0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2)
    ),
    "M": np.array(
        (-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1)
    ),
    "F": np.array(
        (-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1)
    ),
    "P": np.array(
        (-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2)
    ),
    "S": np.array(
        (1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2)
    ),
    "T": np.array(
        (0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0)
    ),
    "W": np.array(
        (-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11,  2, -3)
    ),
    "Y": np.array(
        (-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2 ,-2 ,2 ,7 ,-1)
    ),
    "V": np.array(
        (0,-3,-3,-3,-1,-2,-2,-3,-3,3,1,-2,1,-1,-2,-2,0,-3,-1,4)
    ),
}

amino_acids = [letter for letter in "ARNDCEQGHILKMFPSTWYV"]
amino_to_index = {
    amino: index for index, amino in enumerate(["PAD"] + amino_acids)
}

vocab_size = len(amino_to_index)
amino_to_index["[UNK]"] = vocab_size
tokenizer = np.vectorize(lambda x: amino_to_index.get(x, amino_to_index["[UNK]"]))

def pad_mask(tcr_true_len: int, peptide_true_len: int, tcr_max_len: int , peptide_max_len) -> np.ndarray:
    tcr_delta = tcr_max_len - tcr_true_len
    peptide_delta = peptide_max_len - peptide_true_len
    mask_one = np.ones((tcr_true_len, peptide_true_len))
    return np.pad(mask_one, pad_width=((0, tcr_delta), (0, peptide_delta)), constant_values=(0, 0))

def read_data(data_path: str) -> DataFrame:
    return pd.read_csv(data_path)

class DataLoader(Sampler):
    def __init__(
            self,
            data: DataFrame,
            batch_size,
            args,
            tcr_padding_length=-1,
            peptide_padding_length=-1,
            save_dir='',
            operator="absdiff",
            device="cuda",
    ):
        self.features = args.features
        self.operator = operator
        self.tcr_padding_length = tcr_padding_length
        self.peptide_padding_length = peptide_padding_length
        self.embedding_martix = blosum62
        self.vocab_size = len(amino_to_index)
        self.device = device
        self.batch_size = batch_size
        self.data = data.sample(frac=1)
        self.num_batches = int(
            np.ceil(len(self.data) / self.batch_size)
        )

        self.batchify()
        self.tokenize()

    def batchify(self) -> None:
        self.batches = []
        k = -1
        for i in range(self.num_batches - 1):
            self.batches.append(
                self.data.iloc[i * self.batch_size: (i + 1) * self.batch_size, :]
            )
            k = i
        self.batches.append(self.data.iloc[(k + 1) * self.batch_size:, :])

        return

    def tokenize(self) -> None:
        for c, batch in enumerate(self.batches):
            self.batches[c] = self.tokenize_batch(batch)

        return

    def tokenize_batch(self, batch: DataFrame) -> Dict:
        beta_str_list = batch.iloc[:, 0].values.tolist()
        peptide_str_list = batch.iloc[:, -2].values.tolist()
        features_list = parse_features(self.features)
        operator = parse_operator(self.operator)
        feature_builder = CombinedPeptideFeatureBuilder(features_list, operator)

        beta_len_list = [
            len(x) for x in beta_str_list
        ]
        peptide_len_list = [
            len(x) for x in peptide_str_list
        ]

        width = (
            np.max(beta_len_list)
            if self.tcr_padding_length == -1
            else self.tcr_padding_length
        )

        height = (
            np.max(peptide_len_list)
            if self.peptide_padding_length == -1
            else self.peptide_padding_length
        )

        mask = np.stack([pad_mask(x, y, width, height) for x, y in zip(beta_len_list, peptide_len_list)])

        pp_feat = padded_dataset_generator(
            beta_str_list,
            peptide_str_list,
            width,
            height,
            feature_builder=feature_builder
        )

        peptide_lens = [
            len(peptide) for peptide in batch.iloc[:, 1].values.tolist()
        ]

        pep_max_len = self.peptide_padding_length

        peptides = embedding_with_given_matrix(
            batch.iloc[:, 1].values.tolist(), self.embedding_martix, pep_max_len
        )

        beta_chain_lens = [
            len(beta_chain) for beta_chain in batch.iloc[:, 0].values.tolist()
        ]

        tcr_max_len = (
            np.max(beta_chain_lens)
            if self.tcr_padding_length == -1
            else self.tcr_padding_length
        )

        beta_chains = embedding_with_given_matrix(
            batch.iloc[:, 0].values.tolist(), self.embedding_martix, tcr_max_len
        )

        labels = np.array(
            batch.iloc[:, -1].values.tolist()
        )


        return {
            "beta_chains": torch.tensor(beta_chains, dtype=torch.float).to(self.device),
            "peptides": torch.tensor(peptides, dtype=torch.float).to(self.device),
            "pp_feat": torch.tensor(pp_feat, dtype=torch.float).to(self.device),
            "mask": torch.tensor(mask, dtype=torch.float).to(self.device),
            "tcr_lens": torch.tensor(beta_chain_lens, dtype=torch.long).to(self.device),
            "peptide_lens": torch.tensor(peptide_lens, dtype=torch.long).to(self.device),
            "labels": torch.tensor(labels, dtype=torch.float),
        }

    def batch_count(self):
        return self.num_batches

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for i in self.batches:
            yield i


def embedding_with_given_matrix(aa_seqs, embedding_martix, max_seq_len):
    sequences = []

    for seq in aa_seqs:
        e_seq = np.zeros((len(seq), len(embedding_martix["A"])))

        for i, aa in enumerate(seq):
            if aa in embedding_martix:
                e_seq[i] = embedding_martix[aa]
            else:
                sys.stderr.write(
                    "Unknown amino acid in sequence: " + aa + ", encoding aborted!\n"
                )
                sys.exit(2)

        sequences.append(e_seq)

    num_seqs = len(aa_seqs)
    num_features = sequences[0].shape[1]

    embedded_aa_seq = np.zeros(
        (num_seqs, max_seq_len, num_features)
    )

    for i in range(0, num_seqs):
        embedded_aa_seq[i, : sequences[i].shape[0], :num_features] = sequences[i]

    return embedded_aa_seq
