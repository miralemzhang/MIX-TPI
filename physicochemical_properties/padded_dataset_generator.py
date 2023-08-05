import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from physicochemical_properties.feature_builder import FeatureBuilder
from physicochemical_properties.image_generator import ImageGenerator

def padded_dataset_generator(
    beta_str_list,
    peptide_str_list,
    width,
    height,
    feature_builder: FeatureBuilder,
):
    """Create a tensorflow dataset with positive and negative 2d interaction map arrays.

    Can optionally export the positive and generated negative sequence pairs
    to a csv file.

    Parameters
    ----------
    data_stream : DataStream
        A DataStream of positive labeled cdr3-epitope sequence pairs. Expected fromat ( ("CDR3","EPITOPE"), 1)
    feature_builder : FeatureBuilder
        A FeatureBuilder object that can convert the sequences into pairwise interaction arrays.
    cdr3_range : Tuple[int, int]
        The minimum and maximum desired cdr3 sequence length.
    epitope_range : Tuple[int, int]
        The minimum and maximum desired epitope sequence length.
    neg_shuffle : bool
        Whether to create negatives by shuffling/sampling, by default True.
        NOTE: Should always be set to False when evaluating a dataset that already contains negatives.
    epitope_ratio : boolean
        When false, samples an epitope for each CDR3 sequence in the
        proportionally to its occurrence in the other epitope pairs. Does not
        preserve the ratio of positives and negatives within each epitope,
        but does result in every CDR3 sequence having exactly 1 positive and negative.
        When true, samples a set of CDR3 sequences with from the unique list of CDR3s
        for each epitope observation (per epitope), i.e. preserves exact ratio of positives and
        negatives for each epitope, at the expense of some CDR3s appearing more than once
        among the negatives and others only in positives pairs.
    Returns
    -------
    tf.data.Dataset
        A tensorflow DataSet, ready to be used as input for a model.
        NOTE: should still be shuffled and batched.
    """

    width = width
    height = height

    image_gen = ImageGenerator(beta_str_list, peptide_str_list, width, height, feature_builder)
    image_feat = image_gen.transform()
    return image_feat