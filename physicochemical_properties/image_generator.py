import numpy as np

class ImagePadding:
    def __init__(self, width, height, pad_value=0):
        super(ImagePadding, self).__init__()
        self.width = width
        self.height = height
        self.pad_value = pad_value

    def transform(self, image):
        return self._padding(image)

    def _padding(self, image):
        hor_padding = self.width - image.shape[0]
        ver_padding = self.height - image.shape[1]

        hor_padding_before = int(hor_padding // 2)
        hor_padding_after = hor_padding - hor_padding_before

        ver_padding_before = int(ver_padding // 2)
        ver_padding_after = ver_padding - ver_padding_before

        padding = (
            (hor_padding_before, hor_padding_after),
            (ver_padding_before, ver_padding_after),
            (0, 0),
        )

        padded = np.pad(image, padding, mode="constant", constant_values=self.pad_value)

        return padded

class ImageGenerator:
    def __init__(self, beta_str_list, peptide_str_list, width, height, feature_builder):
        super(ImageGenerator, self).__init__()
        self.beta_str_list = beta_str_list
        self.peptide_str_list = peptide_str_list
        self.feature_builder = feature_builder
        self.image_pad = ImagePadding(width, height, pad_value=0)

    def transform(self):
        all_image = []
        for beta, peptide in zip(self.beta_str_list, self.peptide_str_list):
            # shape: (beat_len, peptide_len, channels)
            image = self.feature_builder.generate_feature(beta, peptide)

            # shape: (padded_beta_len, padded_peptide_len, channels)
            padded_feat = self.image_pad.transform(image)
            all_image.append(padded_feat)
        all_image = np.array(all_image)
        return all_image