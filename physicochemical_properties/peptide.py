import random
from physicochemical_properties.peptide_feature import Charge, Hydrophobicity, IsoelectricPoint, Mass, Hydrophilicity, Surface, Flexibility, Transfer
from physicochemical_properties.util import AMINO_ACIDS, nomalize_matrix

class Peptide(str):
    def __init__(self, seq):
        super().__init__()
        if not (isinstance(seq, Peptide) or isinstance(seq, str)):
            raise RuntimeError("Peptide sequence should be of type str or Peptide")

    def get_charge_vector(self):
        return nomalize_matrix(Charge().calculate(self))

    def get_hydrophobicity_vector(self):
        return nomalize_matrix(Hydrophobicity().calculate(self))

    def get_isoelectric_vector(self):
        return nomalize_matrix(IsoelectricPoint().calculate(self))

    def get_mass_vector(self):
        return nomalize_matrix(Mass().calculate(self))

    def get_hydrophilicity_vector(self):
        return nomalize_matrix(Hydrophilicity().calculate(self))

    def get_surface_vector(self):
        return nomalize_matrix(Surface().calculate(self))

    def get_flexibility_vector(self):
        return nomalize_matrix(Flexibility().calculate(self))

    def get_transfer_vector(self):
        return nomalize_matrix(Transfer().calculate(self))

    @staticmethod
    def random(length=10):
        return Peptide("".join(random.choice(AMINO_ACIDS) for _ in range(length)))

    def generate_match(self, feature, length=5, random_inverse=True):
        start_index = random.randint(0, len(self) - length)
        match = ""
        for index, aa in enumerate(self):
            if index < start_index:  # start is random
                match += random.choice(AMINO_ACIDS)
            elif index < start_index + length:  # generate random match
                t = feature.generate_match(aa)
                match += t
            else:  # fill with random
                match += random.choice(AMINO_ACIDS)

        if random_inverse and random.randint(1, 2) == 1:
            match = "".join(reversed(match))
        return Peptide(match)
