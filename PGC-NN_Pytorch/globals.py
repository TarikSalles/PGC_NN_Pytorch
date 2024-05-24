import torch


class Configurations:
    DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    # DEVICE = "cpu"
