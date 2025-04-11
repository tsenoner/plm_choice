import torch


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device:", torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon) device")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device
