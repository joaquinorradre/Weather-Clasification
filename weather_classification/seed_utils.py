import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    # Python (random)
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch (CPU y GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Para multi-GPU

    # Opciones deterministas (opcional pero recomendable)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #print(f"[Seed] Semilla global fijada en: {seed}")
