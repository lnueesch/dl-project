
import torch
print('Torch version:', torch.__version__)
print('MPS available:', torch.backends.mps.is_available())
print('CUDA available:', torch.cuda.is_available())

