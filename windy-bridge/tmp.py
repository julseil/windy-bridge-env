import torch
if torch.cuda.is_available():
    print('it works')
else:
    print('doesnt work')