import torch
import cx_kernels

a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
b = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32)

out = cx_kernels.add_kernel(a, b)
print(out)  # tensor([5., 7., 9.])