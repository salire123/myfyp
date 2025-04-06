import torch
print(torch.__version__)  # Should be 2.0.0 or higher

print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Should print RTX 3060 Ti