# 检测哪些gpu可以正常使用
# import torch

# def test_device(device):
#     try:
#         x = torch.tensor([1.], device=device)
#         y = x * 2
#         print(f"设备 {device} 可以正常工作。")
#     except:
#         print(f"设备 {device} 无法工作。")

# if torch.cuda.is_available():
#     device_count = torch.cuda.device_count()
#     for i in range(device_count):
#         device = torch.device(f'cuda:{i}')
#         test_device(device)
# else:
#     print("CUDA不可用，无法进行测试。")

# 检测mmcv、mmdet、mmrotate、pytorch、cuda是否有冲突
# Check Pytorch installation
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

# Check MMDetection installation
import mmdet
print(mmdet.__version__)

# Check mmcv installation
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print(get_compiling_cuda_version())
print(get_compiler_version())

# 检测torch是否gpu版本
# import torch

# def check_gpu_version():
#     torch_version = torch.__version__
#     cuda_available = torch.cuda.is_available()
#     cuda_version = torch.version.cuda if cuda_available else None
    
#     print(f"PyTorch version: {torch_version}")
#     if cuda_available:
#         print(f"CUDA version: {cuda_version}")
#         print("PyTorch is using GPU version.")
#     else:
#         print("PyTorch is using CPU version.")

# check_gpu_version()
