"""Run on YOUR Windows laptop; this does not inspect remote hardware."""
import sys
import platform
print('Python:', sys.version)
print('Executable:', sys.executable)
print('OS:', platform.platform())
try:
    import torch
except ImportError:
    raise SystemExit('PyTorch is not installed in this Python environment. CPU sklearn experiments still work.')
print('PyTorch:', torch.__version__)
print('Wheel CUDA runtime:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))
    print('VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
    a = torch.randn(1024, 1024, device='cuda')
    b = a @ a.T
    torch.cuda.synchronize()
    print('GPU matrix multiplication:', b.shape, 'finite:', bool(b.isfinite().all()))
else:
    print('Check nvidia-smi, NVIDIA driver, CUDA-enabled wheel, and sys.executable. Do not spend >1 hour on this.')
