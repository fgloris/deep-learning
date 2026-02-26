import torch                                                                                                                                                            
import time                                                                                                                                                             
                                                                                                                                                                        
size = 1024*1024*1024

for i in range(8):
  device = torch.device(f'cuda:{i}')
  
  tensor = torch.randn(size, dtype=torch.float32, device=device)
  print(f'GPU {i}: allocated ~1GB, tensor shape: {tensor.shape}')
  # 防止tensor被释放
  globals()[f'tensor_{i}'] = tensor

print('All GPUs allocated, waiting...')
while True:
  time.sleep(10)