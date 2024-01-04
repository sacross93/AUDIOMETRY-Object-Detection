import numpy as np
import torch

gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
gpu_list = ', '.join(gpu_names)

print(f"""cuda version: {torch.version.cuda}
torch version: {torch.__version__}
torch available gpu check: {torch.cuda.is_available()}
gpu count: {torch.cuda.device_count()}
gpu names: {gpu_list}""")