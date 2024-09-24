from PIL import Image
from rl_benchmarks.models import iBOTViT
import  torch

# instantiate iBOT ViT-B Pancancer model
weights_path = 'ibot_vit_base_pancan.pth'
ibot_base_pancancer = iBOTViT(architecture="vit_base_pancan", encoder="student", weights_path=weights_path)
# load an image and transform it into a normalized tensor
image = Image.open("example.tif")  # (224, 224, 3), uint8
tensor = ibot_base_pancancer.transform(image) # (3, 224, 224), torch.float32
batch = tensor.unsqueeze(0)  # (1, 3, 224, 224), torch.float32

model = torch.jit.trace(ibot_base_pancancer.cpu(),batch.cpu())
save_path = 'iBOTViT_jit_224.pth'
torch.jit.save(model, save_path)
model = torch.jit.load(save_path)
features1 = model(batch).cpu()

features2 = ibot_base_pancancer(batch).detach().cpu()

print(features1.shape)
print(torch.allclose(features1,features2))###Used to check if two tensors are approximately equal within a certain tolerance range