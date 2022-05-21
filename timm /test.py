import timm
import torch
from pprint import pprint

model_names = timm.list_models(pretrained=True)
pprint(model_names)

model_name = 'regnetx_016'
m = timm.create_model(model_name, features_only=True, out_indices=(0, 1,2,3), pretrained=True)
print(f'Feature channels: {m.feature_info.channels()}')
print(f'Feature reduction: {m.feature_info.reduction()}')

o = m(torch.randn(1, 3, 224, 224))
for x in o:
  print(x.shape)