
很方便用來替換 backbone 用

```python=
#列出支援哪些 model 
model_names = timm.list_models(pretrained=True)
pprint(model_names)


#載入 regnetx_016 並印出每一層的特徵數量
model_name = 'regnetx_016'
m = timm.create_model(model_name, features_only=True, out_indices=(0, 1,2,3), pretrained=True)
print(f'Feature channels: {m.feature_info.channels()}')
print(f'Feature reduction: {m.feature_info.reduction()}')

o = m(torch.randn(1, 3, 224, 224))
for x in o:
  print(x.shape)

```