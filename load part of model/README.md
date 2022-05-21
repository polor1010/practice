# practice

讀取 pre trained model (model_dict) 部分 module 範例

比較兩個模型中一樣名稱的 param, 如果相同就 model2_dict 更新, 然後再讀取到 model2 中

checkpoint = torch.load("./mnist_cnn.pt")
model_dict = checkpoint

print("Model's  state_dict:")
for param_tensor in model_dict:
    print(param_tensor, "\t", model_dict[param_tensor].size())
    print(param_tensor, "\t", model_dict[param_tensor])

model2_dict =  model2.state_dict() 

# 1. filter out unnecessary keys
model_dict = {k: v for k, v in model_dict.items() if k in model2_dict}
# 2. overwrite entries in the existing state dict
model2_dict.update(model_dict) 
# 3. load the new state dict
model2.load_state_dict(model2_dict)