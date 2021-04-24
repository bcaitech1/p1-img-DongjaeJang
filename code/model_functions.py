from efficientnet_pytorch import EfficientNet
import os
import torch.nn as nn
import torchvision
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 모델 저장
def save_models(model, save_path) :
    os.makedirs(save_path, exist_ok=True)

    save_file = os.path.join(save_path, "best_model.pth")
    torch.save(model.state_dict(), save_file)
    print(f"Model saving success at {save_file}")
    print(f"Saved models : {os.listdir(save_path)}")

# 평가 - accuracy
def func_eval(model, data_iter) :
    with torch.no_grad() :
        n_total, n_correct = 0, 0
        model.eval()
        for batch_in, batch_out in data_iter :
            y_trgt = batch_out.to(device)
            model_pred = model(batch_in.float().to(device))
            _, y_pred = torch.max(model_pred.data, 1) # 최댓값, 최댓값의 index로 출력되는 듯?

            n_correct += (y_pred == y_trgt).sum().item()
            n_total += batch_in.size(0)
        val_accr = (n_correct / n_total)
        model.train()
    return val_accr

# 모델 불러오기
def load_models(model_name, save_path, pth_name) :
    if model_name == 'effi-b7-only' :
        new_model = EfficientNet.from_pretrained('efficientnet-b7')
        new_model._fc = nn.Linear(in_features = 2560, out_features = 18, bias=True)

        new_model.load_state_dict(torch.load(os.path.join(save_path, pth_name)))
        print(f"Model loading success from {os.path.join(save_path, pth_name)}")
        return new_model
    elif model_name == 'effi-b7' :
        new_model = EfficientNet.from_pretrained('efficientnet-b7')
        new_model._fc = nn.Linear(in_features = 2560, out_features = 3, bias=True)

        new_model.load_state_dict(torch.load(os.path.join(save_path, pth_name)))
        print(f"Model loading success from {os.path.join(save_path, pth_name)}")
        return new_model
    elif model_name == 'effi-b6' :
        new_model = EfficientNet.from_pretrained('efficientnet-b6')
        new_model._fc = nn.Linear(in_features= 2304, out_features=2, bias = True)

        new_model.load_state_dict(torch.load(os.path.join(save_path, pth_name)))
        print(f"Model loading success from {os.path.join(save_path, pth_name)}")
        return new_model
    elif model_name == 'effi-b4' :
        new_model = EfficientNet.from_pretrained('efficientnet-b4')
        new_model._fc = nn.Linear(in_features= 1792, out_features=3, bias =True)
        
        new_model.load_state_dict(torch.load(os.path.join(save_path, pth_name)))
        print(f"Model loading success from {os.path.join(save_path, pth_name)}")
        return new_model
    elif model_name == 'resnext101' :
        new_model = torchvision.models.resnext101_32x8d(pretrained=True)
        new_model.fc = nn.Linear(in_features=2048, out_features=18, bias = True)

        new_model.load_state_dict(torch.load(os.path.join(save_path, pth_name)))
        print(f"Model loading success from {os.path.join(save_path, pth_name)}")
        return new_model
    elif model_name == 'resnext50' :
        new_model = torchvision.models.resnext50_32x4d(pretrained=True)
        new_model.fc = nn.Linear(in_features=2048, out_features=18, bias = True)

        new_model.load_state_dict(torch.load(os.path.join(save_path, pth_name)))
        print(f"Model loading success from {os.path.join(save_path, pth_name)}")
        return new_model
