import os
import torch
import torch.nn as nn
import torchvision
from efficientnet_pytorch import EfficientNet
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset
import loss
import model_functions

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 저장 위치
save_path = "/opt/ml/model_location/0408/resnext50_test"

# 어그멘테이션
transformations = dataset.make_transform(("train"))

''' 
    ###### 1개짜리 ######
'''
# 데이터셋 준비
train_set = dataset.TrainDataset(dataset.train_images_path, transformations["train"])

train_iter = DataLoader(train_set, batch_size = 16, shuffle= True, num_workers=3)

# 학습 준비
epochs = 1

model = torchvision.models.resnext50_32x4d(pretrained=True)
model.fc = nn.Linear(in_features=2048, out_features=18, bias=True)

learning_rate = 1e-4
loss = nn.CrossEntropyLoss()
optm = optim.Adam(model.parameters(), lr= learning_rate)

model.cuda()
model.train()

best_accr = 0.0
best_loss = 10.0

for epoch in tqdm(range(epochs)) :
    loss_sum = 0
    for i, (batch_in, batch_out) in tqdm(enumerate(train_iter)) :
        y_pred = model.forward(batch_in.float().to(device))
        loss_out = loss(y_pred, batch_out.to(device))

        optm.zero_grad()
        loss_out.backward()
        optm.step()
        loss_sum += loss_out

    loss_avg = loss_sum /len(train_iter)
    train_accr = model_functions.func_eval(model, train_iter)
    print("="*15 + f"epoch {epoch + 1}" + "=" * 15)
    print(f"loss_avg : {loss_avg}, train_accr : {train_accr}")

    # 베스트 스코어 모델 저장
    if (train_accr > best_accr) or (loss_avg < best_loss) :
            model_functions.save_models(model, os.path.join(save_path, str(epoch+1)))
            best_accr = train_accr
            best_loss = loss_avg