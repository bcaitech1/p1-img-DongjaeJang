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
save_path = "/opt/ml/model_location/0408/efficientb7_test"

# 어그멘테이션
transformations = dataset.make_transform(("train"))
''' 
    ###### 3개짜리 ######
'''
# 데이터셋 준비
train_age_set = dataset.TrainDatasetForAge(dataset.train_images_path, transformations["train"])
train_gender_set = dataset.TrainDatasetForGender(dataset.train_images_path, transformations["train"])
train_mask_set = dataset.TrainDatasetForMask(dataset.train_images_path, transformations["train"])

train_age_iter = DataLoader(train_age_set, batch_size=16, shuffle=True, num_workers=2)
train_gender_iter = DataLoader(train_gender_set, batch_size=16, shuffle=True, num_workers=2)
train_mask_iter = DataLoader(train_mask_set, batch_size=16, shuffle=True, num_workers=2)

# 학습 준비

epochs = 1

model_age = EfficientNet.from_pretrained('efficientnet-b7')
model_gender = EfficientNet.from_pretrained('efficientnet-b6')
model_mask = EfficientNet.from_pretrained('efficientnet-b4')


model_age._fc = nn.Linear(in_features= 2560, out_features=3, bias = True)
model_gender._fc = nn.Linear(in_features= 2304, out_features=2, bias = True)
model_mask._fc = nn.Linear(in_features= 1792, out_features=3, bias = True)

learning_rate = 3e-4
loss_age = nn.CrossEntropyLoss()
loss_gender = nn.CrossEntropyLoss()
loss_mask = nn.CrossEntropyLoss()

optm_age = optim.Adam(model_age.parameters(), learning_rate)
optm_gender = optim.Adam(model_gender.parameters(), learning_rate)
optm_mask = optim.Adam(model_mask.parameters(), learning_rate)

model_age.cuda().float()
model_gender.cuda().float()
model_mask.cuda().float()

model_age.train()
model_gender.train()
model_mask.train()

# 모델 1번 - age 학습
best_accr = 0.0
best_loss = 10.0
for epoch in tqdm(range(epochs)) :
    loss_sum = 0

    for i, (batch_in, batch_out) in enumerate(train_age_iter) :
        y_pred = model_age.forward(batch_in.float().to(device))
        loss_out = loss_age(y_pred, batch_out.to(device))

        optm_age.zero_grad()
        loss_out.backward()
        optm_age.step()
        loss_sum += loss_out

    loss_avg = loss_sum /len(train_age_iter)
    train_accr = model_functions.func_eval(model_age, train_age_iter)
    print("="*15 + f"age {epoch + 1}" + "=" * 15)
    print(f"loss_avg : {loss_avg}, train_accr : {train_accr}")
    
    # 베스트 스코어 모델 저장
    if (train_accr > best_accr) or (loss_avg < best_loss) :
            model_functions.save_models(model_age, os.path.join(save_path, "age", str(epoch+1)))
            best_accr = train_accr
            best_loss = loss_avg

# 모델 2번 - gender 학습
best_accr = 0.0
best_loss = 10.0
for epoch in tqdm(range(epochs)) :
    loss_sum = 0

    for i, (batch_in, batch_out) in enumerate(train_gender_iter) :
        y_pred = model_gender.forward(batch_in.float().to(device))
        loss_out = loss_gender(y_pred, batch_out.to(device))

        optm_gender.zero_grad()
        loss_out.backward()
        optm_gender.step()
        loss_sum += loss_out

    loss_avg = loss_sum /len(train_gender_iter)
    train_accr = model_functions.func_eval(model_gender, train_gender_iter)
    print("="*15 + f"gender {epoch + 1}" + "=" * 15)
    print(f"loss_avg : {loss_avg}, train_accr : {train_accr}")

    # 베스트 스코어 모델 저장
    if (train_accr > best_accr) or (loss_avg < best_loss) :
            model_functions.save_models(model_gender, os.path.join(save_path, "gender", str(epoch+1)))
            best_accr = train_accr
            best_loss = loss_avg

# 모델 3번 - mask 학습
best_accr = 0.0
best_loss = 10.0
for epoch in tqdm(range(epochs)) :
    loss_sum = 0

    for i, (batch_in, batch_out) in enumerate(train_mask_iter) :
        y_pred = model_mask.forward(batch_in.float().to(device))
        loss_out = loss_mask(y_pred, batch_out.to(device))

        optm_mask.zero_grad()
        loss_out.backward()
        optm_mask.step()
        loss_sum += loss_out

    loss_avg = loss_sum /len(train_age_iter)
    train_accr = model_functions.func_eval(model_mask, train_mask_iter)
    print("="*15 + f"mask {epoch + 1}" + "=" * 15)
    print(f"loss_avg : {loss_avg}, train_accr : {train_accr}")

    # 베스트 스코어 모델 저장
    if (train_accr > best_accr) or (loss_avg < best_loss) :
            model_functions.save_models(model_mask, os.path.join(save_path, "mask", str(epoch+1)))
            best_accr = train_accr
            best_loss = loss_avg