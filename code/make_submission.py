import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset
import model_functions

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model 정의
# 1개
# model = model_functions.load_models('resnext50', "/opt/ml/model_location/0408/resnext50_test/1", "best_model.pth")

# 3개
model_age = model_functions.load_models('effi-b7', "/opt/ml/model_location/0408/efficientb7_test/age/1", "best_model.pth")
model_gender = model_functions.load_models('effi-b6', "/opt/ml/model_location/0408/efficientb7_test/gender/1", "best_model.pth")
model_mask = model_functions.load_models('effi-b4', "/opt/ml/model_location/0408/efficientb7_test/mask/1", "best_model.pth")

# Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
test_csv = dataset.test_csv
image_paths = dataset.image_paths
transformation = dataset.make_transform(transform_type= ('test'))
test_dataset = dataset.TestDataset(image_paths, transformation['test'])

loader = DataLoader(
    test_dataset,
    shuffle=False
)


# 모델이 테스트 데이터셋을 예측하고 결과를 저장합니다.
def make_prediction (model_su, model_list, csv_file, data_loader) :
    all_predictions = []
    # 단일 모델로 클래스 18개 분류할 때
    if model_su == 1 :
        model = model_list[0]
        model.cuda().float()
        model.eval()

        for images in tqdm(data_loader):
            with torch.no_grad():
                images = images.to(device)
                pred = model(images)
                pred = pred.argmax(dim=-1)
                all_predictions.extend(pred.cpu().numpy())

    # 모델 3개로 나이, 성별, 마스크 여부 (3, 2, 3)을 분류할 때            
    elif model_su == 3 :
        model_age, model_gender, model_mask = model_list

        model_age.cuda().float()
        model_gender.cuda().float()
        model_mask.cuda().float()

        model_age.eval()
        model_gender.eval()
        model_mask.eval()

        for images in tqdm(data_loader):
            with torch.no_grad():
                images = images.to(device)
                # age
                pred_age = model_age(images)
                pred_age = pred_age.argmax(dim=-1)
                # gender
                pred_gender = model_gender(images)
                pred_gender = pred_gender.argmax(dim=-1)
                # mask
                pred_mask = model_mask(images)
                pred_mask = pred_mask.argmax(dim=-1)

                # pred
                pred = (pred_mask * 6) + (pred_gender * 3) + (pred_age)
                
                all_predictions.extend(pred.cpu().numpy())
    return all_predictions

# 1개
# all_predictions = make_prediction(1, [model], test_csv, loader)
# 3개
all_predictions = make_prediction(3, [model_age, model_gender, model_mask], test_csv, loader)

test_csv['ans'] = all_predictions

# 제출할 파일을 저장합니다.
test_csv.to_csv(os.path.join(dataset.test_path, 'submission_test2.csv'), index=False)
print('test inference is done!')