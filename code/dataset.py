import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from glob import glob
from PIL import Image
import torchvision.transforms as transforms

'''
    Dataset을 불러오기 전 train, test의 경로를 지정한다

    < Train >
    1. train_path : train 폴더
    2. train_csv_path : train.csv 위치
    3. train_images_path : train 이미지셋이 있는 root 폴더

    < Test >
    1. test_path : eval 폴더
    2. test_csv_path : info.csv 위치
    3. test_images_path : test 이미지셋이 있는 폴더
'''

# Train paths
train_path = "/opt/ml/input/data/train"
train_csv_path = os.path.join(train_path, "train.csv")
train_images_path = os.path.join(train_path, "images")
# Test paths
test_path = "/opt/ml/input/data/eval"
test_csv_path = os.path.join(test_path, "info.csv")
test_images_path = os.path.join(test_path, "images")

# submission csv
test_csv = pd.read_csv(test_csv_path)
# test image files
image_paths = [os.path.join(test_images_path, img_id) for img_id in test_csv.ImageID]

'''
    Train Images 개수 : 18900장
    Test Images 개수 : 12600장

    이슈 :
        ** Train Images의 경우 라벨이 안맞는 경우가 있음. 성별도 마찬가지
            => 어떻게 해결할지 고려
'''

'''
    Augmentation 만들기

    필요해보이는 것 
    - 1. CenterCrop
    - 2. 명암 ?
'''
# transform만드는 함수
def make_transform(transform_type, mean = [0.548, 0.504, 0.479], std = [0.237, 0.247, 0.246]) :
    transformations = {}

    if "train" in transform_type :
        transformations["train"] = transforms.Compose([
            transforms.CenterCrop((400, 300)),
            # transforms.ColorJitter(brightness=(0.2, 3), contrast=(0.2, 3), saturation=(0.2, 3), hue=(-0.5, 0.5)),
            transforms.ToTensor(),
            transforms.Normalize(mean = mean, std = std)
        ])

    if "valid" in  transform_type :
        transformations["valid"] = transforms.Compose([
            transforms.CenterCrop((400, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean = mean, std = std)
        ])
    
    if "test" in transform_type :
        transformations["test"] = transforms.Compose([
            transforms.CenterCrop((400, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean = mean, std = std)
        ])

    return transformations
    
'''
    Dataset 종류

    1. TrainDataset : train_images_path의 이미지 데이터셋

    2. TestDataset : test_images_path에 있는 이미지 데이터셋

    3. TrainDatasetForAge : train_images_path의 이미지 데이터셋을 나이로만 라벨링
        - 1 ~ 29 : 0
        - 30 ~ 59 : 1
        - 60 ~ : 2

    4. TrainDatasetForGender : train_images_path의 이미지 데이터셋을 성별로만 라벨링
        - male : 0
        - female : 1

    5. TrainDatasetForMask : train_images_path의 이미지 데이터셋을 마스크 여부로만 라벨링
        - mask : 0
        - incorrect : 1
        - normal : 2
'''
# 나이 구분 함수
def classify_age(age) :
    return 0 if int(age) < 30 else 1 if int(age) < 58 else 2
# 성별 구분 함수
def classify_gender(gender) :
    return 0 if gender == 'male' else 1
# 마스크 구분 함수
def classify_mask(mask) :
    return 2 if "normal" in mask else 1 if "incorrect" in mask else 0

# 1. TrainDataset
class TrainDataset(Dataset) :
    def __init__(self, root_path=train_images_path, transform = None) :
        self.root_path = root_path
        self.transform = transform
        self.images = []
        self.labels = []

        self.give_label()
        
    def __getitem__(self, index) :
        img_path = self.images[index]
        img = Image.open(img_path)
        label = self.labels[index]

        if self.transform : 
            img = self.transform(img)

        return img, label

    def __len__(self) :
        return len(self.images)

    def give_label(self) :
        folder_list = glob(os.path.join(self.root_path, '*'))
        for folder_name in folder_list :
            file_list = glob(os.path.join(folder_name, '*'))
            _id, gender, race, age = folder_name.split('/')[-1].split('_')

            age_label = classify_age(age)
            gender_label = classify_gender(gender)
            for file_name in file_list :
                real_name = file_name.split('/')[-1]
                mask_label = classify_mask(real_name)
                # print(real_name)
                self.images.append(file_name)
                self.labels.append(mask_label * 6 + gender_label * 3 + age_label)

    def set_transform(self, transform) :
        self.transform = transform


# 2. TestDataset
class TestDataset(Dataset):
    def __init__(self, img_paths=image_paths, transform = None):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)


# 3. TrainDatasetForAge
class TrainDatasetForAge(Dataset) :
    def __init__(self, root_path=train_images_path, transform = None) :
        self.root_path = root_path
        self.transform = transform
        self.images = []
        self.labels = []

        self.give_label()

    def __getitem__(self, index) :
        img_path = self.images[index]
        img = Image.open(img_path)
        label = self.labels[index]

        if self.transform :
            img = self.transform(img)

        return img, label

    def __len__(self) :
        return len(self.images)
    
    def give_label(self) :
        folder_list = glob(os.path.join(self.root_path, "*"))
        for folder_name in folder_list :
            file_list = glob(os.path.join(folder_name, "*"))
            age = folder_name.split("/")[-1].split('_')[-1]
            age_label = classify_age(age)

            for file_name in file_list :
                self.images.append(file_name)
                self.labels.append(age_label)
    
    def set_transform(self, transform) :
        self.transform = transform



# 4. TrainDatasetForGender
class TrainDatasetForGender(Dataset) :
    def __init__(self, root_path=train_images_path, transform = None) :
        self.root_path = root_path
        self.transform = transform
        self.images = []
        self.labels = []

        self.give_label()

    def __getitem__(self, index) :
        img_path = self.images[index]
        img = Image.open(img_path)
        label = self.labels[index]

        if self.transform :
            img = self.transform(img)

        return img, label

    def __len__(self) :
        return len(self.images)
    
    def give_label(self) :
        folder_list = glob(os.path.join(self.root_path, "*"))
        for folder_name in folder_list :
            file_list = glob(os.path.join(folder_name, "*"))
            gender = folder_name.split("/")[-1].split('_')[1]
            gender_label = classify_gender(gender)

            for file_name in file_list :
                self.images.append(file_name)
                self.labels.append(gender_label)
    
    def set_transform(self, transform) :
        self.transform = transform



# 5. TrainDatsetForMask
class TrainDatasetForMask(Dataset) :
    def __init__(self, root_path=train_images_path, transform = None) :
        self.root_path = root_path
        self.transform = transform
        self.images = []
        self.labels = []

        self.give_label()

    def __getitem__(self, index) :
        img_path = self.images[index]
        img = Image.open(img_path)
        label = self.labels[index]
        
        if self.transform :
            img = self.transform(img)

        return img, label

    def __len__(self) :
        return len(self.images)
    
    def give_label(self) :
        folder_list = glob(os.path.join(self.root_path, "*"))
        for folder_name in folder_list :
            file_list = glob(os.path.join(folder_name, "*"))

            for file_name in file_list :
                real_name = file_name.split('/')[-1]
                mask_label = classify_mask(real_name)

                self.images.append(file_name)
                self.labels.append(mask_label)
    
    def set_transform(self, transform) :
        self.transform = transform