import numpy as np
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
import os
import cv2
import pandas as pd
from tqdm import tqdm

gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
gpu_list = ', '.join(gpu_names)

print(f"""cuda version: {torch.version.cuda}
torch version: {torch.__version__}
torch available gpu check: {torch.cuda.is_available()}
gpu count: {torch.cuda.device_count()}
gpu names: {gpu_list}""")

# Mapping category
categories = {
    1: "O", 2: "<", 3: "LeftBracket", 4: "Triangle", 5: "X", 6: ">", 7: "RightBracket", 8: "Square", 9: "S"
}

# Functions to load images and coordinate data
def load_data(image_folder, contour_folder):
    data = []
    category_counts = {category_id: 0 for category_id in categories.keys()}  # 카테고리별 개수를 저장할 사전

    for category_id, contour_name in categories.items():
        contour_path = os.path.join(contour_folder, contour_name)
        if not os.path.exists(contour_path):
            continue
        for file in os.listdir(contour_path):
            if file.endswith(".txt"):
                with open(os.path.join(contour_path, file), 'r') as f:
                    content = f.read()
                    # Centroid 데이터 찾기
                    start = content.find("Centroid: (") + len("Centroid: (")
                    end = content.find(")", start)
                    centroid = content[start:end]
                    x, y = map(int, centroid.split(", "))

                target_image_name = file.replace(".txt", ".png")
                target_image_path= os.path.join(contour_path, target_image_name)
                # print(target_image_path)
                if not os.path.exists(target_image_path):
                    print(f"Warning: Failed to find an image file in the path {target_image_path}")
                    continue  # 파일이 없으면 건너뜀
                target_img = cv2.imread(target_image_path)
                if target_img is None:
                    print(f"Warning: Failed to read the image file at path {target_image_path}")
                    continue  # 이미지 파일을 읽지 못하면 건너뜀
                h, w, _ = target_img.shape
                
                
                # 이미지 이름 변환 규칙 적용
                file_name = file.replace("contour_", "", 1).split("_contour_")[0] + ".png"
                image_path = os.path.join(image_folder, file_name)
                if os.path.exists(image_path):
                    data.append((image_path, (x, y, w, h), category_id))
                    category_counts[category_id] += 1  # 카테고리별 개수 증가
                else:
                    print(f"Warning: Failed to find an image file in the path {image_path}")

    return data, category_counts

# 데이터 및 카테고리별 개수 로드
data, category_counts = load_data("ssimuCode/cropped_graph", "ssimuCode/contours_old")

for category_id, count in category_counts.items():
    print(f"Category {categories[category_id]}: {count} items")

dataDf = pd.DataFrame(data)
print(dataDf[0].drop_duplicates())

######### MODEL
# def collate_fn(batch):
    # images = [item[0] for item in batch]
    # targets = [{k: v for k, v in item[1].items()} for item in batch]
    # images = default_collate(images)
    # return images, targets
    # images = [item['image'] for item in batch]
    # targets = [item['target'] for item in batch]
    # images = torch.stack(images, dim=0)
    # return images, targets
def collate_fn(batch):
    images = [item['image'] for item in batch]  # 이미지 추출
    targets = [item['target'] for item in batch]  # 타겟 추출
    images = torch.stack(images, dim=0)  # 이미지 텐서 스택
    return {'image': images, 'target': targets}  # 딕셔너리 형태로 반환


class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),  # PIL 이미지를 텐서로 변환
            # transforms.Resize((800, 800), antialias=True)
        ])
        
    def __len__(self):
        return len(self.data)  # 데이터셋의 전체 항목 수 반환
    
    def __getitem__(self, idx):
        image_path, bbox, label = self.data[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)  # 이미지 변환 적용

        # target 사전 생성
        target = {}
        # 바운딩 박스의 형태를 [x - w/2, y - h/2, x + w/2, y + h/2]로 변환 (Faster R-CNN에 적합한 형태)
        original_size = image.size
        x_scale = 800 / original_size[0]
        y_scale = 721 / original_size[1]
        x, y, w, h = bbox
        target["boxes"] = torch.tensor([[
            x - w / 2,  # x_min
            y - h / 2,  # y_min
            x + w / 2,  # x_max
            y + h / 2   # y_max
        ]], dtype=torch.float32)
        # target["boxes"] = torch.tensor([[
        #     x * x_scale - w * x_scale / 2,  # x_min
        #     y * y_scale - h * y_scale / 2,  # y_min
        #     x * x_scale + w * x_scale / 2,  # x_max
        #     y * y_scale + h * y_scale / 2   # y_max
        # ]], dtype=torch.float32)
        target["labels"] = torch.tensor([label], dtype=torch.int64)

        # return image, target
        return {'image': image, 'target': target}

# 2. 데이터 로드 및 데이터셋 생성
dataset = CustomDataset(data)
# 데이터셋과 데이터 로더 인스턴스 생성, 이번에는 커스텀 collate_fn 사용
data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

# 3. 사전 훈련된 Faster R-CNN 모델 로드
model = fasterrcnn_resnet50_fpn_v2(pretrained=True)

# 4. 분류기 교체 (귀하의 클래스 수에 맞게)
num_classes = len(categories) + 1 # 클래스 수 + 배경
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# 5. 학습 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

from tqdm import tqdm

# 6. 학습 루프
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    loop = tqdm(data_loader, leave=True)
    for batch in loop:
        # print(type(batch), batch.keys())
        # 딕셔너리 키로 접근하여 배치 데이터를 가져옵니다.
        images = list(img.to(device) for img in batch['image'])
        targets = [{k: v.to(device) for k, v in t.items()} for t in batch['target']]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # 진행률 표시줄 업데이트
        loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
        loop.set_postfix(loss=losses.item())
        
######################################## 모델 학습 완료

# 7. 테스트 데이터 셋 준비 및 모델을 평가 모드로 설정
test_data, test_category_counts = load_data("ssimuCode/cropped_graph", "ssimuCode/contours")
test_dataset = CustomDataset(test_data)
test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
model.eval()

# 8. 검증 루프
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
# 멀티클래스를 위한 'true_labels'와 'pred_scores' 초기화
true_labels = []
pred_scores = []

# 클래스의 개수
n_classes = len(categories)

# 검증 루프
with torch.no_grad():  # 기울기 계산을 비활성화
    for batch in tqdm(test_data_loader):
        images = list(img.to(device) for img in batch['image'])
        # 여기서 'labels' 대신 'targets'를 사용해야 합니다.
        targets = [{k: v.to(device) for k, v in t.items()} for t in batch['target']]
        
        # 모델 예측
        predictions = model(images)
        
        # 예측 결과에서 점수와 실제 라벨을 추출
        for target, prediction in zip(targets, predictions):
            # 실제 클래스 레이블을 one-hot 인코딩으로 변환
            t_labels = label_binarize(target['labels'].cpu().numpy(), classes=range(n_classes))
            true_labels.append(t_labels)
            
            # 예측 점수를 Numpy 배열로 변환 (멀티클래스 예측 점수로 변환 필요)
            p_scores = prediction['scores'].cpu().numpy()
            pred_scores.append(p_scores)

# 리스트를 Numpy 배열로 변환
true_labels = np.concatenate(true_labels, axis=0)
pred_scores = np.concatenate(pred_scores, axis=0)

len(true_labels)
len(pred_scores)

len(true_labels[0][0])
len(pred_scores[0])
pred_scores[0]

# 멀티클래스 AUC 계산
# 'average' 매개변수를 이용해 각 클래스별 AUC의 평균을 계산합니다.
auc_score = roc_auc_score(true_labels, pred_scores, average='macro')
print(f"AUC: {auc_score}")
        

#### Data set TEST code
a = cv2.imread("C:/Users/wlsdu/OneDrive/gitCode/BioLangChain/ssimuCode/contours_jy/square/contour_cropped_PTA25_contour_7.png")
b = cv2.imread("C:/Users/wlsdu/OneDrive/gitCode/BioLangChain/ssimuCode/contours_old/O/contour_cropped_ES004_POST2_contour_52.png")

h, w, _ = a.shape

file_name = file.replace("contour_", "", 1).split("_contour_")[0] + ".png"
image_path = os.path.join(image_folder, file_name)
if os.path.exists(image_path):
    print("aa")