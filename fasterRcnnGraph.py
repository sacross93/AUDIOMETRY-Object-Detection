########## import 및 데이터 셋을 정의하고 불러오기
# 1. All imports
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

# Check torch version and gpu is availability
gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
gpu_list = ', '.join(gpu_names)
print(f"""cuda version: {torch.version.cuda}
torch version: {torch.__version__}
torch available gpu check: {torch.cuda.is_available()}
gpu count: {torch.cuda.device_count()}
gpu names: {gpu_list}""")

# 2. Mapping category
categories = {
    1: "LAB", 2: "LB", 3: "X", 4: "RB", 5: "L", 6: "O", 7: "S", 8: "Square", 9: "RAB", 10: "Triangle"
}

# 3. 데이터 셋 불러오기
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
                target_image_path = os.path.join(contour_path, target_image_name)
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

# 4. 데이터 및 카테고리별 개수 로드
data, category_counts = load_data("data/cropped_graph", "data/contours")

# 5. 카테고리 별 수집 개수와 읽은 파일 숫자
for category_id, count in category_counts.items():
    print(f"Category {categories[category_id]}: {count} items")

dataDf = pd.DataFrame(data)
print(f"Read data count: {len(dataDf[0].drop_duplicates())}")

########## MODEL 코드 시작
# 1. 데이터 함수 정의
def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [{k: v for k, v in item[1].items()} for item in batch]
    images = default_collate(images)
    return images, targets

imageSize = [800, 721] # 비율 조절 800 * 721, 원본 1480*1334

class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),  # PIL 이미지를 텐서로 변환
            transforms.Resize((imageSize[0], imageSize[1]), antialias=True)
        ])

    def __len__(self):
        return len(self.data)  # 데이터셋의 전체 항목 수 반환

    def __getitem__(self, idx):
        image_path, bbox, label = self.data[idx]
        image = Image.open(image_path).convert("RGB")
        original_image_size = image.size
        image = self.transform(image)  # 이미지 변환 적용

        target = {} # target 사전 생성
        ## 바운딩 박스의 형태를 [x - w/2, y - h/2, x + w/2, y + h/2]로 변환 (Faster R-CNN에 적합한 형태)
        ## 이미지 리사이즈에 따라 자동으로 바운딩 박스도 리사이즈 될 수 있도록 함
        x_scale = imageSize[0] / original_image_size[0]
        y_scale = imageSize[1] / original_image_size[1]
        x, y, w, h = bbox
        target["boxes"] = torch.tensor([[
            x * x_scale - w * x_scale / 2,  # x_min
            y * y_scale - h * y_scale / 2,  # y_min
            x * x_scale + w * x_scale / 2,  # x_max
            y * y_scale + h * y_scale / 2  # y_max
        ]], dtype=torch.float32)
        target["labels"] = torch.tensor([label], dtype=torch.int64)

        return image, target

# 2. 데이터 로드 및 데이터셋 생성
dataset = CustomDataset(data)
## 데이터셋과 데이터 로더 인스턴스 생성, 이번에는 커스텀 collate_fn 사용
data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

# 3. 사전 훈련된 Faster R-CNN 모델 로드
model = fasterrcnn_resnet50_fpn_v2(pretrained=True)

# 4. 분류기 교체 (귀하의 클래스 수에 맞게)
num_classes = len(categories) + 1  # 클래스 수 + 배경
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# 5. 학습 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# 6. 학습 루프
num_epochs = 2
for epoch in range(num_epochs):
    model.train()
    loop = tqdm(data_loader, leave=True)
    for images, targets in loop:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        ## 진행률 표시줄 업데이트
        loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
        loop.set_postfix(loss=losses.item())

########## MODEL TEST 코드
# 1. 테스트 데이터 셋 준비 및 모델을 평가 모드로 설정
test_data, test_category_counts = load_data("ssimuCode/cropped_graph", "ssimuCode/contours")
test_dataset = CustomDataset(test_data)
test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
model.eval()

# 2. 검증 루프
from sklearn.preprocessing import label_binarize
true_labels = []
pred_scores = []
prediction_labels = []
image_paths = []

n_classes = len(categories)

with torch.no_grad():
    for images, targets in tqdm(test_data_loader):
        images = list(img.to(device) for img in images)
        predictions = model(images)

        for target, prediction in zip(targets, predictions):
            t_labels = target['labels'].cpu().numpy()
            true_labels.append(t_labels)

            p_labels = prediction.get('labels', torch.tensor([], dtype=torch.int64)).cpu().numpy()
            if len(p_labels) > 0:
                one_hot_p_labels = label_binarize(p_labels, classes=range(1, n_classes + 1))
                prediction_labels.append(one_hot_p_labels)

                p_scores = prediction.get('scores', torch.tensor([], dtype=torch.float32)).cpu().numpy()
                one_hot_p_scores = np.zeros((1, n_classes))
                one_hot_p_scores[:, p_labels] = p_scores
                pred_scores.append(one_hot_p_scores)
            else:
                prediction_labels.append(np.zeros((1, n_classes)))
                pred_scores.append(np.zeros((1, n_classes)))

# 3. 결과 출력, 만약 결과가 없다면 아무것도 출력되지 않음
for true, pred, score, path in zip(true_labels, prediction_labels, pred_scores, image_paths):
    print(f'Image Path: {path} - True Label: {true}, Predicted Label: {np.argmax(pred)}, Score: {np.max(score)}')

########## MODEL TEST 코드2
# 1. 테스트 데이터 셋 준비 및 모델을 평가 모드로 설정
test_data, test_category_counts = load_data("ssimuCode/cropped_graph", "ssimuCode/contours")
test_dataset = CustomDataset(test_data)
test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
model.eval()

# 2. 변수 설정 및 예측 임계값 설정
true_boxes = []
predicted_boxes = []
true_labels = []
predicted_boxes2 = []
predicted_labels = []

model.roi_heads.score_thresh = 0.5

# 3. 검증 루프
with torch.no_grad():
    for images, targets in tqdm(test_data_loader):
        images = list(img.to(device) for img in images)
        outputs = model(images)

        for target, output in zip(targets, outputs):
            true_boxes.append(target['boxes'].cpu().numpy())
            predicted_boxes.append(output['boxes'].cpu().numpy())
            true_labels.extend(target['labels'].cpu().numpy())
            predicted_boxes2.extend(output['boxes'].cpu().numpy())
            predicted_labels.extend(output['labels'].cpu().numpy())


# 4. 결과 출력
print(f"True Bounding Boxes: {true_boxes[0]}")
print(f"Predicted Bounding Boxes: {predicted_boxes[0]}")
predicted_category_names = [categories[label] for label in predicted_labels]
print(f"True Labels: {true_labels[0]}")
print(f"Predicted Bounding Boxes: {predicted_boxes[0]}")
print(f"Predicted Labels: {predicted_labels[0]}")
print(f"Predicted Category Names: {predicted_category_names[0]}")