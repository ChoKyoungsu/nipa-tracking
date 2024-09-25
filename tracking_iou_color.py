import os
import cv2
import numpy as np
from ultralytics import YOLO
import random

# IOU (Intersection over Union) 계산 함수
def iou(bbox1, bbox2):
    # bbox: [x_center, y_center, width, height]
    # 변환: [x_min, y_min, x_max, y_max]
    x1_min, y1_min = bbox1[0] - bbox1[2] / 2, bbox1[1] - bbox1[3] / 2
    x1_max, y1_max = bbox1[0] + bbox1[2] / 2, bbox1[1] + bbox1[3] / 2
    x2_min, y2_min = bbox2[0] - bbox2[2] / 2, bbox2[1] - bbox2[3] / 2
    x2_max, y2_max = bbox2[0] + bbox2[2] / 2, bbox2[1] + bbox2[3] / 2

    # 교차 영역 계산
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    # 두 바운딩 박스의 면적
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)

    # IOU 계산
    union_area = bbox1_area + bbox2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

# 지정된 폴더에서 start_scene부터 end_scene까지의 모든 JPG 이미지를 불러오는 함수
def load_all_images_from_folders(base_folder, start_scene=1149, end_scene=1180):
    image_files = []
    for scene_num in range(start_scene, end_scene + 1):
        folder_path = os.path.join(base_folder, f'scene{scene_num}')
        if os.path.exists(folder_path):
            # 해당 폴더에 있는 모든 jpg 파일을 가져옴
            image_files += sorted([os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(".jpg")])
    return image_files

# 랜덤 색상 생성 함수 (RGB 형식)
def generate_random_color():
    return [random.randint(0, 255) for _ in range(3)]

# IOU 기반 객체 트래킹 함수
def track_objects_using_iou(yolo_model, image_files, output_video_path="output"):
    trackers = []  # 현재 프레임의 트래커 (바운딩 박스와 ID)
    tracker_colors = {}  # 각 ID별로 랜덤 색상을 저장하는 딕셔너리
    next_id = 0  # 새로운 ID
    frame_count = 0

    # 첫 번째 프레임을 사용하여 비디오 작성기 초기화
    first_frame = cv2.imread(image_files[0])
    height, width, _ = first_frame.shape
    max_iou = 0.95
    iou_thresh = 0.68
    iou_weights = 0.1

    # 비디오 작성기 초기화 (코덱, 프레임 속도, 크기 설정)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 설정 (MP4)
    out = cv2.VideoWriter(f"{output_video_path}_{iou_weights}.mp4", fourcc, 30.0, (width, height))  # 30 fps

    for image_file in image_files:
        frame = cv2.imread(image_file)
        if frame is None:
            print(f"Image not found: {image_file}")
            continue

        # YOLO를 이용한 객체 탐지
        results = yolo_model.predict(frame, conf=0.5)

        # 탐지된 객체들의 바운딩 박스 (x, y, w, h)와 클래스 정보 추출
        bboxes = results[0].boxes.xywh.cpu().numpy()  # 바운딩 박스 정보
        classes = results[0].boxes.cls.cpu().numpy()  # 클래스 정보

        # 탐지된 객체의 바운딩 박스 정보와 클래스 정보 출력
        print(f"Frame {frame_count}: Detected {len(bboxes)} objects")

        # 이전 프레임에서 트래커를 복사하여 매칭
        new_trackers = []
        for bbox, class_id in zip(bboxes, classes):
            matched = False
            for tracker in trackers:
                iou_value = iou(bbox, tracker['bbox'])
                if iou_value > min(max_iou, iou_thresh - iou_weights / (bbox[2]*bbox[3])):  # IOU가 0.8 이상이면 같은 객체로 판단
                    new_trackers.append({'bbox': bbox, 'id': tracker['id'], 'class': class_id})
                    matched = True
                    break

            if not matched:
                new_trackers.append({'bbox': bbox, 'id': next_id, 'class': class_id})
                # ID에 대해 새로운 색상을 생성하고 저장
                tracker_colors[next_id] = generate_random_color()
                next_id += 1

        # 새로운 트래커 리스트로 업데이트
        trackers = new_trackers

        # 탐지된 객체 및 트래킹된 객체 그리기
        for tracker in trackers:
            x, y, w, h = tracker['bbox']
            class_id = int(tracker['class'])
            tracking_id = tracker['id']

            # ID에 대응되는 색상을 가져옴
            color = tracker_colors[tracking_id]

            # 바운딩 박스 그리기 (탐지 결과, ID별 랜덤 색상)
            cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), color, 2)
            cv2.putText(frame, f"ID: {tracking_id} ", (int(x - w / 2), int(y - h / 2 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 현재 프레임을 비디오 파일로 저장
        out.write(frame)

        # 결과 보여주기
        cv2.imshow('Object Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    # 비디오 저장 종료
    out.release()
    cv2.destroyAllWindows()

# YOLO 모델 로드
model_path = "weights/last.pt"  # YOLO 모델 가중치 파일 경로
model = YOLO(model_path)

# 추적 실행
base_folder = "dataset/detect_240902_scene/scenes"
image_files = load_all_images_from_folders(base_folder, start_scene=1149, end_scene=1180)

# 출력할 MP4 파일 경로 지정
output_video_path = "results/output_tracking.mp4"

# 탐지 및 IOU 기반 트래킹을 수행하고 비디오 파일로 저장
track_objects_using_iou(model, image_files, output_video_path)
