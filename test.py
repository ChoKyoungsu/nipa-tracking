import os
import cv2
import numpy as np
from ultralytics import YOLO

# 지정된 폴더에서 start_scene부터 end_scene까지의 모든 JPG 이미지를 불러오는 함수
def load_all_images_from_folders(base_folder, start_scene=1149, end_scene=1180):
    image_files = []
    for scene_num in range(start_scene, end_scene + 1):
        folder_path = os.path.join(base_folder, f'scene{scene_num}')
        if os.path.exists(folder_path):
            # 해당 폴더에 있는 모든 jpg 파일을 가져옴
            image_files += sorted([os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(".jpg")])
    return image_files

# 바운딩 박스 및 클래스 정보를 저장하는 함수
def save_bboxes_and_classes(frame_count, bboxes, classes, save_folder):
    os.makedirs(save_folder, exist_ok=True)  # 폴더가 없으면 생성
    file_path = os.path.join(save_folder, f"frame_{frame_count}.txt")

    with open(file_path, 'w') as f:
        for bbox, class_id in zip(bboxes, classes):
            x, y, w, h = bbox
            f.write(f"{x},{y},{w},{h},{int(class_id)}\n")  # x, y, w, h, class_id 저장

# YOLO 기반 객체 탐지 및 시각화 수행 함수
def detect_and_visualize_objects(yolo_model, image_files, save_folder, output_video_path="output"):
    frame_count = 0

    # 첫 번째 프레임을 사용하여 비디오 작성기 초기화
    first_frame = cv2.imread(image_files[0])
    height, width, _ = first_frame.shape

    # 비디오 저장 설정 (코덱, 프레임 속도, 크기 설정)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 설정 (MP4)
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))  # 30 fps

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

        # 바운딩 박스 및 클래스 정보를 파일로 저장
        save_bboxes_and_classes(frame_count, bboxes, classes, save_folder)

        # 탐지된 객체들 화면에 표시
        for i, bbox in enumerate(bboxes):
            x, y, w, h = bbox
            class_id = int(classes[i])  # 탐지된 클래스 ID
            # 바운딩 박스 그리기
            cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)
            cv2.putText(frame, f"Class: {class_id}", (int(x - w / 2), int(y - h / 2 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 현재 프레임을 비디오 파일에 저장
        out.write(frame)

        # 결과 보여주기
        cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    # 비디오 저장 종료
    out.release()
    cv2.destroyAllWindows()

# YOLO 모델 로드
model_path = "weights/last.pt"  # YOLO 모델 가중치 파일 경로
model = YOLO(model_path)

# 실행 경로 및 폴더 설정
base_folder = "dataset/detect_240902_scene/scenes"
image_files = load_all_images_from_folders(base_folder, start_scene=1149, end_scene=1180)

# 저장할 바운딩 박스 정보 폴더 설정
save_folder = "bounding_box_info"

# 출력할 MP4 파일 경로 지정
output_video_path = "result/output_detection.mp4"

# 탐지 및 시각화, 바운딩 박스/클래스 정보 저장
detect_and_visualize_objects(model, image_files, save_folder, output_video_path)
