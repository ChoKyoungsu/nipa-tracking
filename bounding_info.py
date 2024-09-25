import os
import cv2
import numpy as np
from ultralytics import YOLO

# YOLO 모델 로드
model_path = "weights/last.pt"  # 모델 경로를 지정
model = YOLO(model_path)



# 모든 폴더의 이미지 파일들을 읽어오는 함수
def load_all_images_from_folders(base_folder, start_scene=1149, end_scene=1290):
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

# 이미지 파일들로부터 객체 탐지 수행
def detect_objects_from_images(yolo_model, image_files, save_folder, output_video_path):
    frame_count = 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 비디오 코덱 설정 (mp4v는 .mp4 파일을 의미)
    video_writer = None

    for image_file in image_files:
        frame = cv2.imread(image_file)
        if frame is None:
            print(f"Image not found: {image_file}")
            continue

        if video_writer is None:
            # 비디오 저장을 위한 설정 (프레임 크기 설정)
            height, width, _ = frame.shape
            video_writer = cv2.VideoWriter(output_video_path, fourcc, 30, (width, height))

        # YOLO를 이용한 객체 탐지
        results = yolo_model.predict(frame, conf=0.5)

        # 탐지된 객체들의 바운딩 박스 (x, y, w, h)와 클래스 정보 추출
        bboxes = results[0].boxes.xywh.cpu().numpy()  # 바운딩 박스 좌표
        classes = results[0].boxes.cls.cpu().numpy()  # 클래스 정보

        # 바운딩 박스 및 클래스 정보를 파일로 저장
        save_bboxes_and_classes(frame_count, bboxes, classes, save_folder)

        # 탐지된 객체들 화면에 표시
        for i, bbox in enumerate(bboxes):
            x, y, w, h = bbox
            class_id = int(classes[i])  # 탐지된 클래스 ID
            # 바운딩 박스 그리기
            cv2.rectangle(frame, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (0, 255, 0), 2)
            cv2.putText(frame, f"Class: {class_id}", (int(x - w/2), int(y - h/2 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 프레임을 동영상 파일에 기록
        video_writer.write(frame)

        # 결과 화면에 보여주기 (옵션)
        cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    # 동영상 파일 저장을 마무리
    if video_writer is not None:
        video_writer.release()

    cv2.destroyAllWindows()

# 추적 실행 경로 및 폴더 설정
base_folder = "dataset/detect_240902_scene/scenes"
image_files = load_all_images_from_folders(base_folder, start_scene=1149, end_scene=1290)

# 저장할 폴더 경로 설정 (iou_weight 포함)
iou_weight = 0.1  # IOU 가중치 예시

save_folder = f"bounding_box_info_{iou_weight}"

# 동영상 파일 경로 설정
output_video_path = "output_detection_video.mp4"

# 객체 탐지 수행 및 바운딩 박스/클래스 정보 저장, 동영상 저장
detect_objects_from_images(model, image_files, save_folder, output_video_path)
