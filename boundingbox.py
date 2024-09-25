import os
import cv2
import numpy as np
from ultralytics import YOLO

# YOLO 모델 로드
model_path = "weights/last.pt"  # 모델 경로를 지정
model = YOLO(model_path)

# 모든 폴더의 이미지 파일들을 읽어오는 함수
def load_all_images_from_folders(base_folder, start_scene=1149, end_scene=1180):
    image_files = []
    for scene_num in range(start_scene, end_scene + 1):
        folder_path = os.path.join(base_folder, f'scene{scene_num}')
        if os.path.exists(folder_path):
            # 해당 폴더에 있는 모든 jpg 파일을 가져옴
            image_files += sorted([os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(".jpg")])
    return image_files

    
# 결과를 저장할 폴더 이름 및 파일 번호 생성 함수
def get_next_folder_and_file():
    base_result_folder = "result"
    base_video_name = "output_tracking"
    folder_num = 1
    file_num = 1

    # 폴더 이름 증가
    while os.path.exists(f"{base_result_folder}{folder_num}"):
        folder_num += 1
    
    # 새로운 폴더 경로 생성
    result_folder = f"{base_result_folder}{folder_num}"
    os.makedirs(result_folder, exist_ok=True)

    # 비디오 파일 이름 설정
    while os.path.exists(f"{base_video_name}{file_num}.mp4"):
        file_num += 1

    video_file = f"{base_video_name}{file_num}.mp4"
    
    return result_folder, video_file

# 바운딩 박스 및 클래스 정보를 저장하는 함수
def save_bboxes_and_classes(frame_count, bboxes, classes, save_folder):
    file_path = os.path.join(save_folder, f"frame_{frame_count}.txt")

    with open(file_path, 'w') as f:
        for bbox, class_id in zip(bboxes, classes):
            x, y, w, h = bbox
            f.write(f"{x},{y},{w},{h},{int(class_id)}\n")  # x, y, w, h, class_id 저장

# 이미지 파일들로부터 객체 탐지 및 시각화 수행, 결과 저장
def detect_objects_from_images(yolo_model, image_folder, save_folder, video_file):
    # 이미지 파일 목록 불러오기
    image_files = sorted([os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(".jpg")])

    if not image_files:
        print("Error: No image files found. Please check the folder path and file types.")
        return

    frame_count = 0
    first_frame = cv2.imread(image_files[0])
    height, width, _ = first_frame.shape

    # 비디오 저장 설정 (코덱, 프레임 속도, 크기 설정)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 설정 (MP4)
    out = cv2.VideoWriter(video_file, fourcc, 30.0, (width, height))  # 30 fps

    for image_file in image_files:
        frame = cv2.imread(image_file)
        if frame is None:
            print(f"Image not found: {image_file}")
            continue

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

        # 현재 프레임을 비디오 파일에 저장
        out.write(frame)

        # 결과 보여주기
        cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    out.release()
    cv2.destroyAllWindows()


# 실행 함수
def main():
    # 이미지 폴더 경로
    base_folder = "dataset/detect_240902_scene/scenes"
    image_files = load_all_images_from_folders(base_folder, start_scene=1149, end_scene=1180)

    # 이미지 파일이 제대로 불러와졌는지 확인
    if not image_files:
        print("No image files found in the specified folder.")
        return

    print(f"Total images loaded: {len(image_files)}")  # 불러온 이미지 파일 개수 출력

    # 결과를 저장할 폴더 및 비디오 파일 설정
    save_folder, video_file = get_next_folder_and_file()

    # 객체 탐지 수행 및 바운딩 박스/클래스 정보 저장, MP4로 저장
    detect_objects_from_images(model, base_folder, save_folder, video_file)

if __name__ == "__main__":
    main()
``