#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <ctime>
#include <iomanip>

// 추적할 객체를 나타내는 구조체
struct TrackedObject {
    int id;
    cv::KalmanFilter kf;
    cv::Rect2f bbox;
    cv::Mat state;
    cv::Mat measurement;
    int missing_frames;
    cv::Scalar color;
};

int main() {
    // 이미지 파일 경로 설정
    std::string base_folder = "dataset/detect_240902_scene/scenes";
    int start_scene = 1149;
    int end_scene = 1290;

    // 바운딩 박스 정보가 저장된 폴더 경로
    std::string bbox_info_folder = "bounding_box_info_0.1";

    // 이미지 파일들을 로드
    std::vector<std::string> image_files;
    for (int scene_num = start_scene; scene_num <= end_scene; ++scene_num) {
        std::stringstream ss;
        ss << base_folder << "/scene" << scene_num;
        std::string folder_path = ss.str();

        if (std::filesystem::exists(folder_path)) {
            for (const auto& entry : std::filesystem::directory_iterator(folder_path)) {
                if (entry.is_regular_file()) {
                    std::string path = entry.path().string();
                    if (path.substr(path.find_last_of(".") + 1) == "jpg") {
                        image_files.push_back(path);
                    }
                }
            }
        }
    }

    // 이미지 파일 정렬
    std::sort(image_files.begin(), image_files.end());

    // 객체 추적을 위한 변수들
    std::map<int, TrackedObject> tracked_objects;
    int next_id = 0;
    int max_missing_frames = 5;
    int frame_count = 0;

    // 난수 생성기 (고유한 색상을 위해)
    cv::RNG rng(12345);

    // 비디오 저장을 위한 설정
    cv::VideoWriter video_writer;
    bool is_video_writer_initialized = false;

    double total_time = 0;
    int total_frames = 0;

    for (const auto& image_file : image_files) {
        cv::Mat frame = cv::imread(image_file);
        if (frame.empty()) {
            std::cout << "Image not found: " << image_file << std::endl;
            continue;
        }

        if (!is_video_writer_initialized) {
            // 비디오 저장을 위한 초기화
            int width = frame.cols;
            int height = frame.rows;
            video_writer.open("output_tracking_video.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30, cv::Size(width, height));
            is_video_writer_initialized = true;
        }

        // 현재 프레임에 대한 바운딩 박스 정보 로드
        std::stringstream ss;
        ss << bbox_info_folder << "/frame_" << frame_count << ".txt";
        std::string bbox_file = ss.str();

        std::vector<cv::Rect2f> detections;

        std::ifstream infile(bbox_file);
        if (!infile.is_open()) {
            std::cout << "Bounding box file not found: " << bbox_file << std::endl;
            frame_count++;
            continue;
        }

        std::string line;
        while (std::getline(infile, line)) {
            std::stringstream linestream(line);
            std::string value;
            std::vector<float> values;

            while (std::getline(linestream, value, ',')) {
                values.push_back(std::stof(value));
            }

            if (values.size() >= 5) {
                float x = values[0];
                float y = values[1];
                float w = values[2];
                float h = values[3];
                // float class_id = values[4];  // 클래스 정보는 사용하지 않음

                // 바운딩 박스 생성 (x, y는 중심 좌표)
                cv::Rect2f bbox(x - w / 2, y - h / 2, w, h);
                detections.push_back(bbox);
            }
        }
        infile.close();

        // 기존 추적 객체와 현재 프레임의 검출 결과 매칭
        std::map<int, cv::Rect2f> updated_tracked_objects;
        std::vector<bool> detection_matched(detections.size(), false);

        for (auto& [id, obj] : tracked_objects) {
            // 칼만 필터 예측 단계

            double t0 = clock();

            obj.state = obj.kf.predict();
            cv::Point2f predicted_center(obj.state.at<float>(0), obj.state.at<float>(1));

            // 가장 가까운 검출 결과 찾기
            float min_distance = std::numeric_limits<float>::max();
            int min_index = -1;

            for (size_t i = 0; i < detections.size(); ++i) {
                if (detection_matched[i]) continue;

                cv::Point2f detection_center(detections[i].x + detections[i].width / 2, detections[i].y + detections[i].height / 2);
                float distance = cv::norm(predicted_center - detection_center);

                if (distance < min_distance && distance < 100) {  // 임계값 설정
                    min_distance = distance;
                    min_index = i;
                }
            }

            if (min_index != -1) {
                // 칼만 필터 업데이트 단계
                obj.measurement.at<float>(0) = detections[min_index].x + detections[min_index].width / 2;
                obj.measurement.at<float>(1) = detections[min_index].y + detections[min_index].height / 2;
                obj.kf.correct(obj.measurement);

                // 바운딩 박스 업데이트
                obj.bbox = detections[min_index];
                obj.missing_frames = 0;

                detection_matched[min_index] = true;
                updated_tracked_objects[id] = obj.bbox;
            } else {
                // 검출 결과와 매칭되지 않음
                obj.missing_frames++;
            }

            // TODO: 지금 scene이 끝날 때 트래킹 정보가 초기화가 안됨
            //       scene이 끝날 때마다 트래킹 정보와 bbox 초기화

            double elapsed = clock() - t0;
            total_time += elapsed;
            std::cout << "Elapsed time: " << std::setw(4) << elapsed << "ms" << std::endl;
            total_frames++;
        }

        // missing_frames가 max_missing_frames를 넘는 객체는 제거
        for (auto it = tracked_objects.begin(); it != tracked_objects.end();) {
            if (it->second.missing_frames > max_missing_frames) {
                it = tracked_objects.erase(it);
            } else {
                ++it;
            }
        }

        // 매칭되지 않은 검출 결과에 대해 새로운 추적 객체 생성
        for (size_t i = 0; i < detections.size(); ++i) {
            if (!detection_matched[i]) {
                TrackedObject obj;
                obj.id = next_id++;
                obj.bbox = detections[i];
                obj.missing_frames = 0;

                // 칼만 필터 초기화
                obj.kf = cv::KalmanFilter(4, 2, 0);
                obj.state = cv::Mat::zeros(4, 1, CV_32F);
                obj.measurement = cv::Mat::zeros(2, 1, CV_32F);

                // 전이 행렬 설정
                obj.kf.transitionMatrix = (cv::Mat_<float>(4, 4) << 
                    1, 0, 1, 0,
                    0, 1, 0, 1,
                    0, 0, 1, 0,
                    0, 0, 0, 1);

                // 측정 행렬 설정
                obj.kf.measurementMatrix = cv::Mat::zeros(2, 4, CV_32F);
                obj.kf.measurementMatrix.at<float>(0) = 1.0f;
                obj.kf.measurementMatrix.at<float>(5) = 1.0f;

                // 프로세스 노이즈 및 측정 노이즈 행렬 설정
                cv::setIdentity(obj.kf.processNoiseCov, cv::Scalar::all(1e-2));
                cv::setIdentity(obj.kf.measurementNoiseCov, cv::Scalar::all(1e-1));
                cv::setIdentity(obj.kf.errorCovPost, cv::Scalar::all(1));

                // 초기 상태 설정
                obj.state.at<float>(0) = detections[i].x + detections[i].width / 2;
                obj.state.at<float>(1) = detections[i].y + detections[i].height / 2;
                obj.state.at<float>(2) = 0;
                obj.state.at<float>(3) = 0;

                obj.kf.statePost = obj.state;

                // 고유한 색상 설정
                obj.color = cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

                tracked_objects[obj.id] = obj;
            }
        }

        // 시각화
        for (const auto& [id, obj] : tracked_objects) {
            // 바운딩 박스 그리기
            cv::rectangle(frame, obj.bbox, obj.color, 2);

            // 중심에 원 그리기
            cv::Point2f center(obj.state.at<float>(0), obj.state.at<float>(1));
            cv::circle(frame, center, 3, obj.color, -1);

            // ID 표시
            cv::putText(frame, "ID: " + std::to_string(id), cv::Point(obj.bbox.x, obj.bbox.y - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, obj.color, 2);
        }

        // 프레임을 동영상 파일에 기록
        video_writer.write(frame);

        // 결과 화면에 보여주기
        cv::imshow("Kalman Filter Tracking", frame);
        if (cv::waitKey(1) == 'q') {
            break;
        }

        frame_count++;
    }

    // 여기서 평균 시간 계산하기
    // total_time / (float)total_frames

    std::cout << "Mean elapsed time: " << std::setw(6) << total_time / (double)total_frames << "ms" << std::endl;
    
    // 동영상 파일 저장을 마무리
    if (is_video_writer_initialized) {
        video_writer.release();
    }
    cv::destroyAllWindows();

    return 0;
}
