#include <opencv2/opencv.hpp>
#include <fstream>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

struct BoundingBox {
    int class_id;
    Rect box;
};

// 칼만 필터 클래스
class KalmanTracker {
public:
    KalmanTracker(Point2f pt) {
        kalman = KalmanFilter(4, 2, 0);  // 상태: x, y, 속도 vx, vy, 측정: x, y
        state = Mat::zeros(4, 1, CV_32F);  // [x, y, vx, vy]
        measurement = Mat::zeros(2, 1, CV_32F);  // [x, y]

        // 전이 행렬 설정
        kalman.transitionMatrix = (Mat_<float>(4, 4) << 1, 0, 1, 0,
                                                        0, 1, 0, 1,
                                                        0, 0, 1, 0,
                                                        0, 0, 0, 1);
        // 측정 행렬 설정
        setIdentity(kalman.measurementMatrix);
        setIdentity(kalman.processNoiseCov, Scalar::all(1e-4));
        setIdentity(kalman.measurementNoiseCov, Scalar::all(1e-1));
        setIdentity(kalman.errorCovPost, Scalar::all(1));

        state.at<float>(0) = pt.x;
        state.at<float>(1) = pt.y;
        kalman.statePost = state;
    }

    Point2f predict() {
        Mat prediction = kalman.predict();
        return Point2f(prediction.at<float>(0), prediction.at<float>(1));
    }

    Point2f update(Point2f pt) {
        measurement.at<float>(0) = pt.x;
        measurement.at<float>(1) = pt.y;
        kalman.correct(measurement);
        return predict();
    }

private:
    KalmanFilter kalman;
    Mat state;
    Mat measurement;
};

// 바운딩박스 중앙점 계산
Point2f getCenter(const Rect& rect) {
    return Point2f(rect.x + rect.width / 2, rect.y + rect.height / 2);
}

// 바운딩박스 정보 로드
vector<BoundingBox> loadBoundingBoxes(const string& filename) {
    vector<BoundingBox> boxes;
    ifstream file(filename);
    string line;
    while (getline(file, line)) {
        istringstream ss(line);
        int class_id, x, y, w, h;
        ss >> class_id >> x >> y >> w >> h;
        boxes.push_back({class_id, Rect(x, y, w, h)});
    }
    return boxes;
}

int main() {
    string image_dir = "dataset/detect_240902_scene/scenes/";
    string bbox_dir = "dataset/detect_240902_scene/bounding_box_info/";

    vector<KalmanTracker> trackers;

    // 첫 프레임을 초기화하는 변수
    bool is_first_frame = true;

    for (int scene = 1149; scene <= 1180; ++scene) {
        string scene_str = "scene" + to_string(scene);
        string image_path = image_dir + scene_str + ".jpg";
        string bbox_path = bbox_dir + scene_str + ".txt";

        Mat frame = imread(image_path);
        if (frame.empty()) {
            cerr << "Could not load image: " << image_path << endl;
            continue;
        }

        // 바운딩박스 정보 로드
        vector<BoundingBox> boxes = loadBoundingBoxes(bbox_path);

        // 첫 프레임이라면 모든 바운딩박스에 대해 새 추적기 초기화
        if (is_first_frame) {
            for (const auto& box : boxes) {
                Point2f center = getCenter(box.box);
                KalmanTracker new_tracker(center);
                trackers.push_back(new_tracker);
            }
            is_first_frame = false;  // 첫 프레임 처리 완료
        } else {
            // 기존 추적기 업데이트 및 시각화
            vector<Point2f> centers;
            for (const auto& box : boxes) {
                Point2f center = getCenter(box.box);
                centers.push_back(center);
            }

            // 기존 객체 업데이트
            for (int i = 0; i < trackers.size(); ++i) {
                Point2f prediction = trackers[i].predict();
                if (i < centers.size()) {
                    Point2f updated = trackers[i].update(centers[i]);
                    rectangle(frame, boxes[i].box, Scalar(255, 0, 0), 2);
                    circle(frame, updated, 5, Scalar(0, 255, 0), -1);
                    putText(frame, "ID: " + to_string(i), updated, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);
                }
            }

            // 새 객체 추가
            for (int i = trackers.size(); i < centers.size(); ++i) {
                KalmanTracker new_tracker(centers[i]);
                trackers.push_back(new_tracker);
            }
        }

        // 결과 시각화
        imshow("Tracking", frame);
        waitKey(30);
    }

    return 0;
}
