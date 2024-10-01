#include <SFML/Graphics.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <cmath>

using namespace sf;
using namespace Eigen;
using namespace std;

// 칼만 필터 클래스 정의
class KalmanFilter {
public:
    KalmanFilter(double dt) {
        delta_t = dt;

        // 상태 벡터 초기화 (x, y, w, h, vx, vy)
        x = Vector6d::Zero();

        // 상태 전이 행렬 A 초기화
        A = Matrix6d::Identity();
        A(0, 4) = delta_t;  // vx에 대한 x 위치 업데이트
        A(1, 5) = delta_t;  // vy에 대한 y 위치 업데이트

        // 관측 행렬 H 초기화 (x, y, w, h만 관측)
        H = Matrix<double, 4, 6>::Zero();
        H(0, 0) = 1;
        H(1, 1) = 1;
        H(2, 2) = 1;
        H(3, 3) = 1;

        // 프로세스 노이즈 공분산 행렬 Q 초기화
        Q = Matrix6d::Identity() * 0.1;

        // 측정 노이즈 공분산 행렬 R 초기화
        R = Matrix4d::Identity() * 0.1;

        // 추정 오차 공분산 행렬 P 초기화
        P = Matrix6d::Identity();
    }

    void predict() {
        // 상태 예측
        x = A * x;

        // 공분산 예측
        P = A * P * A.transpose() + Q;
    }

    void update(const Vector4d& z) {
        // 칼만 이득 계산
        Matrix<double, 6, 4> K = P * H.transpose() * (H * P * H.transpose() + R).inverse();

        // 상태 업데이트 (관측된 값으로 보정)
        x = x + K * (z - H * x);

        // 공분산 업데이트
        P = (Matrix6d::Identity() - K * H) * P;
    }

    Vector6d getState() {
        return x;
    }

private:
    Vector6d x;          // 상태 벡터 [x, y, w, h, vx, vy]
    Matrix6d A;          // 상태 전이 행렬
    Matrix<double, 4, 6> H; // 관측 행렬
    Matrix6d Q;          // 프로세스 노이즈 공분산 행렬
    Matrix4d R;          // 측정 노이즈 공분산 행렬
    Matrix6d P;          // 추정 오차 공분산 행렬
    double delta_t;      // 시간 간격
};

// IoU 계산 함수
double calculateIoU(const Vector4d& box1, const Vector4d& box2) {
    // box1, box2는 각각 [x_center, y_center, width, height] 형태입니다.
    
    // 두 박스의 좌상단(x1, y1)과 우하단(x2, y2) 좌표 계산
    double box1_x1 = box1(0) - box1(2) / 2;
    double box1_y1 = box1(1) - box1(3) / 2;
    double box1_x2 = box1(0) + box1(2) / 2;
    double box1_y2 = box1(1) + box1(3) / 2;
    
    double box2_x1 = box2(0) - box2(2) / 2;
    double box2_y1 = box2(1) - box2(3) / 2;
    double box2_x2 = box2(0) + box2(2) / 2;
    double box2_y2 = box2(1) + box2(3) / 2;

    // 교차 영역의 좌상단과 우하단 좌표 계산
    double inter_x1 = max(box1_x1, box2_x1);
    double inter_y1 = max(box1_y1, box2_y1);
    double inter_x2 = min(box1_x2, box2_x2);
    double inter_y2 = min(box1_y2, box2_y2);

    // 교차 영역의 넓이 계산 (겹치는 영역이 없는 경우 면적 0)
    double inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1);

    // 각 박스의 넓이 계산
    double box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1);
    double box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1);

    // IoU 계산: 교차 영역을 두 박스 넓이의 합에서 교차 영역을 뺀 값으로 나눔
    double iou = inter_area / (box1_area + box2_area - inter_area);

    return iou;
}

int main() {
    // 이미지 및 바운딩 박스 정보 경로 설정
    string img_dir = "datasets/tracking/07";
    string bbox_info_dir = "bbox_info";

    // 이미지 파일 목록 가져오기
    vector<string> img_files;
    for (int i = 0;; ++i) {
        stringstream ss;
        ss << img_dir << "/frame_" << i << ".jpg";
        ifstream file_check(ss.str());
        if (!file_check.good()) break;
        img_files.push_back(ss.str());
    }

    // 창 생성 (이미지 크기에 따라 자동 설정)
    Texture tempTexture;
    if (!tempTexture.loadFromFile(img_files[0])) {
        cerr << "이미지를 로드할 수 없습니다: " << img_files[0] << endl;
        return -1;
    }
    Vector2u img_size = tempTexture.getSize();
    RenderWindow window(VideoMode(img_size.x, img_size.y), "Object Tracking");

    // 객체 추적기 및 IOU 기반 매칭 설정
    map<int, KalmanFilter> trackers;
    map<int, int> tracker_missed;
    int next_id = 0;

    // 고유 ID별 색상
    auto getColor = [](int id) -> Color {
        static vector<Color> colors = {
            Color::Red, Color::Green, Color::Blue, Color::Magenta, Color::Cyan, Color::Yellow
        };
        return colors[id % colors.size()];
    };

    for (size_t frame_idx = 0; frame_idx < img_files.size(); ++frame_idx) {
        // 이미지 로드
        Texture texture;
        if (!texture.loadFromFile(img_files[frame_idx])) {
            cerr << "이미지를 로드할 수 없습니다: " << img_files[frame_idx] << endl;
            continue;
        }
        Sprite sprite(texture);

        // 바운딩 박스 정보 로드
        vector<Vector4d> detections;
        stringstream ss;
        ss << bbox_info_dir << "/frame_" << frame_idx << ".txt";
        ifstream bbox_file(ss.str());
        if (!bbox_file.is_open()) {
            cerr << "바운딩 박스 정보를 로드할 수 없습니다: " << ss.str() << endl;
            continue;
        }

        string line;
        while (getline(bbox_file, line)) {
            istringstream iss(line);
            double x, y, w, h;
            if (!(iss >> x >> y >> w >> h)) break;
            Vector4d detection;
            detection << x, y, w, h;
            detections.push_back(detection);
        }

        // 모든 추적기 예측 단계 수행
        for (auto& [id, tracker] : trackers) {
            tracker.predict();
        }

        // IoU 기반 매칭을 위한 비용 행렬 생성
        Matrix<double, Dynamic, Dynamic> costMatrix(trackers.size(), detections.size());

        vector<int> trk_ids;
        int idx = 0;
        for (auto& [trk_id, tracker] : trackers) {
            trk_ids.push_back(trk_id);
            auto pred = tracker.getState().head<4>(); // 상태에서 [x, y, w, h] 가져옴

            for (size_t det_idx = 0; det_idx < detections.size(); ++det_idx) {
                double iou = calculateIoU(detections[det_idx], pred);
                costMatrix(idx, det_idx) = 1.0 - iou;  // IoU가 클수록 비용이 작아지도록 설정
            }
            ++idx;
        }

        // 매칭 결과 적용 및 객체 추적 업데이트
        for (int i = 0; i < trackers.size(); ++i) {
            for (int j = 0; j < detections.size(); ++j) {
                if (costMatrix(i, j) == 0) {
                    // 매칭된 경우
                    int trk_id = trk_ids[i];
                    trackers[trk_id].update(detections[j]);
                }
            }
        }

        // 시각화 및 프레임 처리
        window.draw(sprite);
        for (const auto& [id, tracker] : trackers) {
            auto state = tracker.getState();
            Vector2f pos(state(0), state(1));
            Vector2f size(state(2) / 2, state(3) / 2); // 절반 크기로 시각화

            RectangleShape rect(size);
            rect.setPosition(pos.x - size.x / 2, pos.y - size.y / 2);
            rect.setOutlineColor(getColor(id));
            rect.setOutlineThickness(2);
            rect.setFillColor(Color::Transparent);

            window.draw(rect);
        }

        window.display();
        window.clear();
    }

    return 0;
}
