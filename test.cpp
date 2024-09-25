#include <iostream>

using namespace std;

int main(){

    cout << "\nhello world !" << endl;
    int num1 = 10;
    int num2 = 0;

    num2 - num1 * 7;
    cout << "num2 : " << num2 << endl;
    return 0;
}// 바운딩박스 중앙점 계산
Point2f getCenter(const Rect& rect) {
    return Point2f(rect.x + rect.width / 2, rect.y + rect.height / 2);
}