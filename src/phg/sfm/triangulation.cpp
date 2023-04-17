#include "triangulation.h"

#include "defines.h"

#include <Eigen/SVD>

// Используем DLT метод, составляем систему уравнений. Система похожа на систему для гомографии, там пары уравнений получались из выражений вида x (cross) Hx = 0, а здесь будет x (cross) PX = 0
// (см. Hartley & Zisserman p.312)
cv::Vec4d phg::triangulatePoint(const cv::Matx34d *Ps, const cv::Vec3d *ms, int count)
{
    // составление однородной системы + SVD
    // без подвохов
    cv::Mat A(2 * count, 4, CV_64F);
    for (int i = 0; i < count; i++) {
        const cv::Matx34d& P = Ps[i];
        const cv::Vec3d& m = ms[i];
        A.at<double>(2 * i, 0) = m[0] * P(2, 0) - P(0, 0);
        A.at<double>(2 * i, 1) = m[0] * P(2, 1) - P(0, 1);
        A.at<double>(2 * i, 2) = m[0] * P(2, 2) - P(0, 2);
        A.at<double>(2 * i, 3) = m[0] * P(2, 3) - P(0, 3);
        A.at<double>(2 * i + 1, 0) = m[1] * P(2, 0) - P(1, 0);
        A.at<double>(2 * i + 1, 1) = m[1] * P(2, 1) - P(1, 1);
        A.at<double>(2 * i + 1, 2) = m[1] * P(2, 2) - P(1, 2);
        A.at<double>(2 * i + 1, 3) = m[1] * P(2, 3) - P(1, 3);
    }

    cv::Mat w, u, vt;
    cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    cv::Vec4d X(vt.at<double>(3, 0), vt.at<double>(3, 1), vt.at<double>(3, 2), vt.at<double>(3, 3));
    X /= X[3]; 

    return X;
}
