#include <opencv2/opencv.hpp>
#include <string>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cv.h>
#include <highgui.h>
using namespace std;
using namespace cv;

// this program shows how to use optical flow

string file_1 = "/home/lxd/my_code/project/disparity_optical_flow/left.png";  // first image
string file_2 = "/home/lxd/my_code/project/disparity_optical_flow/right.png";  // second image

// TODO implement this funciton
/**
 * single level optical flow
 * @param [in] img1 the first image
 * @param [in] img2 the second image
 * @param [in] kp1 keypoints in img1
 * @param [in|out] kp2 keypoints in img2, if empty, use initial guess in kp1
 * @param [out] success true if a keypoint is tracked successfully
 * @param [in] inverse use inverse formulation?
 */
void OpticalFlowSingleLevel(
        const Mat &img1,
        const Mat &img2,
        const vector<KeyPoint> &kp1,
        vector<KeyPoint> &kp2,
        vector<bool> &success,
        bool inverse = false
);

// TODO implement this funciton
/**
 * multi level optical flow, scale of pyramid is set to 2 by default
 * the image pyramid will be create inside the function
 * @param [in] img1 the first pyramid
 * @param [in] img2 the second pyramid
 * @param [in] kp1 keypoints in img1
 * @param [out] kp2 keypoints in img2
 * @param [out] success true if a keypoint is tracked successfully
 * @param [in] inverse set true to enable inverse formulation
 */
void OpticalFlowMultiLevel(
        const Mat &img1,
        const Mat &img2,
        const vector<KeyPoint> &kp1,
        vector<KeyPoint> &kp2,
        vector<bool> &success,
        bool inverse = false
);

/**
 * get a gray scale value from reference image (bi-linear interpolated)
 * @param img
 * @param x
 * @param y
 * @return
 */
inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
            (1 - xx) * (1 - yy) * data[0] +
            xx * (1 - yy) * data[1] +
            (1 - xx) * yy * data[img.step] +
            xx * yy * data[img.step + 1]
    );
}


int main(int argc, char **argv) {

    // images, note they are CV_8UC1, not CV_8UC3
    Mat img1 = imread(file_1, 0);
    Mat img2 = imread(file_2, 0);
    // key points, using GFTT here.
    vector<KeyPoint> kp1;
    Ptr<GFTTDetector> detector = GFTTDetector::create(500, 0.01, 20); // maximum 500 keypoints
    detector->detect(img1, kp1);

    // now lets track these key points in the second image
    // first use single level LK in the validation picture
    vector<KeyPoint> kp2_single;
    vector<bool> success_single;
    OpticalFlowSingleLevel(img1, img2, kp1, kp2_single, success_single);

    // then test multi-level LK
    vector<KeyPoint> kp2_multi;
    vector<bool> success_multi;
    OpticalFlowMultiLevel(img1, img2, kp1, kp2_multi, success_multi);

    // use opencv's flow for validation
    vector<Point2f> pt1, pt2;
    for (auto &kp: kp1) pt1.push_back(kp.pt);
    vector<uchar> status;
    vector<float> error;
    cv::calcOpticalFlowPyrLK(img1, img2, pt1, pt2, status, error, cv::Size(8, 8));

    // plot the differences of those functions
    Mat img2_single;
    cv::cvtColor(img2, img2_single, CV_GRAY2BGR);
    for (int i = 0; i < kp2_single.size(); i++) {
        if (success_single[i]) {
            cv::circle(img2_single, kp2_single[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_single, kp1[i].pt, kp2_single[i].pt, cv::Scalar(0, 250, 0));
        }
    }

    Mat img2_multi;
    cv::cvtColor(img2, img2_multi, CV_GRAY2BGR);
    for (int i = 0; i < kp2_multi.size(); i++) {
        if (success_multi[i]) {
            cv::circle(img2_multi, kp2_multi[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_multi, kp1[i].pt, kp2_multi[i].pt, cv::Scalar(0, 250, 0));
        }
    }

    Mat img2_CV;
    cv::cvtColor(img2, img2_CV, CV_GRAY2BGR);
    for (int i = 0; i < pt2.size(); i++) {
        if (status[i]) {
            cv::circle(img2_CV, pt2[i], 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_CV, pt1[i], pt2[i], cv::Scalar(0, 250, 0));
        }
    }
    cv::imshow("tracked single level", img2_single);
    cv::imshow("tracked multi level", img2_multi);
    cv::imshow("tracked by opencv", img2_CV);
    cv::waitKey(0);

    return 0;
}

void OpticalFlowSingleLevel(
        const Mat &img1,
        const Mat &img2,
        const vector<KeyPoint> &kp1,
        vector<KeyPoint> &kp2,
        vector<bool> &success,
        bool inverse
) {

    // parameters
    int half_patch_size = 4;
    int iterations = 10;
    bool have_initial = !kp2.empty();
    Mat disparity(img2.rows,img2.cols,CV_8U,Scalar::all(0)),vdisparity;
    vector<KeyPoint>keypoint_1,keypoint_2;
    for (size_t i = 0; i < kp1.size(); i++) {
        auto kp = kp1[i];
        double dx = 0, dy = 0; // dx,dy need to be estimated
        if (have_initial) {
            dx = kp2[i].pt.x - kp.pt.x;
            dy = kp2[i].pt.y - kp.pt.y;
        }

        double cost = 0, lastCost = 0;
        bool succ = true; // indicate if this point succeeded
        // Gauss-Newton iterations
        for (int iter = 0; iter < iterations; iter++) {
            Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
            Eigen::Vector2d b = Eigen::Vector2d::Zero();
            cost = 0;

            if (kp.pt.x + dx <= half_patch_size || kp.pt.x + dx >= img1.cols - half_patch_size ||
                kp.pt.y + dy <= half_patch_size || kp.pt.y + dy >= img1.rows - half_patch_size) {   // go outside
                succ = false;
                break;
            }

            // compute cost and jacobian
            for (int x = -half_patch_size; x < half_patch_size; x++)
                for (int y = -half_patch_size; y < half_patch_size; y++) {

                    // TODO START YOUR CODE HERE (~8 lines)
                    double error = 0;
                    Eigen::Vector2d J;  // Jacobian
                    J<<0,0;
                    if (inverse == false) {
                        // Forward Jacobian
                        //判断边界
                        if (kp.pt.x + x <= half_patch_size || kp.pt.x - x >= img1.cols - half_patch_size ||
                            kp.pt.y + y <= half_patch_size || kp.pt.y - y >= img1.rows - half_patch_size) {
                            break;
                        }
                        else {
                            //计算雅阁比矩阵和误差
                            J<<(GetPixelValue(img2,kp.pt.x+x+dx+1,kp.pt.y+dy+y)-GetPixelValue(img2,kp.pt.x+x+dx-1,kp.pt.y+y+dy))/2,
                             (GetPixelValue(img2,kp.pt.x+x+dx,kp.pt.y+y+dy+1)-GetPixelValue(img2,kp.pt.x+x+dx,kp.pt.y+y+dy-1))/2;
                            error=GetPixelValue(img1,kp.pt.x+x,kp.pt.y+y)-GetPixelValue(img2,kp.pt.x+x+dx,kp.pt.y+y+dy);
                        }

                    } else {
                        // Inverse Jacobian
                        J<<(GetPixelValue(img1,kp.pt.x+x+1,kp.pt.y+y)-GetPixelValue(img1,kp.pt.x+x-1,kp.pt.y+y))/2,
                         (GetPixelValue(img1,kp.pt.x+x,kp.pt.y+y+1)-GetPixelValue(img1,kp.pt.x+x,kp.pt.y+y-1))/2;
                        error=GetPixelValue(img1,kp.pt.x+x,kp.pt.y+y)-GetPixelValue(img2,kp.pt.x+x+dx,kp.pt.y+y+dy);
                        // NOTE this J does not change when dx, dy is updated, so we can store it and only compute error
                    }
                    // compute H, b and set cost;
                    H+=J*J.transpose();
                    b+=error*J;
                    cost+=error*error;
                    // TODO END YOUR CODE HERE
                }

            // compute update
            // TODO START YOUR CODE HERE (~1 lines)
            Eigen::Vector2d update;
            update=H.ldlt().solve(b);
            // TODO END YOUR CODE HERE

            if (isnan(update[0])) {
                // sometimes occurred when we have a black or white patch and H is irreversible
                cout << "update is nan" << endl;
                succ = false;
                break;
            }
            if (iter > 0 && cost > lastCost) {
                cout << "cost increased: " << cost << ", " << lastCost << endl;
                break;
            }

            // update dx, dy
            dx += update[0];
            dy += update[1];
            lastCost = cost;
            succ = true;
        }

        success.push_back(succ);

        // set kp2
        if (have_initial) {
            kp2[i].pt = kp.pt + Point2f(dx, dy);
        } else {
            KeyPoint tracked = kp;
            tracked.pt += cv::Point2f(dx, dy);
            kp2.push_back(tracked);
        }
        keypoint_1.push_back(kp1[i]);
        keypoint_2.push_back(kp2[i]);
    }
    for(int i=0;i<keypoint_2.size();i++)
    {
        disparity.at<float>(keypoint_2[i].pt.y,keypoint_2[i].pt.x)=keypoint_1[i].pt.x-keypoint_2[i].pt.x;
    }
    Mat dis =imread("/home/lxd/my_code/project/disparity_optical_flow/disparity.png",0);
    dis=dis - disparity;
    imshow("disparity",dis);
    waitKey(0);
}

void OpticalFlowMultiLevel(
        const Mat &img1,
        const Mat &img2,
        const vector<KeyPoint> &kp1,
        vector<KeyPoint> &kp2,
        vector<bool> &success,
        bool inverse) {

    // parameters
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};
    // create pyramids
    vector<Mat> pyr1, pyr2; // image pyramids
    // TODO START YOUR CODE HERE (~8 lines)
    for (int i = 0; i < pyramids; i++) {
        Mat a,b;
        Size dsize1 = Size(img1.cols*scales[i], img1.rows*scales[i]);
        Size dsize2 = Size(img2.cols*scales[i], img2.rows*scales[i]);
        resize(img1,a,dsize1,0,0);
        resize(img2,b,dsize2,0,0);
        pyr1.push_back(a);
        pyr2.push_back(b);
    }
    // TODO END YOUR CODE HERE
    // coarse-to-fine LK tracking in pyramids
    int half_patch_size = 4;
    int iterations = 10;
    bool have_initial = !kp2.empty();

    for (size_t i = 0; i < kp1.size(); i++) {
        auto kp = kp1[i];
        double dx = 0, dy = 0; // dx,dy need to be estimated
        if (have_initial) {
            dx = kp2[i].pt.x - kp.pt.x;
            dy = kp2[i].pt.y - kp.pt.y;
        }

        double cost = 0, lastCost = 0;
        bool succ = true; // indicate if this point succeeded
        // Gauss-Newton iterations
        for (int iter = 0; iter < iterations; iter++) {
            Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
            Eigen::Vector2d b = Eigen::Vector2d::Zero();
            cost = 0;

            if (kp.pt.x/8 + dx <= half_patch_size || kp.pt.x/8 + dx >= pyr1[3].cols - half_patch_size ||
                kp.pt.y/8 + dy <= half_patch_size || kp.pt.y/8 + dy >= pyr1[3].rows - half_patch_size) {   // go outside
                succ = false;
                break;
            }

            // compute cost and jacobian
            for (int x = -half_patch_size; x < half_patch_size; x++)
                for (int y = -half_patch_size; y < half_patch_size; y++) {

                    // TODO START YOUR CODE HERE (~8 lines)
                    double error = 0;
                    Eigen::Vector2d J;  // Jacobian
                    J<<0,0;

                        // Forward Jacobian
                        //判断边界
                    if (kp.pt.x/8 + x <= half_patch_size || kp.pt.x/8 - x >= pyr1[3].cols - half_patch_size ||
                        kp.pt.y/8 + y <= half_patch_size || kp.pt.y/8 - y >= pyr1[3].rows - half_patch_size) {
                            break;
                    }
                    else {
                            //计算雅阁比矩阵和误差
                        J<<(GetPixelValue(pyr2[3],kp.pt.x/8+x+dx+1,kp.pt.y/8+dy+y)-GetPixelValue(pyr2[3],kp.pt.x/8+x+dx-1,kp.pt.y/8+y+dy))/2,
                                    (GetPixelValue(pyr2[3],kp.pt.x/8+x+dx,kp.pt.y/8+y+dy+1)-GetPixelValue(pyr2[3],kp.pt.x/8+x+dx,kp.pt.y/8+y+dy-1))/2;
                        error=GetPixelValue(pyr1[3],kp.pt.x/8+x,kp.pt.y/8+y)-GetPixelValue(pyr2[3],kp.pt.x/8+x+dx,kp.pt.y/8+y+dy);
                    }

                    H+=J*J.transpose();
                    b+=error*J;
                    cost+=error*error;
                    // TODO END YOUR CODE HERE
                }

            // compute update
            // TODO START YOUR CODE HERE (~1 lines)
            Eigen::Vector2d update;
            update=H.ldlt().solve(b);
            // TODO END YOUR CODE HERE

            if (isnan(update[0])) {
                // sometimes occurred when we have a black or white patch and H is irreversible
                cout << "update is nan" << endl;
                succ = false;
                break;
            }
            if (iter > 0 && cost > lastCost) {
                cout << "cost increased: " << cost << ", " << lastCost << endl;
                for (int iter = 0; iter < iterations; iter++) {
                    Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
                    Eigen::Vector2d b = Eigen::Vector2d::Zero();
                    cost = 0;

                    if (kp.pt.x/4 + dx <= half_patch_size || kp.pt.x/4 + dx >= pyr1[2].cols - half_patch_size ||
                        kp.pt.y/4 + dy <= half_patch_size || kp.pt.y/4 + dy >= pyr1[2].rows - half_patch_size) {   // go outside
                        succ = false;
                        break;
                    }

                    // compute cost and jacobian
                    for (int x = -half_patch_size; x < half_patch_size; x++)
                        for (int y = -half_patch_size; y < half_patch_size; y++) {

                            // TODO START YOUR CODE HERE (~8 lines)
                            double error = 0;
                            Eigen::Vector2d J;  // Jacobian
                            J<<0,0;

                                // Forward Jacobian
                                //判断边界
                            if (kp.pt.x/4 + x <= half_patch_size || kp.pt.x/4 - x >= pyr1[2].cols - half_patch_size ||
                                kp.pt.y/4 + y <= half_patch_size || kp.pt.y/4 - y >= pyr1[2].rows - half_patch_size) {
                                break;
                            }
                            else {
                                    //计算雅阁比矩阵和误差
                                J<<(GetPixelValue(pyr2[2],kp.pt.x/4+dx+x+1,kp.pt.y/4+dy+y)-GetPixelValue(pyr2[2],kp.pt.x/4+x+dx-1,kp.pt.y/4+y+dy))/2,
                                (GetPixelValue(pyr2[2],kp.pt.x/4+x+dx,kp.pt.y/4+y+dy+1)-GetPixelValue(pyr2[2],kp.pt.x/4+x+dx,kp.pt.y/4+y+dy-1))/2;
                                error=GetPixelValue(pyr1[2],kp.pt.x/4+x,kp.pt.y/4+y)-GetPixelValue(pyr2[2],kp.pt.x/4+x+dx,kp.pt.y/4+y+dy);
                            }

                            // compute H, b and set cost;
                            H+=J*J.transpose();
                            b+=error*J;
                            cost+=error*error;
                            // TODO END YOUR CODE HERE
                        }

                    // compute update
                    // TODO START YOUR CODE HERE (~1 lines)
                    Eigen::Vector2d update;
                    update=H.ldlt().solve(b);
                    // TODO END YOUR CODE HERE

                    if (isnan(update[0])) {
                        // sometimes occurred when we have a black or white patch and H is irreversible
                        cout << "update is nan" << endl;
                        succ = false;
                        break;
                    }
                    if (iter > 0 && cost > lastCost) {
                        cout << "cost increased: " << cost << ", " << lastCost << endl;
                        for (int iter = 0; iter < iterations; iter++) {
                            Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
                            Eigen::Vector2d b = Eigen::Vector2d::Zero();
                            cost = 0;

                            if (kp.pt.x/2 + dx <= half_patch_size || kp.pt.x/2 + dx >= pyr1[1].cols - half_patch_size ||
                                kp.pt.y/2 + dy <= half_patch_size || kp.pt.y/2 + dy >= pyr1[1].rows - half_patch_size) {   // go outside
                                succ = false;
                                break;
                            }

                            // compute cost and jacobian
                            for (int x = -half_patch_size; x < half_patch_size; x++)
                                for (int y = -half_patch_size; y < half_patch_size; y++) {

                                    // TODO START YOUR CODE HERE (~8 lines)
                                    double error = 0;
                                    Eigen::Vector2d J;  // Jacobian
                                    J<<0,0;

                                    // Forward Jacobian
                                    //判断边界
                                    if (kp.pt.x/2 + x <= half_patch_size || kp.pt.x/2 - x >= pyr1[1].cols - half_patch_size ||
                                        kp.pt.y/2 + y <= half_patch_size || kp.pt.y/2 - y >= pyr1[1].rows - half_patch_size) {
                                        break;
                                    }
                                    else {
                                            //计算雅阁比矩阵和误差
                                        J<<(GetPixelValue(pyr2[1],kp.pt.x/2+x+dx+1,kp.pt.y/2+dy+y)-GetPixelValue(pyr2[1],kp.pt.x/2+x+dx-1,kp.pt.y/2+y+dy))/2,
                                        (GetPixelValue(pyr2[1],kp.pt.x/2+x+dx,kp.pt.y/2+y+dy+1)-GetPixelValue(pyr2[1],kp.pt.x/2+x+dx,kp.pt.y/2+y+dy-1))/2;
                                        error=GetPixelValue(pyr1[1],kp.pt.x/2+x,kp.pt.y/2+y)-GetPixelValue(pyr2[1],kp.pt.x/2+x+dx,kp.pt.y/2+y+dy);
                                    }

                                    // compute H, b and set cost;
                                    H+=J*J.transpose();
                                    b+=error*J;
                                    cost+=error*error;
                                    // TODO END YOUR CODE HERE
                                }

                            // compute update
                            // TODO START YOUR CODE HERE (~1 lines)
                            Eigen::Vector2d update;
                            update=H.ldlt().solve(b);
                            // TODO END YOUR CODE HERE

                            if (isnan(update[0])) {
                                // sometimes occurred when we have a black or white patch and H is irreversible
                                cout << "update is nan" << endl;
                                succ = false;
                                break;
                            }
                            if (iter > 0 && cost > lastCost) {
                                cout << "cost increased: " << cost << ", " << lastCost << endl;
                                for (int iter = 0; iter < iterations; iter++) {
                                    Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
                                    Eigen::Vector2d b = Eigen::Vector2d::Zero();
                                    cost = 0;

                                    if (kp.pt.x + dx <= half_patch_size || kp.pt.x + dx >= pyr1[0].cols - half_patch_size ||
                                        kp.pt.y + dy <= half_patch_size || kp.pt.y + dy >= pyr1[0].rows - half_patch_size) {   // go outside
                                        succ = false;
                                        break;
                                    }

                                    // compute cost and jacobian
                                    for (int x = -half_patch_size; x < half_patch_size; x++)
                                        for (int y = -half_patch_size; y < half_patch_size; y++) {

                                            // TODO START YOUR CODE HERE (~8 lines)
                                            double error = 0;
                                            Eigen::Vector2d J;  // Jacobian
                                            J<<0,0;
                                            // Forward Jacobian
                                            //判断边界
                                            if (kp.pt.x + x <= half_patch_size || kp.pt.x - x >= pyr1[0].cols - half_patch_size ||
                                                kp.pt.y + y <= half_patch_size || kp.pt.y - y >= pyr1[0].rows - half_patch_size) {
                                                break;
                                            }
                                            else {
                                                    //计算雅阁比矩阵和误差
                                                J<<(GetPixelValue(pyr2[0],kp.pt.x+x+dx+1,kp.pt.y+dy+y)-GetPixelValue(pyr2[0],kp.pt.x+x+dx-1,kp.pt.y+y+dy))/2,
                                                 (GetPixelValue(pyr2[0],kp.pt.x+x+dx,kp.pt.y+y+dy+1)-GetPixelValue(pyr2[0],kp.pt.x+x+dx,kp.pt.y+y+dy-1))/2;
                                                error=GetPixelValue(pyr1[0],kp.pt.x+x,kp.pt.y+y)-GetPixelValue(pyr2[0],kp.pt.x+x+dx,kp.pt.y+y+dy);
                                            }


                                            // compute H, b and set cost;
                                            H+=J*J.transpose();
                                            b+=error*J;
                                            cost+=error*error;
                                            // TODO END YOUR CODE HERE
                                        }

                                    // compute update
                                    // TODO START YOUR CODE HERE (~1 lines)
                                    Eigen::Vector2d update;
                                    update=H.ldlt().solve(b);
                                    // TODO END YOUR CODE HERE

                                    if (isnan(update[0])) {
                                        // sometimes occurred when we have a black or white patch and H is irreversible
                                        cout << "update is nan" << endl;
                                        succ = false;
                                        break;
                                    }
                                    if (iter > 0 && cost > lastCost) {
                                        cout << "cost increased: " << cost << ", " << lastCost << endl;
                                        break;
                                    }

                                    // update dx, dy
                                    dx += update[0];
                                    dy += update[1];
                                    lastCost = cost;
                                    succ = true;
                                }

                                break;
                            }

                            // update dx, dy
                            dx += update[0];
                            dy += update[1];
                            lastCost = cost;
                            succ = true;
                        }


                        break;
                    }

                    // update dx, dy
                    dx += update[0];
                    dy += update[1];
                    lastCost = cost;
                    succ = true;
                }
                break;
            }

            // update dx, dy
            dx += update[0];
            dy += update[1];
            lastCost = cost;
            succ = true;
        }

        success.push_back(succ);

        // set kp2
        if (have_initial) {
            kp2[i].pt = kp.pt + Point2f(dx, dy);
        } else {
            KeyPoint tracked = kp;
            tracked.pt += cv::Point2f(dx, dy);
            kp2.push_back(tracked);
        }
    }
    // TODO END YOUR CODE HERE
    // don't forget to set the results into kp2
}
