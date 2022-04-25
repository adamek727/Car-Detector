#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#

void estimate_background(std::vector<cv::Mat>& background_frames, cv::Mat& background) {
    const auto rows = background_frames.at(0).rows;
    const auto cols = background_frames.at(0).cols;
    const auto type = background_frames.at(0).type();
    background = cv::Mat::ones(rows, cols, type) * 255;
    for (int r = 0 ; r < rows ; r++) {
        for (int c = 0 ; c < cols ; c++) {
            std::vector<uint8_t> pixels;
            for (const auto& background_frame : background_frames) {
                pixels.push_back(background_frame.at<uchar>(r,c));
            }
            std::sort(pixels.begin(), pixels.end());
            background.at<uchar>(r,c) = pixels.at(pixels.size()/2);
        }
    }
}

void visualize_images(cv::Mat original, cv::Mat grayscale, cv::Mat background, cv::Mat diff, cv::Mat th) {
    cv::imshow( "original", original );
    cv::imshow( "grayscale", grayscale );
    cv::imshow( "background", background );
    cv::imshow( "diff", diff );
    cv::imshow( "th", th );
    cv::waitKey(1);
}

int main() {
    cv::VideoCapture cap("../resources/road.mp4");
    if(!cap.isOpened()){
        std::cout << "Unable to open video read" << std::endl;
        return -1;
    }

    auto fps = cap.get(CV_CAP_PROP_FPS);
    auto size = cv::Size(cap.get(CV_CAP_PROP_FRAME_WIDTH),cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    int codec = CV_FOURCC('M', 'P', 'E', 'G');

    auto output_th = cv::VideoWriter("../resources/output_th.avi",codec,fps,size);
    auto output_orig = cv::VideoWriter("../resources/output_orig.avi",codec,fps,size);
    auto output_gray = cv::VideoWriter("../resources/output_gray.avi",codec,fps,size);
    auto output_back = cv::VideoWriter("../resources/output_back.avi",codec,fps,size);
    if (!output_th.isOpened() || !output_orig.isOpened() || !output_gray.isOpened() || !output_back.isOpened() ) {
        std::cout << "Unable to open video for write" << std::endl;
        return -2;
    }

    cv::Mat background;
    std::vector<cv::Mat> background_frames;

    while(true) {
        cv::Mat frame, grayscale;
        cap >> frame;
        if (frame.empty())
            break;

        cv::cvtColor(frame, grayscale, CV_BGR2GRAY);

        auto frame_num = static_cast<size_t>(cap.get(CV_CAP_PROP_POS_FRAMES));
        std::cout << "Frame: " << frame_num << std::endl;
        if(frame_num % 100 == 0) {
            background_frames.emplace_back(grayscale.clone());
            if (background_frames.size() > 10) {
                background_frames.erase(background_frames.begin());
            }
            estimate_background(background_frames, background);
        }

        if (background_frames.size() == 10) {
            cv::Mat diff = grayscale - background;
            cv::Mat th;

            cv::threshold(diff, th, 50, 255, CV_THRESH_BINARY);

            cv::Mat o;
            cv::cvtColor(th, o, CV_GRAY2BGR);
            output_th.write(o);
            output_orig.write(frame);
            cv::cvtColor(grayscale, o, CV_GRAY2BGR);
            output_gray.write(o);
            cv::cvtColor(background, o, CV_GRAY2BGR);
            output_back.write(o);

            visualize_images(frame, grayscale, background, diff, th);
        }
    }

    cap.release();
    output_th.release();
    output_gray.release();
    output_back.release();
    return 0;
}
