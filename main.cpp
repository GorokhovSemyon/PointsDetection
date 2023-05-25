#include <iostream>
#include <opencv2/aruco.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace cv;
using namespace std;

cv::Ptr<cv::FeatureDetector> detector;
cv::Ptr<cv::DescriptorExtractor> extractor;
cv::BFMatcher matcher;

vector<cv::KeyPoint> keys1, keys2;
vector<cv::DMatch> matches;

Mat descr1, descr2, img_matches;
cv::Mat src, src2;

void apply_detector() {
  detector->detect(src, keys1);
  detector->detect(src2, keys2);

  std::cout << "Number of keypoints in image 1: " << keys1.size() << std::endl;
  std::cout << "Number of keypoints in image 2: " << keys2.size() << std::endl;

  extractor->compute(src, keys1, descr1);
  extractor->compute(src2, keys2, descr2);

  matcher.match(descr1, descr2, matches);

  cv::drawMatches(src, keys1, src2, keys2, matches, img_matches);
}

int main(int argc, char *argv[]) {

  cv::CommandLineParser parser(argc, argv,
    "{d | 0 | detector: SIFT=0, SURF=1, BRISK=2}");

  int detectorId = parser.get<int>("d");

  switch (detectorId) {

  case 0:
    detector = cv::SIFT::create();
    extractor = cv::SIFT::create();
    break;
  case 1:
    detector = cv::xfeatures2d::SURF::create();
    extractor = cv::xfeatures2d::SURF::create();
    break;
  case 2:
    detector = cv::BRISK::create();
    extractor = cv::BRISK::create();
    break;

  default:
    cout << "Допустимые значения: 0, 1 или 2 \n";
    return -1;
  }

  cv::VideoCapture cap("../video/sample_mpg.avi");
  if (!cap.isOpened())
    return -1;

  bool stop = false;
  double rate = cap.get(cv::CAP_PROP_FPS);
  int delay = 1000 / rate;

  cout << "Frame rate of video is " << rate << endl;
  while (!stop) {

    bool result = cap.read(src);

    if (src.empty())
      break;

    src.copyTo(src2);

    apply_detector();

    cv::imshow("Original", src);
    cv::imshow("matches", img_matches);

    // Ждём нажатия на кнопку
    int key = cv::waitKey(delay);
    if (key == 'q') // кнопка "q"
      stop = true; // завершение
  }

  return 0;
}