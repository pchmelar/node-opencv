#include "OpenCV.h"

#if ((CV_MAJOR_VERSION == 2) && (CV_MINOR_VERSION >=4))
#include "Features2d.h"
#include "Matrix.h"
#include <nan.h>
#include <stdio.h>

#ifdef HAVE_OPENCV_FEATURES2D

void Features::Init(Local<Object> target) {
  Nan::HandleScope scope;

  Nan::SetMethod(target, "ImageSimilarity", Similarity);
}

class AsyncDetectSimilarity: public Nan::AsyncWorker {
public:
  AsyncDetectSimilarity(Nan::Callback *callback, cv::Mat image1, cv::Mat image2) :
      Nan::AsyncWorker(callback),
      image1(image1),
      image2(image2),
      score(0) {
  }

  ~AsyncDetectSimilarity() {
  }

  void Execute() {

    cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create("ORB");
    cv::Ptr<cv::DescriptorExtractor> extractor =
        cv::DescriptorExtractor::create("ORB");
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(
        "BruteForce-Hamming");

    std::vector<std::vector<cv::DMatch>> matches;

    cv::Mat descriptors1 = cv::Mat();
    cv::Mat descriptors2 = cv::Mat();

    std::vector<cv::KeyPoint> keypoints1;
    std::vector<cv::KeyPoint> keypoints2;

    detector->detect(image1, keypoints1);
    detector->detect(image2, keypoints2);

    extractor->compute(image1, keypoints1, descriptors1);
    extractor->compute(image2, keypoints2, descriptors2);

    matcher->knnMatch(descriptors1, descriptors2, matches, 2);

    int goodMatches = 0;

    for (unsigned i = 0; i < matches.size(); i++) {
      if (matches[i].size() >= 2 && matches[i][0].distance < 0.8*matches[i][1].distance) {
        goodMatches++;
      }
    }

    score = goodMatches;
  }

  void HandleOKCallback() {
    Nan::HandleScope scope;

    Local<Value> argv[2];

    argv[0] = Nan::Null();
    argv[1] = Nan::New<Number>(score);

    callback->Call(2, argv);
  }

private:
  cv::Mat image1;
  cv::Mat image2;
  int score;
};

NAN_METHOD(Features::Similarity) {
  Nan::HandleScope scope;

  REQ_FUN_ARG(2, cb);

  cv::Mat image1 = Nan::ObjectWrap::Unwrap<Matrix>(info[0]->ToObject())->mat;
  cv::Mat image2 = Nan::ObjectWrap::Unwrap<Matrix>(info[1]->ToObject())->mat;

  Nan::Callback *callback = new Nan::Callback(cb.As<Function>());

  Nan::AsyncQueueWorker( new AsyncDetectSimilarity(callback, image1, image2) );
  return;
}

#endif
#endif
