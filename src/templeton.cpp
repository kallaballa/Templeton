#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/photo/photo.hpp>

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>

#include <iostream>
#include <fstream>
#include <limits>
#include <vector>
#include <cassert>
#include <getopt.h>

using namespace cv;
using namespace std;
namespace d = dlib;

struct Features {
  vector<Point> chin;
  vector<Point> top_nose;
  vector<Point> bottom_nose;
  vector<Point> left_eyebrow;
  vector<Point> right_eyebrow;
  vector<Point> left_eye;
  vector<Point> right_eye;
  vector<Point> outer_lips;
  vector<Point> inside_lips;
};


Features detectFeatures(const string& filename) {
  Features features;
  d::shape_predictor sp;
  d::deserialize("shape_predictor_68_face_landmarks.dat") >> sp;

  d::frontal_face_detector detector = d::get_frontal_face_detector();

  d::array2d<d::rgb_pixel> img;
  d::load_image(img, filename);
  // Make the image larger so we can detect small faces.
  size_t orig_rows = img.nr();
  size_t orig_columns = img.nc();

  d::pyramid_up(img);

  size_t scaled_rows = img.nr();
  size_t scaled_columns = img.nc();

  double rf = (double)scaled_rows / (double)orig_rows;
  double cf = (double)scaled_columns / (double)orig_columns;

  // Now tell the face detector to give us a list of bounding boxes
  // around all the faces in the image.
  std::vector<dlib::rectangle> dets = detector(img);
  cout << "Number of faces detected: " << dets.size() << endl;

  if(dets.empty())
    return  {{},{}};

  // Now we will go ask the shape_predictor to tell us the pose of
  // each face we detected.
  std::vector<d::full_object_detection> shapes;

  d::full_object_detection shape = sp(img, dets[0]);
  Point2f nose_bottom(0,0);
  Point2f lips_top(0,std::numeric_limits<float>().max());
  cout << "number of parts: " << shape.num_parts() << endl;


  // Around Chin. Ear to Ear
  for (unsigned long i = 1; i <= 16; ++i)
    features.chin.push_back(Point2f(shape.part(i).x()/rf, shape.part(i).y()/cf));

  // Line on top of nose

  for (unsigned long i = 28; i <= 30; ++i)
     features.top_nose.push_back(Point2f(shape.part(i).x()/rf, shape.part(i).y()/cf));

  // left eyebrow
  for (unsigned long i = 18; i <= 21; ++i)
      features.left_eyebrow.push_back(Point2f(shape.part(i).x()/rf, shape.part(i).y()/cf));

  // Right eyebrow
  for (unsigned long i = 23; i <= 26; ++i)
      features.right_eyebrow.push_back(Point2f(shape.part(i).x()/rf, shape.part(i).y()/cf));

  // Bottom part of the nose
  for (unsigned long i = 31; i <= 35; ++i)
      features.bottom_nose.push_back(Point2f(shape.part(i).x()/rf, shape.part(i).y()/cf));

  // Left eye
  for (unsigned long i = 37; i <= 41; ++i)
      features.left_eye.push_back(Point2f(shape.part(i).x()/rf, shape.part(i).y()/cf));

  // Right eye
  for (unsigned long i = 43; i <= 47; ++i)
    features.right_eye.push_back(Point2f(shape.part(i).x()/rf, shape.part(i).y()/cf));

  // Lips outer part
  for (unsigned long i = 49; i <= 59; ++i)
    features.outer_lips.push_back(Point2f(shape.part(i).x()/rf, shape.part(i).y()/cf));

  // Lips inside part
  for (unsigned long i = 61; i <= 67; ++i)
    features.inside_lips.push_back(Point2f(shape.part(i).x()/rf, shape.part(i).y()/cf));

  return features;
}

void makeWhiteTransparent(Mat& m) {
  // find all white pixel and set alpha value to zero:
  for (int y = 0; y < m.rows; ++y)
  for (int x = 0; x < m.cols; ++x)
  {
      cv::Vec4b & pixel = m.at<cv::Vec4b>(y, x);
      // if pixel is white
      if (pixel[0] == 255 && pixel[1] == 255 && pixel[2] == 255)
      {
          // set alpha to zero:
          pixel[3] = 0;
      }
  }
}

void overlayImage(Mat* src, Mat* overlay, const Point& location)
{
    for (int y = max(location.y, 0); y < src->rows; ++y)
    {
        int fY = y - location.y;

        if (fY >= overlay->rows)
            break;

        for (int x = max(location.x, 0); x < src->cols; ++x)
        {
            int fX = x - location.x;

            if (fX >= overlay->cols)
                break;

            double opacity = ((double)overlay->data[fY * overlay->step + fX * overlay->channels() + 3]) / 255;

            for (int c = 0; opacity > 0 && c < src->channels(); ++c)
            {
                unsigned char overlayPx = overlay->data[fY * overlay->step + fX * overlay->channels() + c];
                unsigned char srcPx = src->data[y * src->step + x * src->channels() + c];
                src->data[y * src->step + src->channels() * x + c] = srcPx * (1. - opacity) + overlayPx * opacity;
            }
        }
    }
}

void mustacheOverlay(string faceFile, string mustacheFile, string outputFile) {
  //Read input images
  Mat faceImg = imread(faceFile);
  Mat mustacheImg = imread(mustacheFile);

  //convert Mat to float data type
//  faceImg.convertTo(faceImg, CV_32FC3);
//  mustacheImg.convertTo(mustacheImg, CV_32FC3);
  cv::cvtColor(mustacheImg, mustacheImg, CV_BGR2BGRA);
  cv::cvtColor(faceImg, faceImg, CV_BGR2BGRA);

  //Read points
  Features features = detectFeatures(faceFile);
  Point2f nose_bottom(0,0);
  Point2f lips_top(0,std::numeric_limits<float>().max());
  Point2f chinLeft(std::numeric_limits<float>().max(),0);
  Point2f chinRight;

  for(Point2f p : features.bottom_nose) {
    if(p.y > nose_bottom.y)
      nose_bottom = p;
  }

  for(Point2f p : features.outer_lips) {
    if(p.y < lips_top.y)
      lips_top = p;
  }

  for(Point2f p : features.chin) {
    if(p.x > chinRight.x)
      chinRight = p;

    if(p.x < chinLeft.x)
      chinLeft = p;
  }

  float mustachHeight = lips_top.y - nose_bottom.y;


  float faceWidth = chinRight.x - chinLeft.x;

  Mat mustacheScaled;
  Size msize1 = mustacheImg.size();
  Size msize2 = mustacheImg.size();
  Size msize;

  float factor = msize1.width / faceWidth;
  msize1.width = faceWidth;
  msize1.height = msize1.height / factor;

  float factor2 = msize2.height / mustachHeight;
  msize2.width = (msize2.width / factor2) * 5;
  msize2.height = mustachHeight * 5;

  msize = msize1.width < msize2.width ? msize1 : msize2;

  resize(mustacheImg,mustacheScaled, msize);
  makeWhiteTransparent(mustacheScaled);

  Point2f to(nose_bottom.x - (msize.width / 2), nose_bottom.y - (msize.height / 10));
  overlayImage(&faceImg, &mustacheScaled, cv::Point(to.x, to.y));
  //  mustacheScaled.copyTo(faceImg.colRange(to.x, mustacheScaled.size().width + to.x).rowRange(to.y, mustacheScaled.size().height + to.y));

  faceImg.convertTo(faceImg, CV_8UC3);

  imwrite(outputFile, faceImg);
}

void glassesOverlay(string faceFile, string glassesFile, string outputFile) {
  //Read input images
  Mat faceImg = imread(faceFile);
  Mat glassesImg = imread(glassesFile);

  //convert Mat to float data type
//  faceImg.convertTo(faceImg, CV_32FC3);
//  mustacheImg.convertTo(mustacheImg, CV_32FC3);
  cv::cvtColor(glassesImg, glassesImg, CV_BGR2BGRA);
  cv::cvtColor(faceImg, faceImg, CV_BGR2BGRA);

  //Read points
  //Read points
  Features features = detectFeatures(faceFile);
  Point2f nose_top(0,std::numeric_limits<float>().max());
  Point2f chinLeft(std::numeric_limits<float>().max(),0);
  Point2f chinRight;

  for(Point2f p : features.chin) {
    if(p.x > chinRight.x)
      chinRight = p;

    if(p.x < chinLeft.x)
      chinLeft = p;
  }

  for(Point2f p : features.top_nose) {
    if(p.y < nose_top.y) {
      nose_top = p;
    }
  }

  float faceWidth = chinRight.x - chinLeft.x;

  Mat glassesScaled;
  Size gsize = glassesImg.size();

  float factor = gsize.width / faceWidth;
  gsize.width = faceWidth;
  gsize.height = gsize.height / factor;

  resize(glassesImg,glassesScaled, gsize);
  makeWhiteTransparent(glassesScaled);

  Point2f to(nose_top.x - (gsize.width / 2), nose_top.y - (gsize.height/2));
  overlayImage(&faceImg, &glassesScaled, cv::Point(to.x, to.y));

  faceImg.convertTo(faceImg, CV_8UC3);

  imwrite(outputFile, faceImg);
}

void printUsage() {
  std::cerr << "Usage: breezy [-m][-g] <face image> <overlay image> <output image>" << std::endl;
  std::cerr << "Options:" << std::endl;
  std::cerr << "\t-m\t\trender a mustache on top of the image" << std::endl;
  std::cerr << "\t-g\t\trender glasses on top of the image" << std::endl;
  exit(1);
}

int main( int argc, char** argv) {
  bool mustacheOpt = false;
  bool glassesOpt = false;

  const struct option longopts[] =
  {
    {"mustache",   no_argument,        0, 'm'},
    {"glasses",      no_argument,        0, 'g'},
    {0,0,0,0},
  };

  int index;
  int iarg=0;

  //turn off getopt error message
  opterr=1;

  while (iarg != -1) {
    iarg = getopt_long(argc, argv, "gm", longopts, &index);

    switch (iarg) {
    case 'm':
      mustacheOpt = true;
      break;

    case 'g':
      glassesOpt = true;
      break;
    case 'h':
      printUsage();
      return 1;
      break;
    case ':':
      printUsage();
      return 1;
      break;
    case '?':
      printUsage();
      return 1;
      break;

    }
  }
  if(mustacheOpt && glassesOpt) {
    std::cerr << "You can use either -g or -m" << std::endl;
    exit(1);
  }
  if(argc - optind != 3) {
    printUsage();
  }
  string faceFile(argv[optind]);
  string overlayFile(argv[optind+1]);
  string outputFile(argv[optind+2]);


  if(mustacheOpt)
    mustacheOverlay(faceFile, overlayFile, outputFile);
  else if(glassesOpt)
    glassesOverlay(faceFile,overlayFile, outputFile);

  return 0;
}
