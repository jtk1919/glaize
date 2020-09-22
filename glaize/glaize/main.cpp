#define _USE_MATH_DEFINES
#include "main.h"

#include <Windows.h>
#include <iostream>
#include <vector>
#include <set>
#include <math.h>
#include <opencv2//opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>


#define WIN "Image"


using namespace std;
using namespace cv;


const int alpha_slider_max = 100;
int alpha_slider;
double alpha;
double beta;


static void RedirectIOToConsole();
static vector<string> getFiles();
static void on_trackbar(int, void*);
void mouse_callback(int  event, int  x, int  y, int  flag, void* param);
static void processImageScaling();
static size_t distPointToLine(cv::Point l1, cv::Point l2, cv::Point p);
static size_t angle2Lines(cv::Point l1p1, cv::Point l1p2, cv::Point l2p1, cv::Point l2p2);

int font = cv::FONT_HERSHEY_COMPLEX;
cv::Scalar blue = cv::Scalar(0, 0, 255);

cv::Point pt(-1, -1);
bool newCoords = false;

cv::Mat3b canvas;
cv::Rect redo_btn, okay_btn, exit_btn;
boolean redo = false;

Mat img1, img2;
int mcount = 0;
cv::Point se1, se2, se3, se4, le1, le2;

int main(size_t monitorHeight, size_t monitorWidth )
{
    RedirectIOToConsole();

    cout << "Screen size: " << monitorWidth << " x " <<  monitorHeight  << endl;

    vector<string> fn = getFiles();
	cout << "Read " << fn.size() << " files" << endl;

    cv::namedWindow( WIN, cv::WINDOW_GUI_NORMAL);
    cv::moveWindow(WIN, 0, 0);

    for (size_t i = 0; i < fn.size(); ++i)
    {
        if (redo)
        {
            --i;
            redo = false;
        }
        setMouseCallback(WIN, mouse_callback);
        canvas = cv::Mat3b(monitorHeight - 4, monitorWidth - 4, cv::Vec3b(0, 0, 0));

        cv::Mat img = cv::imread(fn[i]);
        int r = img.rows;
        int c = img.cols;
        if ( r > c )
        {
            cv::rotate( img, img, cv::ROTATE_90_COUNTERCLOCKWISE);
            r = c;
            c = img.cols;
        }
        cout << "Image size: " << c << " x " << r << " | ";

        int rr = monitorHeight - 32;
        int cc = c * ( monitorHeight - 32) / r;
        if (cc > ( monitorWidth - 400))
        {
            cc = monitorWidth - 400;
            rr = r * (monitorWidth - 400) / c;
        }
        
        cv::resize(img, img1, cv::Size(cc, rr));
        cv::resizeWindow( WIN , Size(cc, rr));
        cout << cc << " x " << rr << endl;

        cv::imwrite(string(GL_RCNN_PATH) + "img.jpg", img);
        cv::imwrite(string(GL_RCNN_PATH) + "img1.jpg", img1);
        
        mcount = 0;
        img2 = img1.clone();
        img1.copyTo(canvas(Rect(0, 0, img1.cols, img1.rows)));

        cv::putText(canvas, "Mark a short edge of credit card.",
            Size(img1.cols + 6, 40), font, 0.6, Scalar( 0, 255, 0), 1);

        exit_btn = cv::Rect(img1.cols + 40, 600, 320, 40);
        canvas(exit_btn) = Vec3b(0, 200, 0);
        putText(canvas(exit_btn), "Exit Program",
            Point((int)(exit_btn.width * 0.2), (int)(exit_btn.height * 0.7)), font, 0.8, Scalar(0, 0, 0), 2);


        cv::imshow(WIN, canvas);
        cv::waitKey(0);
    }

    cv:destroyWindow(WIN);
}

void processImageScaling()
{
    size_t dist, angleToS1, angleToS2;

    dist = distPointToLine(se1, se2, se3) + distPointToLine(se1, se2, se4)
         + distPointToLine(se3, se4, se1) + distPointToLine(se3, se4, se2);
    dist = dist / 4;

    angleToS1 = angle2Lines(le1, le2, se1, se2);
    angleToS2 = angle2Lines(le1, le2, se3, se4);

    cout << "Angle s1: " << angleToS1 << ",  Angle s2: " << angleToS2 << ", Diatance s1-s2: " << dist << endl;
}

size_t angle2Lines(cv::Point l1p1, cv::Point l1p2, cv::Point l2p1, cv::Point l2p2)
{
    float l1ToHorizontal, l2ToHorizontal, angle;

    l1ToHorizontal = atan2(l1p1.y - l1p2.y, l1p1.x - l1p2.x) * 180 / M_PI;
    l2ToHorizontal = atan2(l2p1.y - l2p2.y, l2p1.x - l2p2.x) * 180 / M_PI;

    l1ToHorizontal = (l1ToHorizontal < 0) ? 180 + l1ToHorizontal : l1ToHorizontal;
    l2ToHorizontal = (l2ToHorizontal < 0) ? 180 + l2ToHorizontal : l2ToHorizontal;

    l1ToHorizontal = (l1ToHorizontal > 90.0) ? 180 - l1ToHorizontal : l1ToHorizontal;
    l2ToHorizontal = (l2ToHorizontal > 90.0) ? 180 - l2ToHorizontal : l2ToHorizontal;

    cout << "Angles le: " << l1ToHorizontal << " se: " << l2ToHorizontal << endl;

   
    angle = l1ToHorizontal + l2ToHorizontal;
    return angle;
}

size_t distPointToLine(cv::Point l1, cv::Point l2, cv::Point p)
{
    long dist, dividend, divisor, y2y1, x2x1;

    y2y1 = l2.y - l1.y;
    x2x1 = l2.x - l1.x;

    dividend = (y2y1 * p.x) - (x2x1 * p.y) + (l2.x * l1.y) - (l2.y * l1.x);
    dividend = abs(dividend);

    divisor = sqrt((y2y1 * y2y1) + (x2x1 * x2x1) );

    dist = dividend / divisor;

    return dist;
}

void mouse_callback(int  event, int  x, int  y, int  flag, void* param)
{
    if (event == EVENT_LBUTTONDOWN)
    {
        if (exit_btn.contains(Point(x, y)))
        {
            cv::destroyAllWindows();
            exit(0);
        }
        else if (redo_btn.contains(Point(x, y)))
        {
            redo = true;
            mcount = 0;
            cv::Rect rc = cv::Rect(280, 350, 1000, 120);
            canvas(rc) = Vec3b(0, 0, 0);
            cv::putText(canvas, "Re-marking this image. Press any key to proceed",
                Size(300, 400), font, 1, Scalar(0, 255, 0), 1);
            cv::imshow(WIN, canvas);
        }
        else if ( okay_btn.contains(Point(x, y)))
        {
            mcount = 0;
            cv::Rect rc = cv::Rect(280, 350, 1000, 120);
            canvas(rc) = Vec3b(0, 0, 0);
            cv::putText(canvas, "Progressing to next image. Press any key to proceed",
                Size(300, 400), font, 1, Scalar(0, 255, 0), 1);
            cv::imshow(WIN, canvas);
        }
        else
        {
            // Store point coordinates
            pt.x = x;
            pt.y = y;
            cout << "Point ( " << pt.x << ", " << pt.y << ")" << endl;

            if (mcount == 0)
            {
                se1 = cv::Point(x, y);
            }
            else if (mcount == 1)
            {
                se2 = cv::Point(x, y);
                cv::line(img1, se1, se2, cv::Scalar(0, 255, 0), 1);
                cv::putText(canvas, "Mark the next short edge.",
                    Size(img1.cols + 6, 80), font, 0.6, Scalar(0, 255, 0), 1);
                img1.copyTo(canvas(Rect(0, 0, img1.cols, img1.rows)));
                cv::imshow(WIN, canvas);
            }
            if (mcount == 2)
            {
                se3 = cv::Point(x, y);
            }
            else if (mcount == 3)
            {
                se4 = cv::Point(x, y);
                cv::line(img1, se3, se4, cv::Scalar(0, 255, 0), 1);
                cv::putText(canvas, "Mark a long edge.",
                    Size(img1.cols + 6, 120), font, 0.6, Scalar(0, 255, 0), 1);
                img1.copyTo(canvas(Rect(0, 0, img1.cols, img1.rows)));
                cv::imshow(WIN, canvas);
            }
            else if (mcount == 4)
            {
                le1 = cv::Point(x, y);
            }
            else if (mcount == 5)
            {
                le2 = cv::Point(x, y);
                cv::line(img1, le1, le2, cv::Scalar(0, 255, 0), 1);

                processImageScaling();

                redo_btn = cv::Rect(img1.cols + 40, 160, 320, 32);
                okay_btn = cv::Rect(img1.cols + 40, 220, 320, 32);
                canvas(redo_btn) = Vec3b(200, 200, 200);
                canvas(okay_btn) = Vec3b(200, 200, 200);
                putText(canvas(redo_btn), "Redo This Image", Point((int)(okay_btn.width * 0.23), (int)(okay_btn.height * 0.7)), font, 0.6, Scalar(0, 0, 0), 2);
                putText(canvas(okay_btn), "Mark Next Image", Point((int)(okay_btn.width * 0.23), (int)(okay_btn.height * 0.7)), font, 0.6, Scalar(0, 0, 0), 2);


                img1.copyTo(canvas(Rect(0, 0, img1.cols, img1.rows)));
                cv::imshow(WIN, canvas);
            }

            ++mcount;
        }
    }
}

vector<string> getFiles()
{
    vector<string> fn, imgf; // std::string in opencv2.4, but cv::String in 3.0
    string path = string(GL_DATA_PATH_LEFT_F) + "*";
    cv::glob(path, fn, false);

    cout << fn.size() << endl;
    
    if (fn.size() > 0)
    {
        for (size_t i = 0; i < fn.size(); ++i)
        {
            size_t dot_i = fn[i].find_last_of('.');
            if (dot_i != string::npos)
            {
                string ext = fn[i].substr(dot_i);
                if (ext_set.find(ext) != ext_set.end())
                {
                    imgf.push_back(fn[i]);
                }
            }
        }
    }
    return imgf;
}


void RedirectIOToConsole() 
{

    //Create a console for this application
    AllocConsole();

    // Get STDOUT handle
    HANDLE ConsoleOutput = GetStdHandle(STD_OUTPUT_HANDLE);
    int SystemOutput = _open_osfhandle(intptr_t(ConsoleOutput), _O_TEXT);
    FILE* COutputHandle = _fdopen(SystemOutput, "w");

    // Get STDERR handle
    HANDLE ConsoleError = GetStdHandle(STD_ERROR_HANDLE);
    int SystemError = _open_osfhandle(intptr_t(ConsoleError), _O_TEXT);
    FILE* CErrorHandle = _fdopen(SystemError, "w");

    // Get STDIN handle
    HANDLE ConsoleInput = GetStdHandle(STD_INPUT_HANDLE);
    int SystemInput = _open_osfhandle(intptr_t(ConsoleInput), _O_TEXT);
    FILE* CInputHandle = _fdopen(SystemInput, "r");

    //make cout, wcout, cin, wcin, wcerr, cerr, wclog and clog point to console as well
    ios::sync_with_stdio(true);

    // Redirect the CRT standard input, output, and error handles to the console
    freopen_s(&CInputHandle, "CONIN$", "r", stdin);
    freopen_s(&COutputHandle, "CONOUT$", "w", stdout);
    freopen_s(&CErrorHandle, "CONOUT$", "w", stderr);

    //Clear the error state for each of the C++ standard stream objects. We need to do this, as
    //attempts to access the standard streams before they refer to a valid target will cause the
    //iostream objects to enter an error state. In versions of Visual Studio after 2005, this seems
    //to always occur during startup regardless of whether anything has been read from or written to
    //the console or not.
    std::wcout.clear();
    std::cout.clear();
    std::wcerr.clear();
    std::cerr.clear();
    std::wcin.clear();
    std::cin.clear();

}