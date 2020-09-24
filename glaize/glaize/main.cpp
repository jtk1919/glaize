#define _USE_MATH_DEFINES
#include "main.h"

#include <Windows.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <math.h>
#include <stdexcept>
#include <opencv2//opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>


#define WIN "Image"


using namespace std;
using namespace cv;

vector< pair< vector<float>, vector<float> > >  nail_metrics( GL_NUM_FINGERS, pair< vector<float>, vector<float> >() );


const int alpha_slider_max = 100;
int alpha_slider;
double alpha;
double beta;

static void RedirectIOToConsole();
static vector<string> getFiles();
void mouse_callback(int  event, int  x, int  y, int  flag, void* param);
static void processImageScaling();
static size_t distPointToLine(cv::Point l1, cv::Point l2, cv::Point p);
static float angle2Lines(cv::Point l1p1, cv::Point l1p2, cv::Point l2p1, cv::Point l2p2);
static cv::Mat get_finger(size_t idx, size_t xpos);
static void output_csv();

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
size_t cc_length = 0;
double conversion_rate = 1;
string current_fn;


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

        current_fn = fn[i];
        cv::Mat img = cv::imread(fn[i] +"_image.png");
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
        conversion_rate = cc / c;
        
        cv::resize(img, img1, cv::Size(cc, rr));
        cv::resizeWindow( WIN , Size(monitorWidth - 4, monitorHeight - 4));
        cout << cc << " x " << rr << endl;

        mcount = 0;
        img2 = img1.clone();
        img1.copyTo(canvas(Rect(0, 0, img1.cols, img1.rows)));

        cv::putText(canvas, "Mark a short edge of credit card.",
            Size(img1.cols + 6, 40), font, 0.6, Scalar( 0, 255, 0), 1);

        exit_btn = cv::Rect(canvas.cols - 400, 600, 320, 40);
        canvas(exit_btn) = Vec3b(0, 200, 0);
        putText(canvas(exit_btn), "Exit Program",
            Point((int)(exit_btn.width * 0.2), (int)(exit_btn.height * 0.7)), font, 0.8, Scalar(0, 0, 0), 2);

        cv::imshow(WIN, canvas);
        cv::waitKey(0);

        if (!redo)
        {
            cout << "Processing..." << endl;
            canvas = cv::Mat3b(monitorHeight - 4, monitorWidth - 4, cv::Vec3b(0, 0, 0));

            exit_btn = cv::Rect( canvas.cols - 400, 600, 320, 40);
            canvas(exit_btn) = Vec3b(0, 200, 0);
            putText(canvas(exit_btn), "Exit Program",
                Point((int)(exit_btn.width * 0.2), (int)(exit_btn.height * 0.7)), font, 0.8, Scalar(0, 0, 0), 2);

            putText(canvas, "Scale: 85.6 millimeters",
                cv::Point(int((cc - CC_LEN_PX) / 2), 70), font, 0.73, Scalar(0, 255, 0), 1);
            size_t x1, x2;
            x1 = int((cc - CC_LEN_PX) / 2);
            x2 = x1 + CC_LEN_PX;
            cv::line(   canvas, cv::Point( x1 , 100 ) ,  cv::Point( x2, 100 ),  cv::Scalar(0, 255, 0), 2);
            for (size_t i = 0; i < 4; ++i)
            {
                cv::Mat3b lf = get_finger(i, 200 * i + 190 );
                cout << "L Finger: " << i << " size ( r, c ) : ( " << lf.rows << ", " << lf.cols << " )" << endl;
                lf.copyTo(canvas(Rect(200 * i + 190, 400 - lf.rows, lf.cols, lf.rows)));
            }
            output_csv();
            cout << cc_length << endl;
            cv::imshow(WIN, canvas);
            cv::waitKey(0);
        }
    }

    cv:destroyWindow(WIN);
}


void processImageScaling()
{
    size_t dist = 0, d;
    float angleToS1, angleToS2;

    angleToS1 = angle2Lines(le1, le2, se1, se2);
    d = distPointToLine(se1, se2, se3);
    d /= sin(angleToS1 * M_PI / 180);
    dist = d;
    d = distPointToLine(se1, se2, se4);
    d /= sin(angleToS1 * M_PI / 180);
    dist += d;

    angleToS2 = angle2Lines(le1, le2, se3, se4);
    d = distPointToLine(se3, se4, se1);
    d /= sin(angleToS2 * M_PI / 180);
    dist += d;
    d = distPointToLine(se3, se4, se2);
    d /= sin(angleToS2 * M_PI / 180);
    dist += d;

    dist = dist / 4;

    cc_length = dist;
    conversion_rate = conversion_rate * CC_LEN_PX / dist;

    cout << "Angle s1: " << angleToS1 << ",  Angle s2: " << angleToS2 << ", Diatance s1-s2: " << dist << endl;
}


float angle2Lines(cv::Point l1p1, cv::Point l1p2, cv::Point l2p1, cv::Point l2p2)
{
    float l1ToHorizontal, l2ToHorizontal, angle;

    l1ToHorizontal = atan2(l1p1.y - l1p2.y, l1p1.x - l1p2.x) * 180 / M_PI;
    l2ToHorizontal = atan2(l2p1.y - l2p2.y, l2p1.x - l2p2.x) * 180 / M_PI;

    l1ToHorizontal = (l1ToHorizontal < 0) ? 180 + l1ToHorizontal : l1ToHorizontal;
    l2ToHorizontal = (l2ToHorizontal < 0) ? 180 + l2ToHorizontal : l2ToHorizontal;

    //l1ToHorizontal = (l1ToHorizontal > 90.0) ? 180 - l1ToHorizontal : l1ToHorizontal;
    //l2ToHorizontal = (l2ToHorizontal > 90.0) ? 180 - l2ToHorizontal : l2ToHorizontal;

    angle = l1ToHorizontal + l2ToHorizontal;

    cout << "Angle between le-se:"<< angle<< "  le: " << l1ToHorizontal << " se: " << l2ToHorizontal << endl;

    return angle;
}


size_t distPointToLine(cv::Point l1, cv::Point l2, cv::Point p)
{
    long dist, dividend, divisor, y2y1, x2x1;

    y2y1 = l2.y - l1.y;
    x2x1 = l2.x - l1.x;

    dividend = (y2y1 * p.x) - (x2x1 * p.y) + (l2.x * l1.y) - (l2.y * l1.x);
    dividend = abs(dividend);

    divisor = sqrt((y2y1 * y2y1) + (x2x1 * x2x1));

    dist = dividend / divisor;

    return dist;
}



string itos(int i)
{
    stringstream ss;
    ss << i;
    return ss.str();
}

string ftos(double d)
{
    stringstream ss;
    ss << std::fixed;
    ss << std::setprecision(1);
    ss << d;
    return ss.str();
}

vector<Point> denoise(Mat &gray, Mat &img)
{
    vector< vector<Point> > cnts;

    cv::findContours(gray, cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (cnts.size() == 0)
    {
        throw std::domain_error("[fillMask] Failed to find nail contorus.");
    }
    if (cnts.size() > 1)
    {
        sort(cnts.begin(), cnts.end(), [](const vector<Point>& c1, const vector<Point>& c2)
            {
                return contourArea(c1, false) > contourArea(c2, false);
            });
    }

    Mat newgray( gray.size(), gray.type(), Scalar(0) );
    cv::drawContours(newgray, cnts, 0, 255, -1);
    gray.release();
    gray = newgray;

    Mat newcol( img.size(), img.type(), Scalar(0, 0, 0));
    cv::drawContours(newcol, cnts, 0, Scalar(255, 255, 255), -1);
    img.release();
    img = newcol;

    return cnts[0];
}

void printrow( Mat &gray, size_t y)
{
    cout << "Gray size rows: " << gray.rows << " cols: " << gray.cols << endl;
    for (size_t i = 0; i < gray.cols; ++i)
    {
        cout << i << ":" << (int)gray.at<uchar>(y, i) << " ";
    }
    cout << endl;
}

double pxtomm(size_t px)
{
    double mm = px;
    mm /= CC_PX_PER_MM;
    return mm;
}


cv::Mat get_finger(size_t idx, size_t xpos)
{
    size_t w, h, cross_h, cross_v, hmid, x, y, x1=0, x2=0, left, right, step;

    string f = current_fn + "_l" + itos(idx) + ".png";
    cv::Mat img = cv::imread( f );
    w = (int)(img.cols * conversion_rate);
    h = (int)(img.rows * conversion_rate);
    cv::resize(img, img, cv::Size( w, h));

    Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    vector<Point>  cntr = denoise(gray, img);

    step = (int)(CC_PX_PER_MM/2);
    cross_h = (int)(h / step);
    cross_v = (int)(w / step);
    hmid = (int)(w / 2);
    for (size_t i = 0; i < cross_h; ++i )
    {
        x1 = x2 = hmid;
        y = h - step * i - 3 ;
        if (y < 0) break;

        //printrow(gray, y);

        left = 0;
        right = 0;
        for (size_t j = 0; j < hmid; ++j)
        {
            if (((int)gray.at<uchar>(y, j)) > 0)
            {
                x1 = j;
                left = hmid - j;
                break;
            }
        }
        for (size_t j = w-1; j > hmid; --j)
        {
            if (((int)gray.at<uchar>(y, j)) > 0)
            {
                x2 = j;
                right = j - hmid;
                break;
            }
        }
        cv::line(img, Point(x1, y), Point(hmid, y), Scalar(0, 255, 0), 1);
        cv::line(img, Point(hmid, y), Point(x2, y), Scalar(255, 0, 0), 1);

        cout<< x1 << " - " << hmid << " - " << x2 << "    ";

        string metrics;
        metrics += ftos(pxtomm(left)) + " mm | " + ftos(pxtomm(right)) + " mm ";
        cout << metrics << endl;
        size_t H = canvas.rows;
        putText(canvas, metrics.c_str(),
            cv::Point(xpos - 30, H - i * 40 - 80), font, 0.5, Scalar(0, 255, 0), 1);

        nail_metrics[idx].first.push_back(pxtomm(left));
        nail_metrics[idx].second.push_back(pxtomm(right));
    }

    return img;
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
            cv::putText(canvas, "Progressing image. Press any key to proceed",
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
                cv::circle(img1, se1, 3, (0, 255, 255), 2);
                img1.copyTo(canvas(Rect(0, 0, img1.cols, img1.rows)));
                cv::imshow(WIN, canvas);
            }
            else if (mcount == 1)
            {
                se2 = cv::Point(x, y);
                cv::circle(img1, se2, 3, (0, 255, 255), 2);
                cv::line(img1, se1, se2, cv::Scalar(0, 255, 0), 1);
                cv::putText(canvas, "Mark the next short edge.",
                    Size(img1.cols + 6, 80), font, 0.6, Scalar(0, 255, 0), 1);
                img1.copyTo(canvas(Rect(0, 0, img1.cols, img1.rows)));
                cv::imshow(WIN, canvas);
            }
            if (mcount == 2)
            {
                se3 = cv::Point(x, y);
                cv::circle(img1, se3, 3, (0, 255, 255), 2);
                img1.copyTo(canvas(Rect(0, 0, img1.cols, img1.rows)));
                cv::imshow(WIN, canvas);
            }
            else if (mcount == 3)
            {
                se4 = cv::Point(x, y);
                cv::circle(img1, se4, 3, (0, 255, 255), 2);
                cv::line(img1, se3, se4, cv::Scalar(0, 255, 0), 1);
                cv::putText(canvas, "Mark a long edge.",
                    Size(img1.cols + 6, 120), font, 0.6, Scalar(0, 255, 0), 1);
                img1.copyTo(canvas(Rect(0, 0, img1.cols, img1.rows)));
                cv::imshow(WIN, canvas);
            }
            else if (mcount == 4)
            {
                le1 = cv::Point(x, y);
                cv::circle(img1, le1, 3, (0, 255, 255), 2);
                img1.copyTo(canvas(Rect(0, 0, img1.cols, img1.rows)));
                cv::imshow(WIN, canvas);
            }
            else if (mcount == 5)
            {
                le2 = cv::Point(x, y);
                cv::circle(img1, le2, 3, (0, 255, 255), 2);
                cv::line(img1, le1, le2, cv::Scalar(0, 255, 0), 1);

                processImageScaling();

                redo_btn = cv::Rect(img1.cols + 40, 160, 320, 32);
                okay_btn = cv::Rect(img1.cols + 40, 220, 320, 32);
                canvas(redo_btn) = Vec3b(200, 200, 200);
                canvas(okay_btn) = Vec3b(200, 200, 200);
                putText(canvas(redo_btn), "Redo This Image", Point((int)(okay_btn.width * 0.23), (int)(okay_btn.height * 0.7)), font, 0.6, Scalar(0, 0, 0), 2);
                putText(canvas(okay_btn), "Process Image", Point((int)(okay_btn.width * 0.23), (int)(okay_btn.height * 0.7)), font, 0.6, Scalar(0, 0, 0), 2);


                img1.copyTo(canvas(Rect(0, 0, img1.cols, img1.rows)));
                cv::imshow(WIN, canvas);
            }

            ++mcount;
        }
    }
}

void output_csv()
{
    size_t idx1, idx2;
    string csvf, fn;

    idx1 = current_fn.find_last_of("/\\");
    idx2 = current_fn.find_last_of("/\\", idx1 -1);

    fn = current_fn.substr(idx1);
    csvf = current_fn.substr(0, idx2) + "/results/csv/" + fn + ".csv";
    cout << "outputting to: " << csvf;

    ofstream ofs(csvf.c_str(), ios_base::out);
    ofs << std::fixed;
    ofs << std::setprecision(1);
    ofs << "finger ID from left small finger, left half (l) or right half (r), "
        << "half cross section mm at 0.5mm intervals from bottom up... "<< endl;
    for (size_t i = 0; i < GL_NUM_FINGERS; ++i)
    {
        ofs << i << ",l,";
        for (size_t j = 0; j < nail_metrics[i].first.size(); ++j)
        {
            ofs<< nail_metrics[i].first[j] << ",";
        }
        ofs << endl;

        ofs << i << ",r,";
        for (size_t j = 0; j < nail_metrics[i].second.size(); ++j)
        {
            ofs << nail_metrics[i].second[j] << ",";
        }
        ofs << endl;
    }
}

vector<string> getFiles()
{
    vector<string> imgf; // std::string in opencv2.4, but cv::String in 3.0
    string path = string(GL_DATA_WORKING_DIR) + "*_image.png";
    cv::glob(path, imgf, false);

    cout << "Processing "<< imgf.size() << " hand images."<< endl;

    vector<string> fn(imgf.size(), "");
    for (size_t i = 0; i < imgf.size(); ++i)
    {
        string p = imgf[i];
        size_t end = p.find("_image.png");
        fn[i] = p.substr(0, end);
    }
    
    /*if (fn.size() > 0)
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
    }*/
    return fn;
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