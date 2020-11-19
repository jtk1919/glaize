#define _USE_MATH_DEFINES
#include "main.h"

#include <Windows.h>
#include <tchar.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <math.h>
#include <stdexcept>
#include <opencv2//opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "config.h"
#include "ImageIO.h"


#define WIN_DEF "Image"

using namespace std;
using namespace cv;

enum IMAGE_STATE
{
    STARTING,
    MARKING_CC_FIN,
    REDO_OKAY_FIN,
    MARKING_CC_THUMB,
    REDO_OKAY_THUMB,
    STRAIGHTENING,
    COMPOSING,
};

IMAGE_STATE state = STARTING;

void reset();
static void startScreen(size_t monitorHeight, size_t monitorWidth);
static vector<string> getFiles();
void mouse_callback(int  event, int  x, int  y, int  flag, void* param);
static void processImageScaling();
static size_t distPointToLine(cv::Point l1, cv::Point l2, cv::Point p);
static float angle2Lines(cv::Point l1p1, cv::Point l1p2, cv::Point l2p1, cv::Point l2p2);
static cv::Mat get_finger(size_t idx, size_t xpos);
static void clear_metrics();
static void compose();

size_t pixels_per_mm;
vector< pair< vector<float>, vector<float> > >  nail_metrics( GL_NUM_FINGERS, pair< vector<float>, vector<float> >() );

int font = cv::FONT_HERSHEY_COMPLEX;
cv::Scalar blue = cv::Scalar(0, 0, 255);

cv::Point pt(-1, -1);
bool newCoords = false;

cv::Mat3b canvas;
cv::Rect redo_btn, okay_btn, next_hand_btn, exit_btn;
boolean redo = false;
bool genmetrics = false;

Mat img1, img2;
int mcount = 0;
cv::Point se1, se2, se3, se4, le1, le2;

ImageIO imgFiles;
size_t current_finger_image_idx = 0;
size_t cc_length[] = { 0, 0 };
double conversion_rate[] = { 1.0, 1.0 };
int turn_angle[] = { 0, 0, 0, 0, 0 };

bool process_next = true;
size_t cc = 0;

Mat fingers[5];
Mat save[5];
Rect fingerLoc[5];

size_t monitorHeight, monitorWidth;
wstring pydir;
string svgdir = string(GL_DATA_DIR) + GL_SVG_PATH;


int main(size_t monitorH, size_t monitorW, wstring pyd)
{
    monitorHeight = monitorH;
    monitorWidth = monitorW;
    pydir = pyd;
    cout << "Screen size: " << monitorWidth << " x " <<  monitorHeight  << endl;

    //vector<string> fn = getFiles();
	//cout << "Read " << fn.size() << " files" << endl;

    cv::Mat img;
    string fn;

    bool done = false;
    for (size_t i = 0; !done; ++i)
    {
        state = STARTING;
        if (done) break;
        ++cc;
        clear_metrics();
        redo = false;
        genmetrics = true;

        for (size_t j = 0; j < 2; redo ? redo = true : ++j)
        {         
            state = (j == 0) ? MARKING_CC_FIN : MARKING_CC_THUMB;
            if (!redo)
            {
                current_finger_image_idx = j;
                if (j == 0)
                {
                    fn = imgFiles.getLeftFingerF();
                    cout << "\nCross sections for: " << fn << endl;
                }
                else
                {
                    fn = imgFiles.getLeftThumbF();
                }
                if (fn.empty() )
                {
                    done = true;
                    break;
                }
                img = cv::imread(fn);
                redo = false;
            }

            if (!process_next)
            {
                int fid = ImageIO::last_run_file_id(fn);
                process_next = (j == 1) && (fid == cfg.last_run_file_id);
                if (process_next)
                {
                    cc = 0;
                }
                cout << "skipping file: " << fn << endl;
                continue;  // skip this one and start running for the next
            }

            cv::namedWindow(WIN_DEF, cv::WINDOW_GUI_NORMAL);
            cv::moveWindow(WIN_DEF, 0, 0);
            setMouseCallback(WIN_DEF, mouse_callback);

            canvas = cv::Mat3b(monitorHeight - 4, monitorWidth - 4, cv::Vec3b(0, 0, 0));

            int r = img.rows;
            int c = img.cols;
            if (r > c)
            {
                cv::rotate(img, img, cv::ROTATE_90_COUNTERCLOCKWISE);
                r = c;
                c = img.cols;
            }
            cout << "Image size: " << c << " x " << r << " | ";

            int rr = monitorHeight - 32;
            int cc = c * (monitorHeight - 32);
            cc /= r;
            if (cc > (monitorWidth - 400))
            {
                cc = monitorWidth - 400;
                rr = r * (monitorWidth - 400);
                rr /= c;
            }
            conversion_rate[j] = cc;
            conversion_rate[j] /= c;

            cv::resize(img, img1, cv::Size(cc, rr));
            cv::resizeWindow(WIN_DEF, Size(monitorWidth - 4, monitorHeight - 4));
            cout << cc << " x " << rr << endl;

            mcount = 0;
            img2 = img1.clone();
            img1.copyTo(canvas(Rect(0, 0, img1.cols, img1.rows)));

            cv::putText(canvas, "Mark a short edge of credit card.",
                Size(img1.cols + 6, 40), font, 0.7, Scalar(0, 255, 0), 1);

            exit_btn = cv::Rect(canvas.cols - 400, 600, 320, 40);
            canvas(exit_btn) = Vec3b(0, 200, 0);
            putText(canvas(exit_btn), "Exit Application",
                Point((int)(exit_btn.width * 0.17), (int)(exit_btn.height * 0.7)), font, 0.8, Scalar(0, 0, 0), 2);

            cv::imshow(WIN_DEF, canvas);
            cv::waitKey(0);

        }

        if (process_next && ( cc > 0) )
        {

        }
        else
        {
            continue;
        }
        

        if (!redo  && !done )
        {
            cout << "Processing . . ." << endl;
            canvas = cv::Mat3b(monitorHeight - 4, monitorWidth - 4, cv::Vec3b(0, 0, 0));

            state = STRAIGHTENING;
            cv::putText(canvas, "Straignten finger nails and click \"Proceed.\"",
                Size(canvas.cols - 600, 40), font, 0.7, Scalar(0, 255, 0), 1);

            next_hand_btn = cv::Rect(canvas.cols - 400, 500, 320, 40);
            canvas(next_hand_btn) = Vec3b(0, 200, 0);
            putText(canvas(next_hand_btn), "Proceed",
                Point((int)(next_hand_btn.width * 0.3), (int)(next_hand_btn.height * 0.7)), font, 0.8, Scalar(0, 0, 0), 2);

            exit_btn = cv::Rect( canvas.cols - 400, 600, 320, 40);
            canvas(exit_btn) = Vec3b(0, 200, 0);
            putText(canvas(exit_btn), "Exit Application",
                Point((int)(exit_btn.width * 0.17), (int)(exit_btn.height * 0.7)), font, 0.8, Scalar(0, 0, 0), 2);

            putText(canvas, "Scale: 85.6 millimeters",
                cv::Point(int(( monitorWidth - 400 - CC_LEN_PX) / 2), 70), font, 0.73, Scalar(0, 255, 0), 1);
            size_t x1, x2;
            x1 = int((monitorWidth - 400 - CC_LEN_PX) / 2);
            x2 = x1 + CC_LEN_PX;
            cv::line(   canvas, cv::Point( x1 , 100 ) ,  cv::Point( x2, 100 ),  cv::Scalar(0, 255, 0), 2);

            for (size_t k = 0; k < 5; ++k)
            {
                cv::Mat3b lf = get_finger(k, 200 * k + 190 );
                //cv::imshow("finger", lf);  cv::waitKey(0);
                cout << "L Finger: " << k << " size ( r, c ) : ( " << lf.rows << ", " << lf.cols << " )" << endl;
                fingerLoc[k] = Rect(200 * k + 190, 400 - lf.rows, lf.cols, lf.rows);
                lf.copyTo(canvas( fingerLoc[k] ));
                mcount = -1;
            }

            int xhand = 200 * 5 + 260 + 300;
            float tmp = monitorHeight;
            tmp /= 20;
            int sec = (int)tmp;
            int yhand = 2 * sec;
            int h = 8 * sec;
            string f = imgFiles.getFingerSeg();
            Mat img = cv::imread(f);
            int w = (int)(h * img.cols / img.rows);
            cv::resize(img, img, Size(w, h));
            img.copyTo( canvas( Rect(xhand, yhand, w, h) ) );

            yhand = 11 * sec;
            f = imgFiles.getThumbSeg();
            img = cv::imread(f);
            w = (int)(h * img.cols / img.rows);
            cv::resize(img, img, Size(w, h));
            img.copyTo(canvas(Rect(xhand, yhand, w, h)));

            cv::imshow(WIN_DEF, canvas);
            cv::waitKey(0);


            state = COMPOSING;
        }
    }

    cv:destroyWindow(WIN_DEF);
}


void fakeNailMapping()
{
    int w, h, x;
    STARTUPINFO si = { sizeof(STARTUPINFO) };
    PROCESS_INFORMATION pi;

    string csv = imgFiles.getCsvFile();
    string tmp = imgFiles.getTempFile();
    string cm = string("/C python model_rec.py --image \"") + csv + "\" > \"" + tmp + "\"";
    cout << "Running fake nail classification.";
    cout << "  " << cm << endl;

    wstring cmds( cm.begin(), cm.end());
    LPTSTR cmd = _tcsdup(cmds.c_str());
    if (!CreateProcess(L"C:\\Windows\\System32\\cmd.exe",
        cmd,
        NULL, NULL,
        0, 0,
        NULL,
        pydir.c_str(),
        &si, &pi))
    {
        printf("CreateProcess failed (%d) for: %s\n", GetLastError(), cm);
        return;
    }
    WaitForSingleObject(pi.hProcess, INFINITE);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);

    ifstream fs(tmp);
    int combi;
    fs >> combi;
    fs.close();
    cout << "Hand classified to combi " << combi + 1;

    string msg = string("Hand classified to pres-on combi size ") + ImageIO::itos(combi + 1);
    cv::putText(canvas, msg.c_str(),
        Size(canvas.cols - 600, 80), font, 0.7, Scalar(0, 255, 0), 1);

    cm = string("/C python compose.py --image \"") + csv + "\"";
    cout << "Running customised composition.";
    cout << "  " << cm << endl;
    wstring cmds1(cm.begin(), cm.end());
    cmd = _tcsdup(cmds1.c_str());
    if (!CreateProcess(L"C:\\Windows\\System32\\cmd.exe",
        cmd,
        NULL, NULL,
        0, 0,
        NULL,
        pydir.c_str(),
        &si, &pi))
    {
        printf("CreateProcess failed (%d) for: %s\n", GetLastError(), cm);
        return;
    }
    WaitForSingleObject(pi.hProcess, INFINITE);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);


    std::pair< cv::Mat, cv::Mat> set;
    svgdir = imgFiles.getComposedSet(set);
    Rect r1 = Rect( 82, 400 - set.first.rows, set.first.cols, set.first.rows);
    set.first.copyTo(canvas(r1));
    r1 = Rect( 82 + set.first.cols + 28, 400 - set.first.rows, set.second.cols, set.second.rows);
    set.second.copyTo(canvas(r1));
    set.first.release();
    set.second.release();


    cv::Mat fin3d[5], fake[5];
    size_t cc[2];
    int rc;

    rc = imgFiles.getFin3d(combi, fin3d, fake, cc);

    for (size_t k = 0; k < 5; ++k)
    {
        size_t ci = (k == 4) ? 1 : 0;
        double cr = CC_LEN_PX;
        cr /= cc[ci];

        cv::Mat lf1 = fingers[k];
        x = 340 * k + 300 - round(lf1.cols / 2);
        fingerLoc[k] = Rect( x, 700 - lf1.rows, lf1.cols, lf1.rows);
        lf1.copyTo(canvas(fingerLoc[k]));

        cv::Mat lf = fin3d[k];
        w = (int)(lf.cols * cr);
        h = (int)(lf.rows * cr);
        cv::resize( lf, lf, cv::Size(w, h));

        cout << "L Finger: " << k << " size ( r, c ) : ( " << lf.rows << ", " << lf.cols << " )" << endl;
        x = 340 * k + 300 - round(lf.cols / 2);
        fingerLoc[k] = Rect( x, 1000 - lf.rows, lf.cols, lf.rows);
        lf.copyTo(canvas(fingerLoc[k]));

        cv::Mat lf2 = fake[k];
        w = (int)(lf2.cols * cr);
        h = (int)(lf2.rows * cr);
        cv::resize(lf2, lf2, cv::Size(w, h));
        x = 340 * k + 300 - round(lf2.cols / 2);
        fingerLoc[k] = Rect( x, 1300 - lf2.rows, lf2.cols, lf2.rows);
        lf2.copyTo(canvas(fingerLoc[k]));

    }
    
}

void compose()
{
    cout << "Composing . . ." << endl;
    
    state = COMPOSING;

    canvas = cv::Mat3b(monitorHeight - 4, monitorWidth - 4, cv::Vec3b(0, 0, 0));
    cv::putText(canvas, "Composing customized nails . . .",
        Size(canvas.cols - 600, 40), font, 0.7, Scalar(0, 255, 0), 1);

    next_hand_btn = cv::Rect(canvas.cols - 400, 500, 320, 40);
    canvas(next_hand_btn) = Vec3b(0, 200, 0);
    putText(canvas(next_hand_btn), "Print",
        Point((int)(next_hand_btn.width * 0.4), (int)(next_hand_btn.height * 0.7)), font, 0.8, Scalar(0, 0, 0), 2);

    exit_btn = cv::Rect(canvas.cols - 400, 600, 320, 40);
    canvas(exit_btn) = Vec3b(0, 200, 0);
    putText(canvas(exit_btn), "Exit Application",
        Point((int)(exit_btn.width * 0.17), (int)(exit_btn.height * 0.7)), font, 0.8, Scalar(0, 0, 0), 2);

    putText(canvas, "Scale: 85.6 millimeters",
        cv::Point(int((monitorWidth - 400 - CC_LEN_PX) / 2), 70), font, 0.73, Scalar(0, 255, 0), 1);
    size_t x1, x2;
    x1 = int((monitorWidth - 400 - CC_LEN_PX) / 2);
    x2 = x1 + CC_LEN_PX;
    cv::line(canvas, cv::Point(x1, 100), cv::Point(x2, 100), cv::Scalar(0, 255, 0), 2);

    int xhand = 200 * 5 + 240 + 700;
    float tmp = monitorHeight;
    tmp /= 20;
    int sec = (int)tmp;
    int yhand = 2 * sec;
    int h = 8 * sec;
    string f = imgFiles.getFingerSeg();
    Mat img = cv::imread(f);
    int w = (int)(h * img.cols / img.rows);
    cv::resize(img, img, Size(w, h));
    img.copyTo(canvas(Rect(xhand, yhand, w, h)));

    yhand = 11 * sec;
    f = imgFiles.getThumbSeg();
    img = cv::imread(f);
    w = (int)(h * img.cols / img.rows);
    cv::resize(img, img, Size(w, h));
    img.copyTo(canvas(Rect(xhand, yhand, w, h)));


    fakeNailMapping();

    cv::imshow(WIN_DEF, canvas);
    cv::waitKey(0);
}

size_t fingerFileIndex(size_t finger_id)
{
    size_t idx = 0;
    switch (finger_id )
    {
        case 0:
        case 1:
        case 2:
        case 3: idx = 0;
            break;
        case 4: idx = 1;
            break;
    }
    return idx;
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

    double dd = dist;
    dd /= conversion_rate[current_finger_image_idx];
    cc_length[current_finger_image_idx] = (int)dd;
    conversion_rate[current_finger_image_idx] = conversion_rate[current_finger_image_idx] * CC_LEN_PX / dist;

    cout << "Angle s1: " << angleToS1 << ",  Angle s2: " << angleToS2 << ", Distance s1-s2: " << dist << endl;
}


float angle2Lines(cv::Point l1p1, cv::Point l1p2, cv::Point l2p1, cv::Point l2p2)
{
    float l1ToHorizontal, l2ToHorizontal, angle;

    l1ToHorizontal = atan2(l1p1.y - l1p2.y, l1p1.x - l1p2.x) * 180 / M_PI;
    l2ToHorizontal = atan2(l2p1.y - l2p2.y, l2p1.x - l2p2.x) * 180 / M_PI;

    l1ToHorizontal = (l1ToHorizontal < 0) ? 180 + l1ToHorizontal : l1ToHorizontal;
    l2ToHorizontal = (l2ToHorizontal < 0) ? 180 + l2ToHorizontal : l2ToHorizontal;

    l1ToHorizontal = (l1ToHorizontal > 90.0) ? 180 - l1ToHorizontal : l1ToHorizontal;
    l2ToHorizontal = (l2ToHorizontal > 90.0) ? 180 - l2ToHorizontal : l2ToHorizontal;

    angle = l1ToHorizontal + l2ToHorizontal;

    //cout << "Angle between le-se:"<< angle<< "  le: " << l1ToHorizontal << " se: " << l2ToHorizontal << endl;

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
    float cr = 1.0;

    string f = imgFiles.getFingerMask(idx);
    
    switch (idx)
    {
        case 0:
        case 1:
        case 2:
        case 3: cr = conversion_rate[0];
                break;
        case 4: cr = conversion_rate[1];
                break;
    }
    cv::Mat img = cv::imread(f);
    cr = conversion_rate[fingerFileIndex(idx)];
    w = (int)(img.cols * cr);
    h = (int)(img.rows * cr);
    cv::resize(img, img, cv::Size( w, h));
    
    fingers[idx] = img;
    save[idx] = img.clone();
    return img;
}


void rotate(size_t finidx, float angle, bool dosave = false)
{
    Mat img; 
    if (dosave)
    {
        img = save[finidx];
    }
    else
    {
        img = fingers[finidx];
    }

    cv::Point2f center((img.cols - 1) / 2.0, (img.rows - 1) / 2.0);
    cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
    // determine bounding rectangle, center not relevant
    cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), img.size(), angle).boundingRect2f();
    // adjust transformation matrix
    rot.at<double>(0, 2) += bbox.width / 2.0 - img.cols / 2.0;
    rot.at<double>(1, 2) += bbox.height / 2.0 - img.rows / 2.0;

    Mat dst;
    cv::warpAffine(img, dst, rot, bbox.size());
    if (dosave)
    {
        save[finidx] = dst;
    }
    else
    {
        fingers[finidx] = dst;
    }
}


void turn(size_t finidx, int x, int y)
{
    Rect fr = fingerLoc[finidx];
    int fx = x - fr.x;
    int midx = fr.width / 2;
    int dist = midx - fx;
    int direction = (dist > 0) ? 1 : -1;
    dist = abs(dist);
    if (dist > (fr.width * 6 / 8))
    {
        rotate(finidx, 2 * direction);
        turn_angle[finidx] += 2 * direction;
    }
    else
    {
        rotate(finidx, 1 * direction);
        turn_angle[finidx] += direction;
    }
    Mat lf = fingers[finidx];
    size_t k = finidx;
    Rect r = fingerLoc[k];
    cv::rectangle(canvas, Point(r.x, r.y), Point(r.x + r.width, r.y + r.height), Scalar(0, 0, 0), -1);
    fingerLoc[k] = Rect(200 * k + 190, 400 - lf.rows, lf.cols, lf.rows);
    r = fingerLoc[k];
    cv::rectangle(canvas, Point(r.x, r.y), Point(r.x + r.width, r.y + r.height), Scalar( 200, 0, 0), 1);
    lf.copyTo(canvas(fingerLoc[k]));
    cv::imshow(WIN_DEF, canvas);
}


void genNailMetrics()
{
    size_t w, h, cross_h, cross_v, hmid, x, y, x1 = 0, x2 = 0, left, right, step, xpos;
    Mat gray;
    float cr = 1.0;

    if (!genmetrics) return;

    for ( size_t idx = 0; idx < 5; ++idx )
    {
        Mat img = save[idx];
        string f = imgFiles.getWriteMask(idx);
        rotate(idx, turn_angle[idx], true);
        img = save[idx];
        //cv::threshold(img, img, 0.5, 255, THRESH_BINARY);
        cv::imwrite(f, img);

        xpos = 200 * idx + 190;

        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        h = gray.rows;
        w = gray.cols;

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

            //cout<< x1 << " - " << hmid << " - " << x2 << "    ";

            string metrics;
            metrics += ImageIO::ftos(pxtomm(left)) + " mm | " + ImageIO::ftos(pxtomm(right)) + " mm ";
            //cout << metrics << endl;
            size_t H = canvas.rows;
            size_t pos = xpos - 20;
            switch (idx)
            {
                case 0: pos -= 7; break;
                case 2: pos += 5; break;
                case 4: pos += 19; break;
            }
            putText(canvas, metrics.c_str(),
                cv::Point( pos, H - i * 30 - 50), font, 0.5, Scalar(0, 255, 0), 1);

            nail_metrics[idx].first.push_back(pxtomm(left));
            nail_metrics[idx].second.push_back(pxtomm(right));
        }

        Mat lf = save[idx];
        size_t k = idx;
        Rect rc = fingerLoc[k];
        cv::rectangle(canvas, Point(rc.x, rc.y), Point(rc.x + rc.width, rc.y + rc.height), Scalar(0, 0, 0), -1);
        fingerLoc[k] = Rect(200 * k + 190, 400 - lf.rows, lf.cols, lf.rows);
        rc = fingerLoc[k];
        cv::rectangle(canvas, Point(rc.x, rc.y), Point(rc.x + rc.width, rc.y + rc.height), Scalar(200, 0, 0), 1);
        lf.copyTo(canvas(fingerLoc[k]));
    }
    cv::imshow(WIN_DEF, canvas);

    imgFiles.output_csv(nail_metrics, turn_angle, cc_length);
    genmetrics = false;
}


void mouse_callback(int event, int  x, int  y, int  flag, void* param)
{
    if (event == EVENT_LBUTTONDOWN)
    {
            
        if (exit_btn.contains(Point(x, y)))
        {
            redo = false;
            cv::destroyAllWindows();
            exit(0);
        }
        else if (redo_btn.contains(Point(x, y)))
        {
            if (state == REDO_OKAY_FIN  ||  state == REDO_OKAY_THUMB)
            {
                redo = true;
                mcount = 0;
                cv::Rect rc = cv::Rect(280, 350, 1000, 120);
                canvas(rc) = Vec3b(0, 0, 0);
                cv::putText(canvas, "Re-marking this image. Press any key to proceed.",
                    Size(300, 400), font, 1, Scalar(0, 255, 0), 1);
                cv::imshow(WIN_DEF, canvas);

                state = (state == REDO_OKAY_FIN) ? MARKING_CC_FIN : MARKING_CC_THUMB;
            }
        }
        else if ( okay_btn.contains(Point(x, y)))
        {
            if (state == REDO_OKAY_FIN || state == REDO_OKAY_THUMB)
            {
                redo = false;
                mcount = 0;
                cv::Rect rc = cv::Rect(280, 350, 1000, 120);
                canvas(rc) = Vec3b(0, 0, 0);
                cv::putText(canvas, "Processing image. Press any key to proceed.",
                    Size(300, 400), font, 1, Scalar(0, 255, 0), 1);
                cv::imshow(WIN_DEF, canvas);

                state = (state == REDO_OKAY_FIN) ? MARKING_CC_THUMB : STRAIGHTENING;
            }

        }
        else if (next_hand_btn.contains(Point(x, y)))
        {
            if (state == STRAIGHTENING  )
            {
                redo = false;
                mcount = 0;
                cv::Rect rc = cv::Rect( 1190, 350, 1000, 120);
                canvas(rc) = Vec3b(0, 0, 0);
                cv::putText(canvas, "Press any key to compose output.",
                    Size(1200, 400), font, 1, Scalar(0, 255, 0), 1);
                genNailMetrics();
                cv::imshow(WIN_DEF, canvas);
                cv::waitKey(0);

                state = COMPOSING;
                compose();
            }
            else if (state == COMPOSING)
            {
                redo = false;
                mcount = 0;
                string msg = string("SVG file printed to ") + svgdir;
                cv::putText(canvas, msg.c_str(),
                    Size(canvas.cols - 600, 120), font, 0.7, Scalar(0, 255, 0), 1);
                cv::Rect rc = cv::Rect(1190, 750, 1000, 120);
                canvas(rc) = Vec3b(0, 0, 0);
                cv::putText(canvas, "Press any key to process next hand.",
                    Size(970, 800), font, 1, Scalar(0, 255, 0), 1);
                cv::imshow(WIN_DEF, canvas);
                cv::waitKey(0);

                state = STARTING;
            }

        }
        else if (mcount < 0)
        {
            if (state == STRAIGHTENING)
            {
                if (fingerLoc[0].contains(Point(x, y)))
                {
                    cout << "finger 0" << endl;
                    turn(0, x, y);
                    mcount = -1;
                }
                else if (fingerLoc[1].contains(Point(x, y)))
                {
                    cout << "finger 1" << endl;
                    turn(1, x, y);
                    mcount = -1;
                }
                else if (fingerLoc[2].contains(Point(x, y)))
                {
                    cout << "finger 2" << endl;
                    turn(2, x, y);
                    mcount = -1;
                }
                else if (fingerLoc[3].contains(Point(x, y)))
                {
                    cout << "finger 2" << endl;
                    turn(3, x, y);
                    mcount = -1;
                }
                else if (fingerLoc[4].contains(Point(x, y)))
                {
                    cout << "finger 4" << endl;
                    turn(4, x, y);
                    mcount = -1;
                }
            }
        }
        else
        {
            // Store point coordinates
            pt.x = x;
            pt.y = y;
            //cout << "Point ( " << pt.x << ", " << pt.y << ")" << endl;

            if (mcount == 0)
            {
                se1 = cv::Point(x, y);
                cv::circle(img1, se1, 3, (0, 255, 255), 2);
                img1.copyTo(canvas(Rect(0, 0, img1.cols, img1.rows)));
                cv::imshow(WIN_DEF, canvas);
            }
            else if (mcount == 1)
            {
                se2 = cv::Point(x, y);
                cv::circle(img1, se2, 3, (0, 255, 255), 2);
                cv::line(img1, se1, se2, cv::Scalar(0, 255, 0), 1);
                cv::putText(canvas, "Mark the next short edge.",
                    Size(img1.cols + 6, 80), font, 0.7, Scalar(0, 255, 0), 1);
                img1.copyTo(canvas(Rect(0, 0, img1.cols, img1.rows)));
                cv::imshow(WIN_DEF, canvas);
            }
            if (mcount == 2)
            {
                se3 = cv::Point(x, y);
                cv::circle(img1, se3, 3, (0, 255, 255), 2);
                img1.copyTo(canvas(Rect(0, 0, img1.cols, img1.rows)));
                cv::imshow(WIN_DEF, canvas);
            }
            else if (mcount == 3)
            {
                se4 = cv::Point(x, y);
                cv::circle(img1, se4, 3, (0, 255, 255), 2);
                cv::line(img1, se3, se4, cv::Scalar(0, 255, 0), 1);
                cv::putText(canvas, "Mark a long edge.",
                    Size(img1.cols + 6, 120), font, 0.7, Scalar(0, 255, 0), 1);
                img1.copyTo(canvas(Rect(0, 0, img1.cols, img1.rows)));
                cv::imshow(WIN_DEF, canvas);
            }
            else if (mcount == 4)
            {
                le1 = cv::Point(x, y);
                cv::circle(img1, le1, 3, (0, 255, 255), 2);
                img1.copyTo(canvas(Rect(0, 0, img1.cols, img1.rows)));
                cv::imshow(WIN_DEF, canvas);
            }
            else if (mcount == 5)
            {
                le2 = cv::Point(x, y);
                cv::circle(img1, le2, 3, (0, 255, 255), 2);
                cv::line(img1, le1, le2, cv::Scalar(0, 255, 0), 1);

                state = (state == MARKING_CC_FIN) ? REDO_OKAY_FIN : REDO_OKAY_THUMB;

                processImageScaling();

                redo_btn = cv::Rect(img1.cols + 40, 160, 320, 32);
                okay_btn = cv::Rect(img1.cols + 40, 220, 320, 32);
                canvas(redo_btn) = Vec3b(200, 200, 200);
                canvas(okay_btn) = Vec3b(200, 200, 200);
                putText(canvas(redo_btn), "Redo This Image", Point((int)(okay_btn.width * 0.23), (int)(okay_btn.height * 0.7)), font, 0.6, Scalar(0, 0, 0), 2);
                putText(canvas(okay_btn), "Process Image", Point((int)(okay_btn.width * 0.23), (int)(okay_btn.height * 0.7)), font, 0.6, Scalar(0, 0, 0), 2);

                img1.copyTo(canvas(Rect(0, 0, img1.cols, img1.rows)));
                cv::imshow(WIN_DEF, canvas);
            }
            else
            {
                // do nothing
            }
            ++mcount;
        }
    }
}


vector<string> getFiles()
{
    vector<string> imgf; // std::string in opencv2.4, but cv::String in 3.0
    string path = cfg.working_dir + "*_image.png";
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


void clear_metrics()
{
    for (size_t i = 0; i < GL_NUM_FINGERS; ++i)
    {
        nail_metrics[i].first.clear();
        nail_metrics[i].second.clear();
    }
    reset();
}


void reset()
{
    cc_length[0] = 0;
    cc_length[1] = 0;
    conversion_rate[0] = 0;
    conversion_rate[1] = 0;
    for (size_t i = 0; i < 5; ++i)
    {
        turn_angle[i] = 0;
    }
}

