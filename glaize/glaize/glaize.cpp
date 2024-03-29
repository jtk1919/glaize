// glaize.cpp : Defines the entry point for the application.
//

#include <windows.h> 
#include <shlobj.h>
#include <strsafe.h>
#include <iostream>
#include <opencv2//opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "framework.h"
#include "glaize.h"
#include "config.h"

#include "main.h"

#define MAX_LOADSTRING 100


const int ID_LOGBOX = 10004;
const int ID_PYDIR_TBOX = 10005;
const int ID_PYDIR_BTN = 10006;
HWND logbox;
HWND pydir_tbox;
wstring pydirs;

const int MAX_LOG_LEN = 32000;
wstring logstr;


size_t monitor_width, monitor_height;
DWORD aOldColors[4];
int aElements[4] = { COLOR_WINDOW, COLOR_WINDOWTEXT, COLOR_BTNFACE, COLOR_BTNTEXT };

// Global Variables:
HINSTANCE hInst;                                // current instance
WCHAR szTitle[MAX_LOADSTRING];                  // The title bar text
WCHAR szWindowClass[MAX_LOADSTRING];            // the main window class name

// Forward declarations of functions included in this code module:
ATOM                MyRegisterClass(HINSTANCE hInstance);
BOOL                InitInstance(HINSTANCE, int);
LRESULT CALLBACK    WndProc(HWND, UINT, WPARAM, LPARAM);
INT_PTR CALLBACK    About(HWND, UINT, WPARAM, LPARAM);
static void RedirectIOToConsole();
static void printLog(HWND hWnd, wstring str);
static void ErrorExit(wstring wFunction);


int APIENTRY wWinMain(_In_ HINSTANCE hInstance,
                     _In_opt_ HINSTANCE hPrevInstance,
                     _In_ LPWSTR    lpCmdLine,
                     _In_ int       nCmdShow)
{
    UNREFERENCED_PARAMETER(hPrevInstance);
    UNREFERENCED_PARAMETER(lpCmdLine);

    // TODO: Place code here.

    // Initialize global strings
    LoadStringW(hInstance, IDS_APP_TITLE, szTitle, MAX_LOADSTRING);
    LoadStringW(hInstance, IDC_GLAIZE, szWindowClass, MAX_LOADSTRING);
    MyRegisterClass(hInstance);

    // Perform application initialization:
    if (!InitInstance (hInstance, nCmdShow))
    {
        return FALSE;
    }

    HACCEL hAccelTable = LoadAccelerators(hInstance, MAKEINTRESOURCE(IDC_GLAIZE));

    MSG msg;

    // Main message loop:
    while (GetMessage(&msg, nullptr, 0, 0))
    {
        if (!TranslateAccelerator(msg.hwnd, hAccelTable, &msg))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }

    return (int) msg.wParam;
}



//
//  FUNCTION: MyRegisterClass()
//
//  PURPOSE: Registers the window class.
//
ATOM MyRegisterClass(HINSTANCE hInstance)
{
    WNDCLASSEXW wcex;

    //HBRUSH brush = CreateSolidBrush(RGB(0, 0, 0));

    wcex.cbSize = sizeof(WNDCLASSEX);
    wcex.style          = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc    = WndProc;
    wcex.cbClsExtra     = 0;
    wcex.cbWndExtra     = 0;
    wcex.hInstance      = hInstance;
    wcex.hIcon          = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_GLAIZE));
    wcex.hCursor        = LoadCursor(nullptr, IDC_ARROW);
    wcex.hbrBackground  = (HBRUSH)(COLOR_WINDOW+1);
    wcex.lpszMenuName   = MAKEINTRESOURCEW(IDC_GLAIZE);
    wcex.lpszClassName  = szWindowClass;
    wcex.hIconSm        = LoadIcon(wcex.hInstance, MAKEINTRESOURCE(IDI_SMALL));

    return RegisterClassExW(&wcex);
}

//
//   FUNCTION: InitInstance(HINSTANCE, int)
//
//   PURPOSE: Saves instance handle and creates main window
//
//   COMMENTS:
//
//        In this function, we save the instance handle in a global variable and
//        create and display the main program window.
//
BOOL InitInstance(HINSTANCE hInstance, int nCmdShow)
{
   hInst = hInstance; // Store instance handle in our global variable

   wcscpy_s( szTitle, L"Glaize App" );
  
   HWND hWnd = CreateWindowW( szWindowClass, szTitle, WS_OVERLAPPEDWINDOW,
      100, 0, 1200, 800, nullptr, nullptr, hInstance, nullptr);
   if (!hWnd)
   {
      return FALSE;
   }
   ::SetMenu(hWnd, NULL);
   ShowWindow(hWnd, nCmdShow);
   UpdateWindow(hWnd);

   HMONITOR monitor = MonitorFromWindow(hWnd, MONITOR_DEFAULTTONEAREST);
   MONITORINFO info;
   info.cbSize = sizeof(MONITORINFO);
   GetMonitorInfo(monitor, &info);
   monitor_width = info.rcMonitor.right - info.rcMonitor.left;
   monitor_height = info.rcMonitor.bottom - info.rcMonitor.top;
   int x = (monitor_width - 1200) / 2 - 19;
   int y = (monitor_height - 800) / 2 - 19;
   ::MoveWindow(hWnd, x, y, 1200, 800, true);

   DWORD aNewColors[4], aOldColors[4];
   int aElements[4] = { COLOR_WINDOW, COLOR_WINDOWTEXT, COLOR_BTNFACE, COLOR_BTNTEXT };
   aOldColors[0] = GetSysColor(COLOR_WINDOW);
   aOldColors[1] = GetSysColor(COLOR_WINDOWTEXT);
   aOldColors[2] = GetSysColor(COLOR_BTNFACE);
   aOldColors[3] = GetSysColor(COLOR_BTNTEXT);
   aNewColors[0] = RGB(0xFF, 0xB4, 0x04);
   aNewColors[1] = RGB(240, 240, 240);
   aNewColors[2] = RGB(0, 255, 0);
   aNewColors[3] = RGB(255, 255, 255);
   //SetSysColors(1, aElements, aNewColors);

   return TRUE;
}

void check_python_dir(HWND hWnd)
{
 
    size_t len = pydirs.size();
    wstring pathEnd = pydirs.substr(len - 13);
    if (pathEnd != wstring(L"\\glaize\\nails"))
    {
        printLog(hWnd, wstring(L"[ERROR] selected directory [") + pydirs + L"] is wrong.");
        printLog(hWnd, L"[ERROR] Select Python work directory ending in '\\glaize\\nails'");
        printLog(hWnd, L"[ERROR] Resetting to default.");
        wchar_t home[100];
        GetEnvironmentVariable(L"USERPROFILE", (LPWSTR)&home, 100);
        pydirs = wstring(home) + wstring(L"\\work\\glaize\\nails");
        SetDlgItemText(hWnd, ID_PYDIR_TBOX, pydirs.c_str());
    }
}

wstring get_python_dir( HWND hWnd)
{
    wchar_t pydir[MAX_PATH];
    wmemset(pydir, '\0', MAX_PATH);

    wstring title(L"Select Python '...\\glaize\\nails' working directory");
    wcscpy_s(pydir, pydirs.c_str());
    const wchar_t *path_param = pydirs.c_str();
    printLog(hWnd, L"Select the Python directory ('<PATH>\\glaize\\nails') for RCNN recognition.");

    BROWSEINFO bi = { 0 };
    bi.hwndOwner = hWnd;
    bi.pszDisplayName = pydir;
    bi.lpszTitle = title.c_str();
    bi.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE ;
    bi.lpfn = NULL;
    bi.lParam = (LPARAM)path_param;
    LPITEMIDLIST pidl = SHBrowseForFolder(&bi);
    if (pidl != 0)
    {
        //get the name of the folder and put it in path
        SHGetPathFromIDList(pidl, pydir);

        //free memory used
        IMalloc* imalloc = 0;
        if (SUCCEEDED(SHGetMalloc(&imalloc)))
        {
            imalloc->Free(pidl);
            imalloc->Release();
        }
    }
    SetDlgItemText(hWnd, ID_PYDIR_TBOX, pydir);
    pydirs = wstring(pydir);
    return pydir;
}


static void Create(HWND hWnd, LPPAINTSTRUCT lpPS)
{
    wchar_t home[100];
    GetEnvironmentVariable(L"USERPROFILE", (LPWSTR)&home, 100);
    pydirs = wstring(home) + wstring(L"\\work\\glaize\\nails");

    pydir_tbox = CreateWindow(
        L"EDIT",   // predefined class 
        pydirs.c_str(),
        WS_CHILD | WS_VISIBLE | WS_DISABLED | WS_BORDER | WS_EX_CLIENTEDGE |
        ES_LEFT | ES_MULTILINE | ES_WANTRETURN | ES_AUTOVSCROLL,
        410,         // x position 
        120,         // y position 
        380,        //  width
        30,        //  height
        hWnd,         // parent window 
        (HMENU)ID_PYDIR_TBOX,   // edit control ID 
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);        // pointer not needed 

    CreateWindow(
        L"BUTTON",  // Predefined class; Unicode assumed 
        L"Browse",      // Button text 
        WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,  // Styles 
        800,         // x position 
        120,         // y position 
        200,        // Button width
        30,        // Button height
        hWnd,     // Parent window
        (HMENU)ID_PYDIR_BTN,       // No menu.
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

    CreateWindow(
        L"BUTTON",  // Predefined class; Unicode assumed 
        L"Run RCNN on Images",      // Button text 
        WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,  // Styles 
        480,         // x position 
        190,         // y position 
        200,        // Button width
        30,        // Button height
        hWnd,     // Parent window
        (HMENU)10001,       // No menu.
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

    CreateWindow(
        L"BUTTON",  // Predefined class; Unicode assumed 
        L"Process Hand Images",      // Button text 
        WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,  // Styles 
        480,         // x position 
        260,         // y position 
        200,        // Button width
        30,        // Button height
        hWnd,     // Parent window
        (HMENU)10002,       // No menu.
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

    CreateWindow(
        L"BUTTON",  // Predefined class; Unicode assumed 
        L"Exit Application",      // Button text 
        WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,  // Styles 
        480,         // x position 
        330,         // y position 
        200,        // Button width
        30,        // Button height
        hWnd,     // Parent window
        (HMENU)10003,       // No menu.
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);

    logbox = CreateWindow(
        L"EDIT",   // predefined class 
        NULL,        
        WS_CHILD | WS_VISIBLE | WS_DISABLED | WS_BORDER | WS_EX_CLIENTEDGE |
        ES_LEFT | ES_MULTILINE | ES_WANTRETURN| ES_AUTOVSCROLL,
        200,         // x position 
        420,         // y position 
        800,        //  width
        260,        //  height
        hWnd,         // parent window 
        (HMENU)ID_LOGBOX,   // edit control ID 
        (HINSTANCE)GetWindowLongPtr(hWnd, GWLP_HINSTANCE),
        NULL);        // pointer not needed 


}


static void Paint(HWND hWnd, LPPAINTSTRUCT lpPS)
{
    RECT rc;
    HDC hdcMem;
    HBITMAP hbmMem, hbmOld;
    HBRUSH hbrBkGnd;
    HFONT hFont;
    LOGFONT lf;

    GetClientRect(hWnd, &rc);
    hdcMem = CreateCompatibleDC(lpPS->hdc);
    hbmMem = CreateCompatibleBitmap(lpPS->hdc, rc.right - rc.left, rc.bottom - rc.top);
    hbmOld = (HBITMAP)SelectObject(hdcMem, hbmMem);
    hbrBkGnd = CreateSolidBrush(GetSysColor(COLOR_WINDOW));
    FillRect(hdcMem, &rc, hbrBkGnd);
    DeleteObject(hbrBkGnd);

    GetObject(GetStockObject(DEFAULT_GUI_FONT), sizeof(LOGFONT), &lf);
    hFont = CreateFont( -40, 0,
        lf.lfEscapement, lf.lfOrientation, lf.lfWeight,
        lf.lfItalic, lf.lfUnderline, lf.lfStrikeOut, lf.lfCharSet,
        lf.lfOutPrecision, lf.lfClipPrecision, PROOF_QUALITY,
        VARIABLE_PITCH | FF_ROMAN, TEXT("Times New Roman"));
    SelectObject(hdcMem, hFont);
   
    SetBkMode( hdcMem, TRANSPARENT);
    SetTextColor(hdcMem, RGB( 0, 0, 100));
    TCHAR greeting[] = _T("Welcome to Glaize App");
    RECT loc = rc;
    loc.top = rc.top + 19;
    DrawText(hdcMem, greeting, -1,  &loc,  DT_CENTER);

    hFont = CreateFont(-20, 0,
        lf.lfEscapement, lf.lfOrientation, lf.lfWeight,
        lf.lfItalic, lf.lfUnderline, lf.lfStrikeOut, lf.lfCharSet,
        lf.lfOutPrecision, lf.lfClipPrecision, PROOF_QUALITY,
        VARIABLE_PITCH | FF_ROMAN, TEXT("Times New Roman"));
    SelectObject(hdcMem, hFont);
    SetBkMode(hdcMem, TRANSPARENT);
    SetTextColor(hdcMem, RGB(0, 0, 100));
    TCHAR pdir[] = _T("Python working directory:");
    loc.top = rc.top + 120;
    loc.left += 200;
    DrawText(hdcMem, pdir, -1, &loc, DT_LEFT);

    BitBlt(lpPS->hdc, rc.left, rc.top, rc.right - rc.left, rc.bottom - rc.top, hdcMem, 0, 0, SRCCOPY);
    SelectObject(hdcMem, hbmOld);
    SendMessage(pydir_tbox, WM_SETFONT, (WPARAM)hFont, TRUE);
    SendMessage(logbox, WM_SETFONT, (WPARAM)hFont, TRUE);
    DeleteObject(hbmMem);
    DeleteDC(hdcMem);
}


void printLog( HWND hWnd,  wstring str)
{
    wchar_t log[MAX_LOG_LEN];
    if (logstr.empty())
    {
        logstr += str;
    }
    else
    {
        logstr = logstr + L"\r\n" + str;
    }
    _tcscpy_s( log, _countof(log), logstr.c_str());
    SetDlgItemText(hWnd, ID_LOGBOX, log);
}



int runRCNN( HWND hWnd)
{
    STARTUPINFO si = { sizeof(STARTUPINFO) };
    PROCESS_INFORMATION pi;

    LPCWSTR  msg = _tcsdup(TEXT("New image masks will be generated. \
                        \nPlease put new images in the data directory and click OK"));
    MessageBox(hWnd, msg, L"Running RCNN", NULL);

    LPTSTR cmd = _tcsdup(TEXT("/C python rec.py"));
    //cmd = _tcsdup(TEXT("/C dir"));

    if (!CreateProcess( L"C:\\Windows\\System32\\cmd.exe",
            cmd,
            NULL, NULL, 
            0, 0,
            NULL, 
            pydirs.c_str(), 
            &si, &pi))
    {
        printf("CreateProcess failed (%d).\n", GetLastError());
        return -1;
    }
    WaitForSingleObject(pi.hProcess, INFINITE);
    FreeConsole();
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    printLog( hWnd, L"Completed RCNN run.");

    config cfg;
    wstring csvf(cfg.input_csv_file.begin(), cfg.input_csv_file.end());
    wstring msg1 = L"Correctly processed hand image files are logged in:\n  " + csvf;
    printLog(hWnd, msg1);

    size_t i1 = csvf.find(L"rec");
    csvf.replace(i1, 3, L"err");
    msg1 = L"Hand image files the RCNN did not process correctly are logged in:\n  " + csvf;
    printLog(hWnd, msg1);

    ::SetForegroundWindow(hWnd);
    
}


//
//  FUNCTION: WndProc(HWND, UINT, WPARAM, LPARAM)
//
//  PURPOSE: Processes messages for the main window.
//
//  WM_COMMAND  - process the application menu
//  WM_PAINT    - Paint the main window
//  WM_DESTROY  - post a quit message and return
//
//
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    PAINTSTRUCT ps;

    switch (message)
    {
    case WM_CREATE:
        Create(hWnd, &ps);
        break;
    case WM_COMMAND:
        {
            int wmId = LOWORD(wParam);
            
            // Parse the menu selections:
            switch (wmId)
            {
            case ID_PYDIR_BTN:
                pydirs = get_python_dir(hWnd);
                check_python_dir(hWnd);
                break;
            case 10001:
                printLog(hWnd, L"Running RCNN recognition.  Do not close the window or console. . .");
                runRCNN(hWnd);
                return 0;
            case 10002:
                RedirectIOToConsole();
                main(monitor_height, monitor_width, pydirs);
                return 0;
            case 10003:
                //DestroyWindow(hWnd);
                cv::destroyAllWindows();
                exit(0);
                return 0;
            case IDM_ABOUT:
                DialogBox(hInst, MAKEINTRESOURCE(IDD_ABOUTBOX), hWnd, About);
                break;
            case IDM_EXIT:
                //DestroyWindow(hWnd);
                exit(0);
                cv::destroyAllWindows();
                break;
            default:
                return DefWindowProc(hWnd, message, wParam, lParam);
            }
        }
        break;
    case WM_PAINT:
        {
            BeginPaint(hWnd, &ps);
            // TODO: Add any drawing code that uses hdc here...
            Paint(hWnd, &ps);
            EndPaint(hWnd, &ps);
        }
        break;
    
    case WM_CTLCOLORSTATIC:
        if ((HWND)lParam == GetDlgItem(hWnd, 10004))
        {
            RECT rc;
            COLORREF col= RGB(250, 250, 250);
            HBRUSH br = CreateSolidBrush(col);
            GetClientRect((HWND)lParam, &rc);
            FillRect((HDC)wParam, &rc, br);
            SetBkColor((HDC)wParam, col);
            SetBkMode((HDC)wParam, TRANSPARENT);
            SetTextColor((HDC)wParam, RGB(0, 0, 000));
            return (LRESULT)GetSysColorBrush(COLOR_WINDOW);
            // if edit control is in dialog procedure change LRESULT to INT_PTR
        }
        break;

    case WM_DESTROY:
        aOldColors[0] = RGB(255,255,255);
        SetSysColors(1, aElements, aOldColors);
        PostQuitMessage(0);
        break;
    default:
        return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;
}



// Message handler for about box.
INT_PTR CALLBACK About(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{
    UNREFERENCED_PARAMETER(lParam);
    switch (message)
    {
    case WM_INITDIALOG:
        return (INT_PTR)TRUE;

    case WM_COMMAND:
        if (LOWORD(wParam) == IDOK || LOWORD(wParam) == IDCANCEL)
        {
            EndDialog(hDlg, LOWORD(wParam));
            return (INT_PTR)TRUE;
        }
        break;
    }
    return (INT_PTR)FALSE;
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


// Format a readable error message, display a message box, 
// and exit from the application.
void ErrorExit(wstring wFunction)
{
    LPVOID lpMsgBuf;
    LPVOID lpDisplayBuf;
    DWORD dw = GetLastError();

    const wchar_t* lpszFunction = wFunction.c_str();

    FormatMessage(
        FORMAT_MESSAGE_ALLOCATE_BUFFER |
        FORMAT_MESSAGE_FROM_SYSTEM |
        FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL,
        dw,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPTSTR)&lpMsgBuf,
        0, NULL);

    lpDisplayBuf = (LPVOID)LocalAlloc(LMEM_ZEROINIT,
        (lstrlen((LPCTSTR)lpMsgBuf) + lstrlen((LPCTSTR)lpszFunction) + 40) * sizeof(TCHAR));
    StringCchPrintf((LPTSTR)lpDisplayBuf,
        LocalSize(lpDisplayBuf) / sizeof(TCHAR),
        TEXT("%s failed with error %d: %s"),
        lpszFunction, dw, lpMsgBuf);
    MessageBox(NULL, (LPCTSTR)lpDisplayBuf, TEXT("Error"), MB_OK);

    LocalFree(lpMsgBuf);
    LocalFree(lpDisplayBuf);
    ExitProcess(1);
}