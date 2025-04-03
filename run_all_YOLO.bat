@echo off
echo Running YOLO Batch Case Study 2...
call Yolo_batch_Case_Study_2.bat

:: Check if the first batch script failed
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Yolo_batch_Case_Study_2.bat encountered an error.
    echo Press any key to exit.
    pause > nul
    exit /b %ERRORLEVEL%
)

echo.
echo YOLO Batch Case Study 2 completed successfully!
echo.
echo Running YOLO Batch Case Study 3...
call Yolo_batch_Case_Study_3.bat

:: Check if the second batch script failed
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Yolo_batch_Case_Study_3.bat encountered an error.
    echo Press any key to exit.
    pause > nul
    exit /b %ERRORLEVEL%
)

echo.
echo YOLO Batch Case Study 3 completed successfully!
echo.
echo All batch scripts executed successfully.
echo Press any key to exit.
pause > nul
