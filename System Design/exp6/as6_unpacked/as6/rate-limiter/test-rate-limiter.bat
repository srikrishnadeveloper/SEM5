@echo off
REM test-rate-limiter.bat
REM
REM Sends a burst of requests to the rate-limited /api/resource endpoint
REM and prints the HTTP status code for each one, so you can see the
REM token bucket allow requests up to capacity and then start returning 429.
REM
REM Usage:
REM   test-rate-limiter.bat                 # 15 requests, client-id "user1"
REM   test-rate-limiter.bat 20              # 20 requests
REM   test-rate-limiter.bat 20 user2        # 20 requests, client-id "user2"

setlocal enabledelayedexpansion

set BASE_URL=http://localhost:8080
set COUNT=%~1
set CLIENT_ID=%~2
if "%COUNT%"=="" set COUNT=15
if "%CLIENT_ID%"=="" set CLIENT_ID=user1

echo Sending %COUNT% requests as client-id: %CLIENT_ID%
echo -------------------------------------------------

set success=0
set limited=0

for /L %%i in (1,1,%COUNT%) do (
    for /f %%j in ('curl.exe -s -o nul -w "%%{http_code}" "%BASE_URL%/api/resource" -H "client-id: %CLIENT_ID%"') do set status=%%j

    if "!status!"=="200" (
        echo Request %%i: !status! OK
        set /a success+=1
    ) else if "!status!"=="429" (
        echo Request %%i: !status! TOO MANY REQUESTS
        set /a limited+=1
    ) else (
        echo Request %%i: !status! (unexpected - is the app running?)
    )
)

echo -------------------------------------------------
echo Summary: %success% succeeded, %limited% rate-limited
echo.
echo Current bucket status:
curl.exe -s "%BASE_URL%/api/rate-limit-status" -H "client-id: %CLIENT_ID%"
echo.
