echo off

rem 現在日付
echo %date%
rem 現在時刻
echo %time%

py MachineLearning.py 

rem 現在日付
echo %date%
rem 現在時刻
echo %time%

chime\1.mp3

pause