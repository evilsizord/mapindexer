@echo off
setlocal enabledelayedexpansion

REM Save the current folder
set "original_dir=%cd%"

REM Change to UrbanTerror folder and launch the game
cd /d C:\games\urt\UrbanTerror43_Mapping

Quake3-UrT.exe +set fs_game q3ut4 +devmap layout_del_1 +exec _layout_del_1.cam.cfg

REM Change back to the original folder
cd /d "%original_dir%"

endlocal