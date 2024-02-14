@echo off
setlocal

if "%1"=="" (
    exit /b
) else (
    set "Tag=%1"
)


git commit --allow-empty -m "Release %Tag%"
git tag -a %Tag% -m "Version %Tag%"
git push origin --tags

endlocal
