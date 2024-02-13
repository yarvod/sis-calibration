@echo off
setlocal

if "%1"=="" (
    exit /b
) else (
    set "Tag=%1"
)

# Create tags
git commit --allow-empty -m "Release %Tag%"
git tag -a %Tag% -m "Version %Tag%"

# Push
git push origin --tags

endlocal