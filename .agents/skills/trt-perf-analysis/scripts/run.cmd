@echo off
:: SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
:: SPDX-License-Identifier: Apache-2.0
setlocal

if "%~1"=="" (
    echo usage: run.cmd ^<python-script^> [args...] 1>&2
    exit /b 2
)

set "PY_CMD="
set "PY_ARG="

if not defined SKILL_PYTHON goto after_skill_python
call :select_plain "%SKILL_PYTHON%"
if defined PY_CMD goto run_python
:after_skill_python

if not defined PYTHON goto after_python_env
call :select_plain "%PYTHON%"
if defined PY_CMD goto run_python
:after_python_env

if not defined PYTHON3 goto after_python3_env
call :select_plain "%PYTHON3%"
if defined PY_CMD goto run_python
:after_python3_env

call :select_with_arg py -3
if defined PY_CMD goto run_python
call :select_plain python
if defined PY_CMD goto run_python
call :select_plain python3
if defined PY_CMD goto run_python
call :select_plain python3.12
if defined PY_CMD goto run_python
call :select_plain python3.11
if defined PY_CMD goto run_python
call :select_plain python3.10
if defined PY_CMD goto run_python
call :select_plain python3.9
if defined PY_CMD goto run_python
call :select_plain python3.8
if defined PY_CMD goto run_python

echo error: unable to find Python 3.8+ on PATH. Set SKILL_PYTHON to a Python executable. 1>&2
exit /b 127

:run_python
if "%PY_ARG%"=="" (
    "%PY_CMD%" %*
) else (
    "%PY_CMD%" "%PY_ARG%" %*
)
exit /b %ERRORLEVEL%

:select_plain
set "PY_CMD=%~1"
set "PY_ARG="
if "%PY_CMD%"=="" goto clear_selection
"%PY_CMD%" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)" >nul 2>nul
if errorlevel 1 goto clear_selection
exit /b 0

:select_with_arg
set "PY_CMD=%~1"
set "PY_ARG=%~2"
if "%PY_CMD%"=="" goto clear_selection
"%PY_CMD%" "%PY_ARG%" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)" >nul 2>nul
if errorlevel 1 goto clear_selection
exit /b 0

:clear_selection
set "PY_CMD="
set "PY_ARG="
exit /b 1
