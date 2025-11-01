# N-Body Simulation with CUDA — How to Build and Run (Windows)

This repo contains a sequential CPU implementation and six CUDA approaches that progressively optimize an N-Body simulation. This guide walks you through installing the toolchains, building, and running everything on Windows.

## Overview

- Language/toolchains: C++17, CUDA
- Targets in `src/`:
	- `nbody_sequential.cpp` — CPU baseline
	- `approach1.cu` … `approach6.cu` — CUDA kernels from naive to advanced
- Input data: CSV files under `data/star_cluster_simulation/`
- Output logs: per-file run times stored under `data/star_cluster_simulation/timelog/`
- Progress: executables print step progress as percentages

## 0) Clone the repository and prepare data

Clone this repo:

```cmd
git clone https://github.com/Chonapatcc/N-Body-Simulation-with-CUDA.git
cd N-Body-Simulation-with-CUDA
```

Data files: If your clone doesn’t include the CSVs or you have a newer `data.zip` archive, extract it into the `data` folder so that the CSVs end up under `data\star_cluster_simulation\`.

- Using CMD (tar is available on recent Windows 10/11):
```cmd
mkdir data 2>nul
tar -xf data.zip -C data
```

- Using PowerShell (alternative):
```powershell
PowerShell -NoProfile -Command "New-Item -ItemType Directory -Force -Path data | Out-Null; Expand-Archive -Path data.zip -DestinationPath data -Force"
```

## 1) Prerequisites

Make sure you have a compatible NVIDIA GPU and drivers.

1. NVIDIA GPU Driver
	 - Update to the latest Studio/Game Ready driver.
	 - Verify the GPU is visible:
		 ```cmd
		 nvidia-smi
		 ```

2. Visual Studio C++ toolchain (MSVC)
	 - Install Visual Studio 2022 (Community or Build Tools).
	 - In Visual Studio Installer → Workloads → check:
		 - Desktop development with C++
	 - In the right pane ensure these components are included:
		 - MSVC v143 – VS 2022 C++ x64/x86 build tools
		 - Windows 10 or 11 SDK (latest available)
		 - C++ CMake tools for Windows (optional)

3. CUDA Toolkit
	 - Download and install the latest CUDA Toolkit (e.g., v13.0) from NVIDIA.
	 - Default install path (example):
		 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0`
	 - Verify `nvcc`:
		 ```cmd
		 "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\nvcc.exe" --version
		 ```

## 2) Open the correct developer shell

Use an MSVC-enabled terminal so `cl.exe` is on PATH.

- Start Menu → Visual Studio 2022 → x64 Native Tools Command Prompt for VS 2022

Verify tools:

```cmd
cl /Bv
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\nvcc.exe" --version
```

If you prefer to stay in a normal CMD after installing VS Build Tools:

```cmd
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
```

## 3) Build the project

Change to the repository root:

```cmd
cd N-Body-Simulation-with-CUDA
```

### A) Build the CPU sequential version

Using MSVC (`cl`) from the Native Tools prompt:

```cmd
cl /std:c++17 /O2 /EHsc /Fe:seq_csv_batch.exe src\nbody_sequential.cpp
```

Alternatively with MinGW g++ if installed (optional, not required):

```cmd
g++ -o seq_csv_batch.exe src\nbody_sequential.cpp -std=c++17 -O3
```

### B) Build the CUDA approaches

Set a helper variable for `nvcc` (adjust version if different):

```cmd
set NVCC="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\nvcc.exe"
```

Compile each approach with MSVC as the host compiler:

```cmd
%NVCC% -ccbin "%VCToolsInstallDir%bin\Hostx64\x64" -std=c++17 -O2 -o approach1.exe src\approach1.cu
%NVCC% -ccbin "%VCToolsInstallDir%bin\Hostx64\x64" -std=c++17 -O2 -o approach2.exe src\approach2.cu
%NVCC% -ccbin "%VCToolsInstallDir%bin\Hostx64\x64" -std=c++17 -O2 -o approach3.exe src\approach3.cu
%NVCC% -ccbin "%VCToolsInstallDir%bin\Hostx64\x64" -std=c++17 -O2 -o approach4.exe src\approach4.cu
%NVCC% -ccbin "%VCToolsInstallDir%bin\Hostx64\x64" -std=c++17 -O2 -o approach5.exe src\approach5.cu
%NVCC% -ccbin "%VCToolsInstallDir%bin\Hostx64\x64" -std=c++17 -O2 -o approach6.exe src\approach6.cu
```

Optional: specify GPU architecture (example for Ampere SM 86):

```cmd
%NVCC% -arch=sm_86 -ccbin "%VCToolsInstallDir%bin\Hostx64\x64" -O2 -o approach4.exe src\approach4.cu
```

## 4) Run the executables

Ensure input CSVs are present in:

```
data\star_cluster_simulation\
```

Then run the CPU and CUDA binaries:

```cmd
.\u005cseq_csv_batch.exe
.\u005capproach1.exe
.\u005capproach2.exe
.\u005capproach3.exe
.\u005capproach4.exe
.\u005capproach5.exe
.\u005capproach6.exe
```

What you should see:
- Progress percentage prints during the simulation steps.
- Run time logs written under `data/star_cluster_simulation/timelog/`:
	- Per-file logs: one CSV per input file (same name) containing duration_ms per run.
	- A summary CSV (`star_cluster_simulation.csv`) listing file → duration_ms for the batch.

## 5) Optional: one-click batch build

Create `build_all.bat` in the repo root and run it from the Native Tools prompt:

```bat
@echo off
setlocal
rem Ensure MSVC env is loaded if not already
if "%VCToolsInstallDir%"=="" (
	call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" || goto :eof
)
set NVCC="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\nvcc.exe"

cd /d "%~dp0"

cl /std:c++17 /O2 /EHsc /Fe:seq_csv_batch.exe src\nbody_sequential.cpp || goto :eof

for %%F in (1 2 3 4 5 6) do (
	%NVCC% -ccbin "%VCToolsInstallDir%bin\Hostx64\x64" -std=c++17 -O2 -o approach%%F.exe src\approach%%F.cu || goto :eof
)
echo Build completed.
```

## 6) Troubleshooting

- `nvcc fatal: Cannot find compiler 'cl.exe' in PATH`
	- Use the “x64 Native Tools Command Prompt for VS 2022” or run `vcvars64.bat` as shown above.
	- Verify MSVC: `cl /Bv` prints version details.

- `nvcc is not recognized`
	- Use the full path to `nvcc.exe`, or temporarily add CUDA bin to PATH for the current session:
		```cmd
		set "PATH=%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin"
		nvcc --version
		```

- Windows SDK/link errors
	- Re-run Visual Studio Installer and ensure Windows 10/11 SDK and MSVC v143 are installed.

- Slow runs / too long to finish
	- Reduce `STEPS` or the number of particles (`N`) in the code or input dataset.
	- You’ll still get progress percentages every ~10% of steps.

## 7) Notes

- Commands in `command.txt` mirror the build commands; run them from the Native Tools prompt.
- If your CUDA installs to a different version path (e.g., `v12.4`), replace `v13.0` accordingly.
- Data and results structure (partial):
	- `data/star_cluster_simulation/*.csv` — inputs
	- `data/star_cluster_simulation/timelog/*.csv` — per-file timing logs and `star_cluster_simulation.csv` summary

