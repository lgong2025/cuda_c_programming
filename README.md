# 🚀 CUDA C/C++ Programming 

This repository contains personal code samples created while learning **CUDA C++ programming**, primarily following the [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html).

---

## ⚙️ Setup

Here, we assume the host OS is **Ubuntu** and you have a compatible NVIDIA GPU and modern development environment (GCC, CMake).

### Prerequisites

1.  **NVIDIA Drivers:** Ensure the latest NVIDIA proprietary drivers are installed.
2.  **CUDA Toolkit:** Install the CUDA Toolkit compatible with your drivers.
3.  **Host Compiler:** Ensure a compatible GCC/G++ version is installed. (Check NVIDIA documentation for version compatibility).
4.  **CMake:** Install CMake version 3.18 or higher.

### Installation Check

Verify your setup is correct:
```bash
# Check driver and CUDA version
nvidia-smi

# Check nvcc (CUDA Compiler) version
nvcc --version
```

Example Check Results:

```bash
❯ nvidia-smi                                               
Sun Nov  9 11:31:11 2025       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.274.02             Driver Version: 535.274.02   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce GTX 1070        Off | 00000000:2B:00.0  On |                  N/A |
| 29%   43C    P8              12W / 151W |    315MiB /  8192MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      3063      G   /usr/lib/xorg/Xorg                          143MiB |
|    0   N/A  N/A      3362      G   /usr/bin/gnome-shell                         66MiB |
|    0   N/A  N/A      5795      G   ...cess-track-uuid=3190708988185955192       44MiB |
|    0   N/A  N/A     13955      G   /proc/self/exe                               46MiB |
|    0   N/A  N/A     21107      G   /snap/zotero-snap/117/zotero-bin              9MiB |
+---------------------------------------------------------------------------------------+
```
```bash
❯ nvcc --version                                           
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2020 NVIDIA Corporation
Built on Mon_Nov_30_19:08:53_PST_2020
Cuda compilation tools, release 11.2, V11.2.67
Build cuda_11.2.r11.2/compiler.29373293_0
```

## 📂 Samples


All samples are located in the [samples/](./samples/) directory. Each `.cu` file corresponds to a separate executable target.

### Building and Running

We use the standard out-of-source `build` pattern. All compiled binaries are placed in the `build/bin` directory.

1. Configure the project:
```bash
mkdir build && cd build
cmake ..
```
2. Build or Run the targets: The project is configured to create two targets for every sample file (e.g., for vec_add.cu, targets are vec_add and run_vec_add).
```bash
make <sample_name> # Builds only the executable (e.g., make vec_add).
make run_<sample_name> # Builds AND immediately runs the executable (e.g., make run_vec_add).
```

### Concrete Example: `vec_add`

Use the single run_ command for a clean workflow. We use the -s (silent) flag with make to suppress the "Built target" status messages, leaving only the sample's output.

```bash
# Inside the project root:
mkdir build && cd build
cmake ..

# Build and run with clean output
make -s run_vec_add
```

Expected Output:
```bash
11.000000 22.000000 33.000000 44.000000 55.000000
```

## 📝 Project Files

+ `samples/*.cu`: Contains the source code for each individual CUDA sample.
+ `.gitignore`: Excludes the `build/` directory and other common artifacts.
+ `CMakeLists.txt`: Automatically finds all `.cu` files and defines the necessary executable and run targets for each.

## 🤝 Acknowledgments

Many sections of this repository, including the CMakeLists.txt structure, the .gitignore configuration, and the documentation in this README.md, were created with the helpful guidance and assistance of Google Gemini.