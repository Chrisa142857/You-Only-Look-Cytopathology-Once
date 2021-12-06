# C++ Deployment

OS: Linux

Dependencies:
 - Libtorch=1.0
 - openslide

Install:
```
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=\path\to\Libtorch ..
cmake --build .. --config Release
```

Usage:
`build/main <tile-side> <path-to-detector-model> <path-to-classifier-model> <SVS-path1> [<SVS-path2>...]\n`

Models:
https://drive.google.com/drive/folders/1UoMeYe5coWmgXjRIpHEbuOJ84PaC6AJL?usp=sharing

Programming:
![programming](programming.png)
