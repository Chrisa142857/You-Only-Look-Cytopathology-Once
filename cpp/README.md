# C++ Deployment

OS: Linux

Dependencies:
 - Libtorch=1.0
 - openslide

Usage:
```
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=\path\to\Libtorch ..
cmake --build .. --config Release
```
