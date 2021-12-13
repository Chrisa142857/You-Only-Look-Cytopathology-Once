# C++ Deployment

OS: 
 - Linux

Dependencies:
 - Libtorch
 - Libtorchvision (CPU)
 - openslide

## Quick Start
```
build/main \
--detector /path/to/detector.pt \
--classifier /path/to/classifier.pt \
--input_side 5120 \
--verbose \
"/path/to/WSI1" \
["/path/to/WSI2" "/path/to/WSI3" ...]
```

## Install
Please install `openslide` at first. Let `PackageConfig` can find `openslide.pc`.
```
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=\path\to\Libtorch -DCMAKE_INSTALL_PREFIX=\path\to\Libtorchvision ..
cmake --build ..
```


## Avaliable Models:
https://drive.google.com/drive/folders/1UoMeYe5coWmgXjRIpHEbuOJ84PaC6AJL?usp=sharing

Programming:
![programming](programming.png)
