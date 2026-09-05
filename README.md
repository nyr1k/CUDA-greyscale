# CUDA - greyscale image 
First project in CUDA - learning purposes 

You need to install OpenCV headers and binaries first, along with CMake. 
I use fedora, so my command looked like this:
```
sudo dnf install opencv opencv-devel cmake
```

You also need a working nvcc (nvidia cuda compiler), and if you are all set,
then pass the following command to compile the program (don't forget to change the gpu arch flag): 
```
nvcc -arch=your_arch -o gray image_blur.cu -I/usr/include/opencv4 -lopencv_core -lopencv_imgcodecs
```
Change the include path if you've got OpenCV installed some other way. 

## Run
The first argument is the path to your image, and the second argument is execution environment.

You can either choose to use CPU or GPU. 
```
./gray path/to/image cpu|gpu
```
The program shows the execution time of the grayscale function that you choose.

## Results
On my laptop CPU requires around 1ms to make a grayscale copy of an image, while GPU needs 170-250ms for the same image.

I guess it is because GPU suffers from major overhead such as image buffer allocations and buffer transfers to the device.
Besides, the workload is too small for GPU to really shine. 
