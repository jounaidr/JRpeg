# JRpeg

JRpeg is an image compression codec based on the methodologies used in JPEG. Implementation
is in python with a GUI application runnable as an executable, for which compression parameters can be adjusted,
and memory/disk compression rate as well as mean squared error metrics are displayed.
Full documentation is available [here.](https://github.com/jounaidr/JRpeg/blob/main/docs/JRpeg_docs.pdf) 

---

* [Instructions](#instructions)
* [Parameters and Metrics](#parameters-and-metrics)
* [build](#build)
* [License](#license)

---

## Instructions

Version 1.0 executable is available [here](https://drive.google.com/file/d/1tADfOqESTIk-7R1X5Kpk-qmboAdcY8Tw/view?usp=sharing)

Upon running the executable, the following will be displayed:\
![GUI png](https://github.com/jounaidr/JRpeg/blob/main/docs/resources/GUI.PNG)

In the GUI window the 'Original Image Filename' parameter must be first specified, then the 'Compress Image'
button can be pressed. The original image will then be loaded and displayed, and once closed the compression
process will commence, for which output logs will be displayed in the console window. After successful compression
the memory/disk size and compression rate metrics will be displayed.

A compressed .jrpg filename can then be entered in the 'JRpeg Image Filename', then the 'Decompress Image' button can
be pressed. The .jrpg file will then go through the decompression process, for which the decompressed image is then displayed,
and a BMP copy is also saved to disk. The mean squared error of the decompressed output, against the original image specified in the 'Original Image Filename'
parameter is then calculated and displayed. 

---

## License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

**[MIT license](http://opensource.org/licenses/mit-license.php)** \
Copyright 2020 © JounaidR.