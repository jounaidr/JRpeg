# JRpeg

* [Instructions](#instructions)
* [build](#build)
* [Parameters and Metrics](#parameters-and-metrics)
* [License](#license)

JRpeg is an image compression codec based on the methodologies outlined in JPEG. Implementation
is in python with a GUI application runnable as an executable, for which compression parameters can be adjusted,
and memory/disk compression rate as well as mean squared error metrics are displayed.
Full documentation is available [here.](https://github.com/jounaidr/JRpeg/blob/main/docs/JRpeg_docs.pdf) 

---

### Instructions

Version 1.0 executable is available [here](https://github.com/jounaidr/JRpeg/releases/tag/1.0)

Upon running the executable, the following will be displayed:\
![GUI png](https://github.com/jounaidr/JRpeg/blob/main/docs/resources/GUI.PNG)

In the GUI window the `Original Image Filename` parameter must be first specified, then the `Compress Image`
button can be pressed. The original image will then be loaded and displayed, and once closed the compression
process will commence, for which output logs will be displayed in the console window. After successful compression
the memory/disk size and compression rate metrics will be displayed.

A compressed `.jrpg` filename can then be entered in the `JRpeg Image Filename`, then the `Decompress Image` button can
be pressed. The `.jrpg` file will then go through the decompression process, for which the decompressed image is then displayed,
and a BMP copy is also saved to disk. The mean squared error of the decompressed output, against the original image specified in the `Original Image Filename`
parameter is then calculated and displayed. 

---

### Build

To make changes and rebuild the GUI, do the following:

* First clone this repo from: `https://github.com/jounaidr/JRpeg.git`
* Make required changes, compression and decompression can be tested in console using the following methods respectively: `JRpeg_compress(input_filename, output_filename, cbcr_downsize_rate, QL_rate, QC_rate)` and `JRpeg_decompress(input_filename, original_filename)`
* Run `build.py`, and a directory will be generated called `dist` with the updated JRpeg executable! 

---

### Parameters And Metrics

The following parameters can be adjusted:

* **cbcr_downsize_rate**: Default = 2 (4:2:2 downsampling) any higher becomes slightly noticeable, and greater than 5 will give diminishing reduction in file size
* **QL_rate**: Default = 1 (standard JPEG luminance quantisation) increasing greatly improves compression rate but also has a big effect of image quality
* **QC_rate**: Default = 1 (standard JPEG chrominance quantisation) increasing has a small effect of compression rate but is only noticeable on images with vibrant color spots 

The following metrics can be retrieved from the GUI:

* **In Memory Metrics**
    - Original In Memory Size
    - Compressed In Memory Size
    - In Memory Compression Ratio
    - In Memory Space Saved
* **On Disk Metrics**
    - Original On Disk Size
    - Compressed On Disk Size
    - On Disk Compression Ratio
    - On Disk Space Saved
* **Mean Squared Error** (original vs decompress JRpeg image)

---

### License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

**[MIT license](http://opensource.org/licenses/mit-license.php)** \
Copyright 2020 © JounaidR.
