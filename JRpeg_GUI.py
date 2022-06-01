# Author: jounaidr
# Source: https://github.com/jounaidr/JRpeg
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


from tkinter import *
from compress import compress
from JRpeg_decompress import JRpeg_decompress

JRpeg_metrics = [0,0,0,0]
JRpeg_mse_val = 0


def JRpeg_compress_and_metrics(input_filename, output_filename="JRpeg_encoded_img.jrpg", cbcr_downsize_rate=2, QL_rate=1, QC_rate=1):
    global JRpeg_metrics
    JRpeg_metrics = compress(input_filename, output_filename, int(cbcr_downsize_rate), float(QL_rate), float(QC_rate))

    JRpeg_original_inmem_size.set(str(JRpeg_metrics[0]) + " bytes")
    JRpeg_comp_inmem_size.set(str(JRpeg_metrics[1]) + " bytes")
    JRpeg_inmem_ratio.set(str(round((JRpeg_metrics[0] / JRpeg_metrics[1]), 2)))
    JRpeg_inmem_space.set(str(round(100 - (100 / (JRpeg_metrics[0] / JRpeg_metrics[1])), 3)) + "%")

    JRpeg_original_disk_size.set(str(JRpeg_metrics[2]) + " bytes")
    JRpeg_comp_disk_size.set(str(JRpeg_metrics[3]) + " bytes")
    JRpeg_disk_ratio.set(str(round((JRpeg_metrics[2] / JRpeg_metrics[3]), 2)))
    JRpeg_disk_space.set(str(round(100 - (100 / (JRpeg_metrics[2] / JRpeg_metrics[3])), 3)) + "%")


def JRpeg_decompress_and_metrics(compressed_filename, original_filename):
    global JRpeg_mse_val
    JRpeg_mse_val = JRpeg_decompress(compressed_filename, original_filename)

    JRpeg_mse.set(str(round(JRpeg_mse_val, 3)))


root = Tk()
root.resizable(False, False)
root.title('JRpeg and friends image compression =D')

### JRpeg ###

# Set default values
c_input_filename = StringVar()
c_input_filename.set("img-bmp/IC1.bmp")

d_input_filename = StringVar()
d_input_filename.set("JRpeg_encoded_img")

output_filename = StringVar()
output_filename.set("JRpeg_encoded_img")

cbcr_downsize_rate = StringVar()
cbcr_downsize_rate.set("2")

QL_rate = StringVar()
QL_rate.set("1")

QC_rate = StringVar()
QC_rate.set("1")

JRpeg_original_inmem_size = StringVar()
JRpeg_original_inmem_size.set(str(JRpeg_metrics[0]) + " bytes")

JRpeg_comp_inmem_size = StringVar()
JRpeg_comp_inmem_size.set(str(JRpeg_metrics[1]) + " bytes")

JRpeg_inmem_ratio = StringVar()
JRpeg_inmem_ratio.set("N/A")

JRpeg_inmem_space = StringVar()
JRpeg_inmem_space.set("N/A")

JRpeg_original_disk_size = StringVar()
JRpeg_original_disk_size.set(str(JRpeg_metrics[2]) + " bytes")

JRpeg_comp_disk_size = StringVar()
JRpeg_comp_disk_size.set(str(JRpeg_metrics[3]) + " bytes")

JRpeg_disk_ratio = StringVar()
JRpeg_disk_ratio.set("N/A")

JRpeg_disk_space = StringVar()
JRpeg_disk_space.set("N/A")

JRpeg_mse = StringVar()
JRpeg_mse.set("N/A")

# JRpeg compression widgets

JRpeg_comp_label = Label(root, text="JRpeg Compression:", font='Helvetica 12 bold' )

c_input_filename_label = Label(root, text ='Original Image Filename:')
c_input_filename_input = Entry(root, textvariable = c_input_filename)

output_filename_label = Label(root, text = 'JRpeg Image Filename:')
output_filename_input = Entry(root, justify='left', textvariable = output_filename)

cbcr_downsize_rate_label = Label(root, text = 'CbCr Downsize Rate:')
cbcr_downsize_rate_input = Entry(root, justify='left', textvariable = cbcr_downsize_rate)

QL_rate_label = Label(root, text = 'QL_rate:')
QL_rate_input = Entry(root, justify='left', textvariable = QL_rate)

QC_rate_label = Label(root, text = 'QC_rate:')
QC_rate_input = Entry(root, justify='left', textvariable = QC_rate)

JRpeg_compress_button = Button(master=root, text='Compress Image', command= lambda: JRpeg_compress_and_metrics(c_input_filename.get(), output_filename.get(), cbcr_downsize_rate.get(), QL_rate.get(), QC_rate.get()))

# JRpeg in mem metrics widgets

JRpeg_orig_inmem_size_label = Label(root, text ='Original In Mem Size:')
JRpeg_orig_inmem_size_input = Entry(root, textvariable = JRpeg_original_inmem_size)

JRpeg_comp_inmem_size_label = Label(root, text ='Compressed In Mem Size:')
JRpeg_comp_inmem_size_input = Entry(root, textvariable = JRpeg_comp_inmem_size)

JRpeg_inmem_ratio_label = Label(root, text ='Compression ratio:')
JRpeg_inmem_ratio_input = Entry(root, textvariable = JRpeg_inmem_ratio)

JRpeg_inmem_space_label = Label(root, text ='Space saved:')
JRpeg_inmem_space_input = Entry(root, textvariable = JRpeg_inmem_space)

JRpeg_mse_label = Label(root, text ='Mean Squared Error:')
JRpeg_mse_input = Entry(root, textvariable = JRpeg_mse)

# JRpeg on disk metrics

JRpeg_orig_disk_size_label = Label(root, text ='Original On Disk Size:')
JRpeg_orig_disk_size_input = Entry(root, textvariable = JRpeg_original_disk_size)

JRpeg_comp_disk_size_label = Label(root, text ='Compressed On Disk Size:')
JRpeg_comp_disk_size_input = Entry(root, textvariable = JRpeg_comp_disk_size)

JRpeg_disk_ratio_label = Label(root, text ='Compression ratio:')
JRpeg_disk_ratio_input = Entry(root, textvariable = JRpeg_disk_ratio)

JRpeg_disk_space_label = Label(root, text ='Space saved:')
JRpeg_disk_space_input = Entry(root, textvariable = JRpeg_disk_space)

# JRpeg decompression widgets

JRpeg_decomp_label = Label(root, text="JRpeg Decompression:", font='Helvetica 12 bold')

d_input_filename_label = Label(root, text ='JRpeg Image Filename:')
d_input_filename_input = Entry(root, textvariable = d_input_filename)

JRpeg_decompress_button = Button(master=root, text='Decompress Image', command= lambda: JRpeg_decompress_and_metrics(d_input_filename.get(), c_input_filename.get()))


# Widget mapping

# JRpeg compression mappings
JRpeg_comp_label.grid(row=0, column=0)
JRpeg_compress_button.grid(row=0, column=1)
c_input_filename_label.grid(row=0, column=2)
c_input_filename_input.grid(row=0, column=3)
output_filename_label.grid(row=0, column=4)
output_filename_input.grid(row=0, column=5)
cbcr_downsize_rate_label.grid(row=0, column=6)
cbcr_downsize_rate_input.grid(row=0, column=7)
QL_rate_label.grid(row=0, column=8)
QL_rate_input.grid(row=0, column=9)
QC_rate_label.grid(row=0, column=10)
QC_rate_input.grid(row=0, column=11)

# JRpeg metrics mappings

JRpeg_orig_inmem_size_label.grid(row=1, column=4)
JRpeg_orig_inmem_size_input.grid(row=1, column=5)
JRpeg_comp_inmem_size_label.grid(row=1, column=6)
JRpeg_comp_inmem_size_input.grid(row=1, column=7)
JRpeg_inmem_ratio_label.grid(row=1, column=8)
JRpeg_inmem_ratio_input.grid(row=1, column=9)
JRpeg_inmem_space_label.grid(row=1, column=10)
JRpeg_inmem_space_input.grid(row=1, column=11)


JRpeg_orig_disk_size_label.grid(row=2, column=4)
JRpeg_orig_disk_size_input.grid(row=2, column=5)
JRpeg_comp_disk_size_label.grid(row=2, column=6)
JRpeg_comp_disk_size_input.grid(row=2, column=7)
JRpeg_disk_ratio_label.grid(row=2, column=8)
JRpeg_disk_ratio_input.grid(row=2, column=9)
JRpeg_disk_space_label.grid(row=2, column=10)
JRpeg_disk_space_input.grid(row=2, column=11)

# Spacing
spacing = Label(root, text="", font='Helvetica 12 bold')
spacing.grid(row=3, column=0)

# JRpeg decompression mappings
JRpeg_decomp_label.grid(row=4, column=0)
JRpeg_decompress_button.grid(row=4, column=1)
d_input_filename_label.grid(row=4, column=2)
d_input_filename_input.grid(row=4, column=3)
JRpeg_mse_label.grid(row=4, column=4)
JRpeg_mse_input.grid(row=4, column=5)


root.mainloop()

