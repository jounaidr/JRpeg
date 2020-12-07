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
from JRpeg_compress import JRpeg_compress
from JRpeg_decompress import JRpeg_decompress

root = Tk()
root.resizable(False, False)
root.title('JRpeg and friends image compression =D')

# Set default values
c_input_filename = StringVar()
c_input_filename.set("bmp-img/IC1.bmp")

d_input_filename = StringVar()
d_input_filename.set("encoded_img")

output_filename = StringVar()
output_filename.set("encoded_img")

cbcr_downsize_rate = IntVar()
cbcr_downsize_rate.set(2)

QL_rate = IntVar()
QL_rate.set(1)

QC_rate = IntVar()
QC_rate.set(1)


JRpeg_comp_label = Label(root, text="JRpeg Compression:", font='Helvetica 12 bold' )

c_input_filename_label = Label(root, text ='Input Filename:')
c_input_filename_input = Entry(root, textvariable = c_input_filename)

output_filename_label = Label(root, text = 'Output Filename:')
output_filename_input = Entry(root, justify='left', textvariable = output_filename)

cbcr_downsize_rate_label = Label(root, text = 'CbCr Downsize Rate:')
cbcr_downsize_rate_input = Entry(root, justify='left', textvariable = cbcr_downsize_rate)

QL_rate_label = Label(root, text = 'QL_rate:')
QL_rate_input = Entry(root, justify='left', textvariable = QL_rate)

QC_rate_label = Label(root, text = 'QC_rate:')
QC_rate_input = Entry(root, justify='left', textvariable = QC_rate)

JRpeg_compress_button = Button(master=root, text='Compress Image', command= lambda: JRpeg_compress(c_input_filename.get(), output_filename.get(), int(cbcr_downsize_rate.get()), int(QL_rate.get()), int(QC_rate.get())))


JRpeg_decomp_label = Label(root, text="JRpeg Decompression:", font='Helvetica 12 bold')

d_input_filename_label = Label(root, text ='Input Filename:')
d_input_filename_input = Entry(root, textvariable = d_input_filename)

JRpeg_decompress_button = Button(master=root, text='Decompress Image', command= lambda: JRpeg_decompress(d_input_filename.get()))

JRpeg_comp_label.grid(row=0, column=0)
c_input_filename_label.grid(row=0, column=1)
c_input_filename_input.grid(row=0, column=2)
output_filename_label.grid(row=0, column=3)
output_filename_input.grid(row=0, column=4)
cbcr_downsize_rate_label.grid(row=0, column=5)
cbcr_downsize_rate_input.grid(row=0, column=6)
QL_rate_label.grid(row=0, column=7)
QL_rate_input.grid(row=0, column=8)
QC_rate_label.grid(row=0, column=9)
QC_rate_input.grid(row=0, column=11)
JRpeg_compress_button.grid(row=0, column=12)

JRpeg_decomp_label.grid(row=1, column=0)
d_input_filename_label.grid(row=1, column=1)
d_input_filename_input.grid(row=1, column=2)
JRpeg_decompress_button.grid(row=1, column=12)

root.mainloop()

