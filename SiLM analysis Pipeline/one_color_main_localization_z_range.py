# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 15:17:03 2024

@author: e0947330
"""

# -*- coding: utf-8 -*-
"""
Created on 20240522

@author: Li Hangfeng
"""

import numpy as _np
import tifffile
import numba as _numba
_numba.config.NUMBA_DEFAULT_NUM_THREADS=4
import ctypes as _ctypes
import matplotlib.pyplot as _plt
import os
from datetime import datetime
import time
import cv2
import math
from tkinter import simpledialog
from tkinter import messagebox
import pickle
from PIL import Image
from scipy.ndimage import shift, rotate
from ctypes import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import filedialog
from scipy.ndimage import map_coordinates
import scipy.io as sio
import datetime

_plt.style.use("ggplot")
#Spots detection using the detection algorithm in picasso
#J. Schnitzbauer*, M.T. Strauss*, T. Schlichthaerle, F. Schueder, R. Jungmann
#Super-Resolution Microscopy with DNA-PAINT
#Nature Protocols (2017). 12: 1198-1228 DOI: https://doi.org/10.1038/nprot.2017.024
@_numba.jit(nopython=True, nogil=True, cache=False)
def local_maxima(frame, box):
    """Finds pixels with maximum value within a region of interest"""
    Y, X = frame.shape
    maxima_map = _np.zeros(frame.shape, _np.uint8)
    box_half = int(box / 2)
    box_half_1 = box_half + 1
    for i in range(box_half, Y - box_half_1):
        for j in range(box_half, X - box_half_1):
            local_frame = frame[
                i - box_half : i + box_half + 1,
                j - box_half : j + box_half + 1,
            ]
            flat_max = _np.argmax(local_frame)
            i_local_max = int(flat_max / box)
            j_local_max = int(flat_max % box)
            if (i_local_max == box_half) and (j_local_max == box_half):
                maxima_map[i, j] = 1
    y, x = _np.where(maxima_map)
    return y, x


@_numba.jit(nopython=True, nogil=True, cache=False)
def gradient_at(frame, y, x, i):
    gy = frame[y + 1, x] - frame[y - 1, x]
    gx = frame[y, x + 1] - frame[y, x - 1]
    return gy, gx


@_numba.jit(nopython=True, nogil=True, cache=False)
def net_gradient(frame, y, x, box, uy, ux):
    box_half = int(box / 2)
    ng = _np.zeros(len(x), dtype=_np.float32)
    for i, (yi, xi) in enumerate(zip(y, x)):
        for k_index, k in enumerate(range(yi - box_half, yi + box_half + 1)):
            for l_index, m in enumerate(range(xi - box_half, xi + box_half + 1)):
                if not (k == yi and m == xi):
                    gy, gx = gradient_at(frame, k, m, i)
                    ng[i] += gy * uy[k_index, l_index] + gx * ux[k_index, l_index]
    return ng


@_numba.jit(nopython=True, nogil=True, cache=False)
def identify_in_image(image, minimum_ng, box):
    y, x = local_maxima(image, box)
    box_half = int(box / 2)
    # Now comes basically a meshgrid
    ux = _np.zeros((box, box), dtype=_np.float32)
    uy = _np.zeros((box, box), dtype=_np.float32)
    for i in range(box):
        val = box_half - i
        ux[:, i] = uy[i, :] = val
    unorm = _np.sqrt(ux**2 + uy**2)
    ux /= unorm
    uy /= unorm
    ng = net_gradient(image, y, x, box, uy, ux)
    positives = ng > minimum_ng
    y = y[positives]
    x = x[positives]
    ng = ng[positives]
    return y, x, ng


def identify_in_frame(frame, minimum_ng, box, roi=None):
    # print('start identifying in frame')
    if roi is not None:
        frame = frame[roi[0][0] : roi[1][0], roi[0][1] : roi[1][1]]
    image = _np.float32(frame)  # otherwise numba goes crazy
    # print('start identifying in image')
    y, x, net_gradient = identify_in_image(image, minimum_ng, box)
    # print('done identifying in image')
    if roi is not None:
        y += roi[0][0]
        x += roi[0][1]
    # print('done identifying in frame')
    return y, x, net_gradient



# create window to load the path of calibration parameters(.pkl)
root = tk.Tk()
# show window
root.deiconify()
label = tk.Label(root, text="Select calibration parameters(.pkl)")
label.pack()
# Open the file selection dialog box
file_path_read = filedialog.askopenfilename()
# Prints the selected file path
print("Selected File:", file_path_read)
root.destroy()


# create window to select the path to save points
root = tk.Tk()
# show window
root.deiconify()
label = tk.Label(root, text="Select folder path for saving(folder)")
label.pack()
folder_path_save = filedialog.askdirectory()
print("Selected Folder:", folder_path_save)
# close window
root.destroy()


# open the file of calibration parameters(.pkl)
with open(file_path_read, 'rb') as file:
    alin_list = pickle.load(file) 
x0,y0,x1,y1,x2,y2,x_result1,x_result3=alin_list
print("Done")

# create window to load the path of pending data(.dat)
root = tk.Tk()
# show window
root.deiconify()
label = tk.Label(root, text="Select the data to process(.dat)")
label.pack()
file_path_read_dat = filedialog.askopenfilename()
file_name_read_dat, file_extension_read_dat = os.path.splitext(file_path_read_dat)
# Extract file name (without suffix)
file_name_only_dat = os.path.basename(file_name_read_dat)
print("Selected File:", file_path_read_dat)
root.destroy()
filename1=file_path_read_dat




# Default parameters
defaults = {
    'Frame number(200 x N)': 20000,
    'Image size': 512,
    'EM Gain': 200,
    'Baseline': 200,
    'QE': 1.0,
    "Sensitivity": 15.13,
    "Angle1": 16.7,
    "Angle2": 37.1,
    "Angle3": 49.4,
    "z-min": 0.0,
    "z-max": 250.0,
    "Wavelength": 639.0,
    "Thickness": 481.3,
    "pixel": 133.33
}
values = defaults.copy()

# Update parameters function
def update_values():
    try:
        for key, entry in entries.items():
            user_input = entry.get()
            # Determine the type of the default value and convert accordingly
            if isinstance(defaults[key], int):
                values[key] = int(user_input)
            elif isinstance(defaults[key], float):
                values[key] = float(user_input)
        messagebox.showinfo("Info", "Values updated successfully!")
    except ValueError:
        messagebox.showerror("Error", "Invalid input! Please enter valid numbers.")

# Confirm and terminate the program's function
def confirm():
    root.destroy()

# Initialize the GUI window
root = tk.Tk()
root.title("Update Values")

entries = {}

# Create and place labels and input fields
for key, default_value in values.items():
    frame = tk.Frame(root)
    frame.pack(pady=5)

    label = tk.Label(frame, text=f"{key}:")
    label.pack(side=tk.LEFT)

    entry = tk.Entry(frame)
    entry.insert(0, str(default_value))
    entry.pack(side=tk.LEFT)
    entries[key] = entry

# Create and place buttons
update_button = tk.Button(root, text="OK", command=update_values)
update_button.pack(pady=10)

confirm_button = tk.Button(root, text="Confirm", command=confirm)
confirm_button.pack(pady=10)

root.mainloop()

print("Final values:", values)



#Basic information about the data to be processed
ch1=int(values['Frame number(200 x N)'])#20,000 frames
h1=int(values['Image size'])#size=512x512
w1=int(values['Image size'])

path_curr_dir = os.getcwd()
item_size = _np.dtype('uint16').itemsize#uint16

EMG=float(values['EM Gain'])#EMgain
Baseline=float(values['Baseline'])
QE=float(values['QE'])
sensitivity=float(values['Sensitivity'])#gain1

ww=90#Crop three FOVs out of the image with a size of 180x180
box=7#The window size when looking for a spot
window_size=7

print("Done")

#Displays a spot detection preview for specific images in the image stack
def display_images(image_stack,i,j):
    num_images = image_stack.shape[0]
    root = tk.Tk()
    root.title("Image Viewer")

    # Create canvas
    fig, axes = _plt.subplots(1, num_images, figsize=(12, 3))

    for i1 in range(num_images):
        ax = axes[i1]
        ax.imshow(image_stack[i1], cmap='gray')
        ax.scatter(j, i, s=1, c='r')
        ax.axis('off')

        # Embed the canvas in the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Create toolbar
        toolbar = NavigationToolbar2Tk(canvas, root)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    root.mainloop()

# Create main window
root = tk.Tk()
# Hide main window
root.withdraw()

x_r = _np.linspace(0, 179, 180)
y_r = _np.linspace(0, 179, 180)

new_x_0 = _np.linspace(0, 179, 180)
new_y_0 = _np.linspace(0, 179, 180)

nn_x_1=_np.zeros((180,180))
nn_y_1=_np.zeros((180,180))
nn_x_3=_np.zeros((180,180))
nn_y_3=_np.zeros((180,180))

for i1 in range(180):
    for j1 in range(180):
        nn_y_1[i1,j1]=(new_x_0[j1]*_np.sin(x_result1[4])+new_y_0[i1]*_np.cos(x_result1[4]))/x_result1[3]-x_result1[1]
        nn_x_1[i1,j1]=(new_x_0[j1]*_np.cos(x_result1[4])-new_y_0[i1]*_np.sin(x_result1[4]))/x_result1[2]-x_result1[0]
        nn_y_3[i1,j1]=(new_x_0[j1]*_np.sin(x_result3[4])+new_y_0[i1]*_np.cos(x_result3[4]))/x_result3[3]-x_result3[1]
        nn_x_3[i1,j1]=(new_x_0[j1]*_np.cos(x_result3[4])-new_y_0[i1]*_np.sin(x_result3[4]))/x_result3[2]-x_result3[0]
        
X_r, Y_r = _np.meshgrid(x_r, y_r)  

#the coordinates of the original FOV to be aligned corresponding to each pixel (〖pixel〗_(x_new,y_new )) 
#in the new aligned FOV
coordinates_1=_np.column_stack((nn_x_1.flatten(), nn_y_1.flatten()))
cc1=coordinates_1.copy()
cc1[:,0]=coordinates_1[:,1].copy()
cc1[:,1]=coordinates_1[:,0].copy()
  
coordinates_3=_np.column_stack((nn_x_3.flatten(), nn_y_3.flatten()))
cc3=coordinates_3.copy()
cc3[:,0]=coordinates_3[:,1].copy()
cc3[:,1]=coordinates_3[:,0].copy()

#According to minimum_ng, preview the spot detection of 5 images from beginning to end in the image stack
#Enter minimum_ng (for example, 800) and click OK to update the preview. 
#Click Cancel to perform spot detection on the last minimum_ng.
while True:
    # The input dialog box is displayed
    user_input = simpledialog.askfloat("input_number", "Please enter a number:")
    # Check whether the user entered a value
    if user_input is not None:
        print("A value entered by the user:", user_input)
        minimum_ng = user_input
        stacked_matrix1=_np.zeros((5,512,512))
        tz=ch1/4
        for iii in range(5):
            if iii<4:
                skip_bytes = int(iii*tz) * h1 * w1 * item_size
                with open(filename1, 'rb') as file:
                    file.seek(skip_bytes)
                    data = _np.fromfile(file, dtype='uint16', count=h1 * w1)
                    data = data.reshape(h1, w1)
                image1 = Image.fromarray(data)
                stacked_matrix1[iii,:,:]=image1
            else:
                skip_bytes = int(iii*tz-1) * h1 * w1 * item_size
                with open(filename1, 'rb') as file:
                    file.seek(skip_bytes)
                    data = _np.fromfile(file, dtype='uint16', count=h1 * w1)
                    data = data.reshape(h1, w1)
                image1 = Image.fromarray(data)
                stacked_matrix1[iii,:,:]=image1
        stacked_matrix2=_np.zeros((5,726,726))
        for iii in range(int(len(stacked_matrix1))):
            image1 = Image.fromarray(stacked_matrix1[iii])
            rotated_image1 = image1.rotate(45, expand=True, fillcolor="white")
            rotated_array1= _np.array(rotated_image1)
            stacked_matrix2[iii,:,:]=rotated_image1
        stacked_matrix2=_np.float32(sensitivity*(stacked_matrix2-Baseline)/EMG/QE)  
        stacked_matrix2[stacked_matrix2<0]=0
        real_region1 = stacked_matrix2[:,y0-ww:y0+ww, x0-ww:x0+ww]
        real_region2 = stacked_matrix2[:,y1-ww:y1+ww, x1-ww:x1+ww]
        real_region3 = stacked_matrix2[:,y2-ww:y2+ww, x2-ww:x2+ww]
        real_ali_img_1=_np.zeros(real_region1.shape)
        real_ali_img_3=_np.zeros(real_region1.shape)
        for i1 in range(int(real_region1.shape[0])):
            real_ali_img_1[i1,:,:] = _np.nan_to_num(map_coordinates(real_region1[i1,:,:], cc1.T, order=3).reshape(180, 180))
            real_ali_img_3[i1,:,:] = _np.nan_to_num(map_coordinates(real_region3[i1,:,:], cc3.T, order=3).reshape(180, 180))

        all_image=real_ali_img_1+real_region2+  real_ali_img_3 #The three corrected FOVs are combined into one FOV for display
        Exp_images = all_image.copy()
        Exp_images_16 = real_ali_img_1
        Exp_images_36 = real_region2
        Exp_images_48 = real_ali_img_3 
        coordinate={}
        fit_results = []
        segment_shape=Exp_images.shape
        spot= _np.empty((0, 3))
        fig, axes = _plt.subplots(1, Exp_images.shape[0], figsize=(40, 5))
        for i1 in range(Exp_images.shape[0]):
            frame = Exp_images[i1, :, :]
         
            i, j, net_gradient = identify_in_frame(frame, minimum_ng, box, roi=None)
            axes[i1].imshow(Exp_images[i1], cmap='gray')
            axes[i1].scatter(j, i, s=2, c='r') 
        popup = tk.Toplevel(root)
        popup.title("Image Viewer")
        canvas = FigureCanvasTkAgg(fig, master=popup)
        canvas.draw()
        canvas.get_tk_widget().pack()
        popup.update()      
    else:
        print("The user cancelled the input")
        break
# close window
root.destroy()

def Intensity(thetalist,values):
    lam=float(values['Wavelength'])*1e-9
    d_ox=float(values['Thickness'])*1e-9  
    if float(values['Wavelength'])== 639.0:
        n=1.331#647
        n_ox=1.456#647
        n_si=3.856#647               
    elif float(values['Wavelength'])== 488.0:
        n=1.331#647
        n_ox=1.463#488
        n_si=4.367#488            
    elif float(values['Wavelength'])== 561.0:
        n=1.331#647
        n_ox=1.459#561
        n_si=4.049#561         
    else:
        n=1.331#647
        n_ox=1.456#647
        n_si=3.856#647
   # calculate rTE 
    k_ox=2*math.pi*n_ox/lam
    k_si=2*math.pi*n_si/lam
    theate=thetalist#3 angles
    for ii in range(len(theate)):
        theate[ii]=theate[ii]/180*math.pi
    r_TE=_np.zeros(3).astype(_np.complex64)
    phi_theate=_np.zeros(3)
    abs_r_TE=_np.zeros(3)
    r_TE_angle=_np.zeros(3)
    for i in range(len(theate)):
        sin_theate_ox=n*_np.sin(theate[i])/n_ox
        sin_theate_si=n*_np.sin(theate[i])/n_si
        cos_theate=_np.sqrt(1-(_np.sin(theate[i]))**2)
        cos_theate_ox=_np.sqrt(1-(sin_theate_ox)**2)
        cos_theate_si=_np.sqrt(1-(sin_theate_si)**2)
        p_0=n_si*cos_theate_si
        p_1=n_ox*cos_theate_ox
        p_2=n*cos_theate
        M_TE_21=-1j*p_1*_np.sin(k_ox*d_ox*cos_theate_ox)
        M_TE_12=-1j/p_1*_np.sin(k_ox*d_ox*cos_theate_ox)
        M_TE_11=_np.cos(k_ox*d_ox*cos_theate_ox)
        M_TE_22=M_TE_11
        r_TE[i]=((M_TE_11+p_0*M_TE_12)*p_2-(M_TE_21+M_TE_22*p_0))/((M_TE_11+M_TE_12*p_0)*p_2+(M_TE_21+M_TE_22*p_0))
        phi_theate[i]=4*math.pi/lam*(n*math.cos(theate[i]))*1e-9
        abs_r_TE[i]=abs(r_TE[i])
        r_TE_angle[i]=_np.angle(r_TE[i])

    return abs_r_TE,r_TE_angle, phi_theate






# Main fitting pipeline
# Step 1: Aligment and finding spots in the raw images
#Create the spot storage matrix
aa=_np.empty((0, window_size,window_size))#The light spot after the superposition of the three FOV spots
a_16=_np.empty((0, window_size,window_size))#First light spot
a_36=_np.empty((0, window_size,window_size))#Second light spot
a_48=_np.empty((0, window_size,window_size))#Third light spot

#Spots detection of 200 frames of data at a time
frame_flag=_np.empty((0,4))
start_time = time.time()
ll=int(ch1/200)
for ii in range(ll):   
    stacked_matrix1=_np.zeros((200,726,726))
    skip_bytes = int(200*ii) * h1 * w1 * item_size
    with open(filename1, 'rb') as file:#Load 200 frames of data
        file.seek(skip_bytes)
        data = _np.fromfile(file, dtype='uint16', count=h1 * w1*200)
        data = data.reshape(200,h1, w1)
        
    #Rotate 45 degrees
    for iii in range(int(ch1/ll)):
        image1 = Image.fromarray(data[iii,:,:])
        rotated_image1 = image1.rotate(45, expand=True, fillcolor="white")
        rotated_array1= _np.array(rotated_image1)
        stacked_matrix1[iii,:,:]=rotated_image1
        
    #Convert the data into photon counts
    stacked_matrix1=_np.float32(sensitivity*(_np.float32(stacked_matrix1)-Baseline)/EMG/QE)  
    stacked_matrix1[stacked_matrix1<0]=0
    
    #Crop 3 FOVs from the image
    real_region1 = stacked_matrix1[:,y0-ww:y0+ww, x0-ww:x0+ww]
    real_region2 = stacked_matrix1[:,y1-ww:y1+ww, x1-ww:x1+ww]
    real_region3 = stacked_matrix1[:,y2-ww:y2+ww, x2-ww:x2+ww]

    real_ali_img_1=_np.zeros(real_region1.shape)#.astype(_np.uint16)
    real_ali_img_3=_np.zeros(real_region1.shape)#.astype(_np.uint16)
    
    #The first FOV and the third FOV are calibrated against the second FOV
    for i1 in range(int(real_region1.shape[0])):
        real_ali_img_1[i1,:,:] = _np.nan_to_num(map_coordinates(real_region1[i1,:,:], cc1.T, order=3).reshape(180, 180))
        real_ali_img_3[i1,:,:] = _np.nan_to_num(map_coordinates(real_region3[i1,:,:], cc3.T, order=3).reshape(180, 180))
 
    all_image=real_region2+  real_ali_img_1+real_ali_img_3#Three FOVs are superimposed to obtain the total FOV 
    #Total FOV and three FOVs(after calibration)
    Exp_images = _np.float32(all_image)
    Exp_images_16 = _np.float32(real_ali_img_1)
    Exp_images_36 = _np.float32(real_region2)
    Exp_images_48 = _np.float32(real_ali_img_3)
    
    #spots detection
    coordinate={}
    fit_results = []
    segment_shape=Exp_images.shape
    spot= _np.empty((0, 4))
    for i1 in range(Exp_images.shape[0]):
        frame = Exp_images[i1, :, :]   
        i, j, net_gradient = identify_in_frame(frame, minimum_ng, box, roi=None)
        for i6 in range(len(i)):
            if i[i6]>window_size // 2 and i[i6]<segment_shape[1]-window_size//2:
                    if j[i6]>window_size // 2 and j[i6]<segment_shape[1]-window_size//2:
                        result=[i1+ii*200,j[i6],i[i6],frame[i[i6],j[i6]]]
                        spot = _np.vstack([spot, result])

    segments=[]
    segments_16=[]
    segments_36=[]
    segments_48=[]
    #Crop spots from FOVs 
    for i2 in range(len(spot)):
        
        x_start = int(max(0, spot[i2,1] - window_size // 2))
        x_end = int(min(2*ww, spot[i2,1] + window_size // 2))
        y_start = int(max(0, spot[i2,2] - window_size // 2))
        y_end = int(min(2*ww, spot[i2,2] + window_size // 2))
        window_data = Exp_images[int(spot[i2,0]-ii*200), y_start:y_end+1,x_start:x_end+1]
        segments.append(window_data)
        window_data_16 = Exp_images_16[int(spot[i2,0]-ii*200),y_start:y_end+1,x_start:x_end+1]
        segments_16.append(window_data_16)
        window_data_36 = Exp_images_36[int(spot[i2,0]-ii*200),y_start:y_end+1,x_start:x_end+1]
        segments_36.append(window_data_36)
        window_data_48 = Exp_images_48[int(spot[i2,0]-ii*200),y_start:y_end+1,x_start:x_end+1]
        segments_48.append(window_data_48)
    try:
        result_segments = _np.stack(segments)
        result_segments_16 = _np.stack(segments_48)#In our data, first FOV is the third FOV
        result_segments_36 = _np.stack(segments_36)
        result_segments_48 = _np.stack(segments_16)
        
        aa = _np.concatenate((aa, result_segments), axis=0)#spots image stack of three channel 
        a_16 = _np.concatenate((a_16, result_segments_16), axis=0)#spots image stack of first channel 
        a_36 = _np.concatenate((a_36, result_segments_36), axis=0)#spots image stack of second channel 
        a_48 = _np.concatenate((a_48, result_segments_48), axis=0)#spots image stack of third channel 
        frame_flag = _np.concatenate((frame_flag, spot), axis=0)#Pixel coordinate of spots
    except ValueError:
        print("ValueError occurred. Skipping this data set.")
        continue
    print(ii)

#Empty memory
stack1=[]
result_segments=[]
result_segments_16=[]
result_segments_36=[]
result_segments_48=[]
real_region1=[]
real_region2=[]
real_region3=[]
segments=[]
segments_16=[]
segments_36=[]
segments_48=[]
Exp_images = []
Exp_images_16 = []
Exp_images_36 = []
Exp_images_48 = []
all_image=[]





#Step 2: Save images of the first 100 spots detected so you can check that they are running correctly
n = 100
processed_images = []
for i in range(n):
    image = aa[i]
    scaled_image = (image / _np.max(image)) * 255
    scaled_image = scaled_image.astype(_np.uint8)
    processed_images.append(scaled_image)
# create 70x70 image
big_image = Image.new('L', (70, 70))
x_offset = 0
y_offset = 0
for image in processed_images:
    # convert to pil
    image = Image.fromarray(image)
    # paste
    big_image.paste(image, (x_offset, y_offset))
    # offset
    x_offset += 7
    if x_offset == 70:
        x_offset = 0
        y_offset += 7
# save 100 spots
big_image.save(folder_path_save+"/"+file_name_only_dat+'100_spots_image.png')
    
print("Done")





#Step3: Obtain xy coordinates, photon number, background, and sigma_xy of total spot (the spot after three spots are superimposed)
datanew=aa
datanew=_np.float32(datanew)
#Modules for loading GPU program to calculate the lateral position in the sum image, dll folder should be revised accordingly
hgpu=_ctypes.CDLL('pystormbasic.dll')
#hgpu=_ctypes.CDLL('E:/test9/pystormbasic.dll')
#Run a simple addition operation to check that the GPU call is working
bb=(9,2,26)
b=(_ctypes.c_int)(2)
a=(_ctypes.c_uint*len(bb))(*bb)
hgpu.addone(a,b)
print(a[0])
#check the input of data
pygpufit=hgpu.pystorm
pygpufit.argtypes=[POINTER(c_float),c_size_t,c_size_t,c_size_t,c_float,POINTER(c_float),POINTER(c_float),c_size_t]
data_p=datanew.ctypes.data_as(POINTER(c_float))
psfSigma=1.1
c_psfSigma=float(psfSigma)
iterations=50
sz=datanew.shape[1]
fitraw=datanew.shape[0]
c_fitraw=int(fitraw)
c_sz=int(sz)
c_iterations=int(iterations)
fitmode=4#3
c_fitmode=int(fitmode)
#define the output matrix 
num_fiterror=6#7 for x\y\n\bg\sigam_x\sigam_y;6 for x\y\n\bg\sigam
num_fitresult=5#6 for x\y\n\bg\sigam_x\sigam_y; 5 for x\y\n\bg\sigam
fiterror = _np.zeros(fitraw*num_fiterror).astype('float32')
fitresult=_np.zeros(fitraw*num_fitresult).astype('float32')
fiterror_p=fiterror.ctypes.data_as(POINTER(c_float));
fitresult_p=fitresult.ctypes.data_as(POINTER(c_float));
## call cuda code for 2d fitting
pygpufit(data_p,c_fitraw,c_sz,c_iterations,c_psfSigma,fitresult_p,fiterror_p,4)#1forx\y\n\bg\sigam_x\sigam_y;3for les:x\y\n\bg\sigam_x;4 for mle x\y\n\bg\sigam
fiterror1=_np.reshape(fiterror,[fitraw,num_fiterror])
fitresult1=_np.reshape(fitresult,[fitraw,num_fitresult])#x\y\n\bg\sigam_x\sigam_y
print('fit_x_y_n_bg_sigma_in_fitresult1')





#Step4: According to x\y\sigam_x\sigam_y, the number of photons of the spot in FOV1 ,FOV2 and FOV3 are calculated
# Dll folder should be changed accordingly
hgpuRe=_ctypes.CDLL('pystormRepeat.dll')
#hgpuRe=_ctypes.CDLL('E:/test9/pystormRepeat.dll')
bb=(9,2,26)
b=(_ctypes.c_int)(2)
anew=(_ctypes.c_uint*len(bb))(*bb)
hgpuRe.addone(anew,b)
print(anew[0])
pygpufit_repeat=hgpuRe.pystorm_repeat
pygpufit_repeat.argtypes=[POINTER(c_float),c_size_t,c_size_t,c_size_t,c_float,POINTER(c_float),POINTER(c_float),c_size_t,POINTER(c_float)]
pold_l=_np.zeros(fitraw*num_fitresult).astype('float32')
# print('abc5')

num_fiterror_16=50
fiterror_16= _np.zeros(fitraw*num_fiterror_16).astype('float32')
fitresult_16=_np.zeros(fitraw*2).astype('float32')
fiterror_16_p=fiterror_16.ctypes.data_as(POINTER(c_float));
fitresult_16_p=fitresult_16.ctypes.data_as(POINTER(c_float));   
datanew_l6=_np.float32(a_16)
data_16=datanew_l6.ctypes.data_as(POINTER(c_float));
#call cuda code for fitting FOV1 intensity
pygpufit_repeat(data_16,c_fitraw,c_sz,c_iterations,c_psfSigma,fitresult_16_p,fiterror_16_p,2,fitresult_p)#1 for input x\y\n\bg\sigam_x\sigam_y;2 for input x\y\n\bg\sigam
fiterror_16_L=_np.reshape(fiterror_16,[fitraw,num_fiterror_16])
fitresult_16_L=_np.reshape(fitresult_16,[fitraw,2])

num_fiterror_36=50
fiterror_36= _np.zeros(fitraw*num_fiterror_36).astype('float32')
fitresult_36=_np.zeros(fitraw*2).astype('float32')
fiterror_36_p=fiterror_36.ctypes.data_as(POINTER(c_float));
fitresult_36_p=fitresult_36.ctypes.data_as(POINTER(c_float));  
datanew_36=_np.float32(a_36)
data_36=datanew_36.ctypes.data_as(POINTER(c_float));
#call cuda code for fitting FOV2 intensity
pygpufit_repeat(data_36,c_fitraw,c_sz,c_iterations,c_psfSigma,fitresult_36_p,fiterror_36_p,2,fitresult_p)#1
fiterror_36_L=_np.reshape(fiterror_36,[fitraw,num_fiterror_36])
fitresult_36_L=_np.reshape(fitresult_36,[fitraw,2])

num_fiterror_48=50
fiterror_48= _np.zeros(fitraw*num_fiterror_48).astype('float32')
fitresult_48=_np.zeros(fitraw*2).astype('float32')
fiterror_48_p=fiterror_48.ctypes.data_as(POINTER(c_float));
fitresult_48_p=fitresult_48.ctypes.data_as(POINTER(c_float));   
datanew_48=_np.float32(a_48)
data_48=datanew_48.ctypes.data_as(POINTER(c_float));
#call cuda code for fitting FOV3 intensity
pygpufit_repeat(data_48,c_fitraw,c_sz,c_iterations,c_psfSigma,fitresult_48_p,fiterror_48_p,2,fitresult_p)#1
fiterror_48_L=_np.reshape(fiterror_48,[fitraw,num_fiterror_48])
fitresult_48_L=_np.reshape(fitresult_48,[fitraw,2])




## Step 5: calculating rTE and set angle correction function
#z is obtained by fitting the photon numbers of the three channels
#Calculate abs_r_TE,r_TE_angle, phi_theate of the three angles
thetalist=[float(values['Angle1']),float(values['Angle2']),float(values['Angle3'])]
# Set the Angle correction factor beta whic is decided from the calibration of the real system, method for the calibration can be found in the article
if float(values['Wavelength'])== 639.0:
        beta1=1.24
        beta2=1.07
        beta3=1.0
else:
        beta1=1 
        beta2=1
        beta3=1
absrTE,anglerTE,zfact= Intensity(thetalist,values)
#Shows the intensity curve of the three angles as z changes
z_l=_np.linspace(float(values['z-min']),float(values['z-max']),251)
F1=_np.zeros(251)
F2=_np.zeros(251)
F3=_np.zeros(251)
for i22 in range(len(z_l)):
    F1[i22]=(1+absrTE[0]*absrTE[0])+2*absrTE[0]*_np.cos(zfact[0]*z_l[i22]+anglerTE[0])
    F2[i22]=(1+absrTE[1]*absrTE[1])+2*absrTE[1]*_np.cos(zfact[1]*z_l[i22]+anglerTE[1])
    F3[i22]=(1+absrTE[2]*absrTE[2])+2*absrTE[2]*_np.cos(zfact[2]*z_l[i22]+anglerTE[2])
p0 = _plt.plot(z_l, F1,"r")
p0 = _plt.plot(z_l, F2,"g")
p0 = _plt.plot(z_l, F3,"b")




#Step 6: According to the number of photons in the three FOVs, fit z call the gpu code  
# Folder where the dll is saved needed to changed accordingly
hgpuz=_ctypes.CDLL('fit_z_z_range_change.dll')
##hgpuz=_ctypes.CDLL('E:/test9/fit_z_z_range_change.dll')
bb=(5,2,26)
b=(_ctypes.c_int)(2)
a=(_ctypes.c_uint*len(bb))(*bb)
hgpuz.addone(a,b)
zrange1=_np.array([0, 10], dtype=_np.float32)
zmin=float(values['z-min'])
zmax=float(values['z-max'])
zrange1[0]=_np.floor(zmin/40.0)
zrange1[1]=_np.ceil(zmax/40.0)
zrange=zrange1.astype('int')
zrange_p=zrange.ctypes.data_as(POINTER(c_int))
sz_angle=len(thetalist)
sz_mea=len(fitresult1)
#the number of photons in the three FOVs
Fsimarr=_np.zeros((sz_mea,sz_angle),dtype=_np.float32)
Fsimarr[:,0]=fitresult_16_L[:,0]*beta1
Fsimarr[:,1]=fitresult_36_L[:,0]*beta2
Fsimarr[:,2]=fitresult_48_L[:,0]

Fsimarr=_np.float32(Fsimarr)   
pysaimfitz=hgpuz.pysaimfitz
pysaimfitz.argtypes=[POINTER(c_float),POINTER(c_float),POINTER(c_float),
                    POINTER(c_float),c_size_t,c_size_t,c_size_t,POINTER(c_float),POINTER(c_float),c_size_t,POINTER(c_int)]
data_z_p=Fsimarr.ctypes.data_as(POINTER(c_float))
anglerTE=anglerTE.astype('float32')
anglerTE_p=anglerTE.ctypes.data_as(POINTER(c_float))
absrTE=absrTE.astype('float32')
absrTE_p=absrTE.ctypes.data_as(POINTER(c_float))
zfact=zfact.astype('float32')
zfact_p=zfact.ctypes.data_as(POINTER(c_float))
iterations=50
c_sz=int(sz_angle)
c_fitraw=int(sz_mea)
c_iterations=int(iterations)
fitmode=1
c_fitmode=int(fitmode)
#define the output matrix
num_fiterror_z=3
num_fitresult_z=2
fiterror_z = _np.zeros(sz_mea*num_fiterror_z).astype('float32')
fitresult_z=_np.zeros(sz_mea*num_fitresult_z).astype('float32')
fiterror_z_p=fiterror_z.ctypes.data_as(POINTER(c_float))
fitresult_z_p=fitresult_z.ctypes.data_as(POINTER(c_float))
#call cuda code to fit the z
pysaimfitz(data_z_p,absrTE_p,anglerTE_p,zfact_p,c_fitraw,c_sz,c_iterations,fitresult_z_p,fiterror_z_p,c_fitmode, zrange_p)
fitresult_z_L=_np.reshape(fitresult_z,[sz_mea,num_fitresult_z])#obtain z, I_sig
fiterror_z_L=_np.reshape(fiterror_z,[sz_mea,num_fiterror_z])
print("Done")



#Step7: Calculate crlbz using fitting results
I_sig=fitresult_z_L[:,1]
Fs=_np.zeros((len(fitresult_16_L),len(thetalist)))
Fs[:,0]=(fitresult_16_L[:,0]/I_sig)
Fs[:,1]=(fitresult_36_L[:,0]/I_sig)
Fs[:,2]=(fitresult_48_L[:,0]/I_sig)
acos_Fs=_np.arccos(Fs)
dF1=_np.zeros(251)
dF2=_np.zeros(251)
dF3=_np.zeros(251)
n=1.3313
lam=float(values['Wavelength'])*1e-9
thetalist=[float(values['Angle1']),float(values['Angle2']),float(values['Angle3'])]
theate=thetalist#3 angles
for ii in range(len(theate)):
    theate[ii]=theate[ii]/180*math.pi
for i22 in range(len(z_l)):
    dF1[i22]=2*absrTE[0]*_np.sin(zfact[0]*z_l[i22]+anglerTE[0])*4*_np.pi/lam*(n*_np.cos(theate[0]))*1e-9
    dF2[i22]=2*absrTE[1]*_np.sin(zfact[1]*z_l[i22]+anglerTE[1])*4*_np.pi/lam*(n*_np.cos(theate[1]))*1e-9
    dF3[i22]=2*absrTE[2]*_np.sin(zfact[2]*z_l[i22]+anglerTE[2])*4*_np.pi/lam*(n*_np.cos(theate[2]))*1e-9
    
av_dF1=_np.average(_np.abs(dF1))
av_dF2=_np.average(_np.abs(dF2))
av_dF3=_np.average(_np.abs(dF3))

model=_np.zeros((len(fitresult_16_L),len(thetalist)))
model[:,0]=(1+absrTE[0]*absrTE[0])+2*absrTE[0]*_np.cos(zfact[0]*fitresult_z_L[:,0]+anglerTE[0])
model[:,1]=(1+absrTE[1]*absrTE[1])+2*absrTE[1]*_np.cos(zfact[1]*fitresult_z_L[:,0]+anglerTE[1])
model[:,2]=(1+absrTE[2]*absrTE[2])+2*absrTE[2]*_np.cos(zfact[2]*fitresult_z_L[:,0]+anglerTE[2])

cha=_np.abs(model-Fs)
cha[:,0]=cha[:,0]/av_dF1
cha[:,1]=cha[:,1]/av_dF2
cha[:,2]=cha[:,2]/av_dF3
row_means = _np.mean(cha, axis=1)#sigma_z
print("Done")



#Step 8: SAVE fitting results
## aa6=2, save 1D gaussin width: sx 
## aa6=1, save 2D gaussian width: sx and sy
aa6=2
pixel_p=values['pixel']
if aa6==1:
    crlb_x=_np.sqrt((((fitresult1[:,4]*pixel_p)**2+((pixel_p)**2)/12)/fitresult1[:,2])*(16/9+(8*3.14159*((fitresult1[:,4]*pixel_p)**2+((pixel_p)**2)/12))*(fitresult1[:,3]**0.5)/(fitresult1[:,2]*(pixel_p)**2)))
    crlb_y=_np.sqrt((((fitresult1[:,5]*pixel_p)**2+((pixel_p)**2)/12)/fitresult1[:,2])*(16/9+(8*3.14159*((fitresult1[:,5]*pixel_p)**2+((pixel_p)**2)/12))*(fitresult1[:,3]**0.5)/(fitresult1[:,2]*(pixel_p)**2)))
    offset=int(window_size/2 )
    points=_np.zeros((len(fitresult_z_L),14))#frame number\x\y\z\crlb_x\crlb_y\crlb_z\sigma_x\sigam_z\N\bg
    #points=_np.zeros((len(fitresult_z_L),11))#frame number\x\y\z\crlb_x\crlb_y\crlb_z\sigma_x\sigam_z\N\bg
    points[:,0]=frame_flag[:,0]
    points[:,1:3]=frame_flag[:,1:3]+fitresult1[:,0:2]-offset
    points[:,3]=fitresult_z_L[:,0]
    points[:,4]=crlb_x
    points[:,5]=crlb_y
    points[:,6]=row_means
    points[:,7]=fitresult1[:,4]*pixel_p#sigam_x
    points[:,8]=fitresult1[:,5]*pixel_p#sigma_y
    points[:,9]=fitresult1[:,2]#photon
    points[:,10]=fitresult1[:,3]#bg
    points[:,11:14]=Fsimarr#3 intensities  
else:
    crlb_x=_np.sqrt((((fitresult1[:,4]*pixel_p)**2+((pixel_p)**2)/12)/fitresult1[:,2])*(16/9+(8*3.14159*((fitresult1[:,4]*pixel_p)**2+((pixel_p)**2)/12))*(fitresult1[:,3]**0.5)/(fitresult1[:,2]*(pixel_p)**2)))
    offset=int(window_size/2 )
    points=_np.zeros((len(fitresult_z_L),12))#frame number\x\y\z\crlb_x\crlb_z\sigma_x\N\bg
    # points=_np.zeros((len(fitresult_z_L),9))#frame number\x\y\z\crlb_x\crlb_z\sigma_x\N\bg
    points[:,0]=frame_flag[:,0]
    points[:,1:3]=frame_flag[:,1:3]+fitresult1[:,0:2]-offset
    points[:,3]=fitresult_z_L[:,0]
    points[:,4]=crlb_x
    points[:,5]=row_means
    points[:,6]=fitresult1[:,4]*pixel_p#sigam_x
    points[:,7]=fitresult1[:,2]#photon
    points[:,8]=fitresult1[:,3]#bg
    points[:,9:12]=Fsimarr#3 intensities

end_time = time.time()
# Calculate the runtime
runtime = end_time - start_time
# Print the runtime
print("Runtime: {:.2f} seconds".format(runtime))

## aa6=2, save 1D gaussin width: sx 
## aa6=1, save 2D gaussian width: sx and sy
if aa6==1:
    #save points
    points[:,1:3]=points[:,1:3]*pixel_p
    points = points[~_np.isnan(points[:, 3])]
    now = datetime.datetime.now()
    file_namee = f"{now.strftime('%Y%m%d_%H%M')}.npy"  
    _np.save(folder_path_save+"/"+file_name_only_dat+'_mlefit_frame_x_y_z_crlbx_crlby_crlbz_sigmax_sigma_y_N_bg_N1_N2_N3_01'+file_namee, points)  
    print("File saved successfully.")

    file_namee_mat=f"{now.strftime('%Y%m%d_%H%M')}.mat"
    sio.savemat(folder_path_save+"/"+file_name_only_dat+'_mlefit_frame_x_y_z_crlbx_crlby_crlbz_sigmax_sigma_y_N_bg_N1_N2_N3_01'+file_namee_mat, {'matrix': points})
    print("File saved successfully.")

else:
    #save points
    points[:,1:3]=points[:,1:3]*pixel_p
    points = points[~_np.isnan(points[:, 3])]
    now = datetime.datetime.now()
    file_namee = f"{now.strftime('%Y%m%d_%H%M')}.npy" 
    _np.save(folder_path_save+"/"+file_name_only_dat+'_mlefit_frame_x_y_z_crlbxy_crlbz_sigmax_y_N_bg_N1_N2_N3_01'+file_namee, points)  
    print("File saved successfully.")
    file_namee_mat=f"{now.strftime('%Y%m%d_%H%M')}.mat"
    sio.savemat(folder_path_save+"/"+file_name_only_dat+'_mlefit_frame_x_y_z_crlbxy_crlbz_sigmax_y_N_bg_N1_N2_N3_01'+file_namee_mat, {'matrix': points})
    print("File saved successfully.")

import h5py
import yaml
#points=_np.zeros((len(fitresult_z_L),9))#frame number\x\y\z\crlb_x\crlb_z\sigma_x\N\bg
frame_number=points[:,0].astype(_np.int16)
loc_x = points[:,1]/pixel_p
loc_y = points[:,2]/pixel_p
loc_z = points[:,3]
sigma_x = points[:,6]/pixel_p
sigma_y = points[:,6]/pixel_p
crlb_x = points[:,4]/pixel_p
crlb_y = points[:,4]/pixel_p
crlb_z = points[:,5]
N = points[:,7]
bg = points[:,8]
pixel_size = pixel_p
label=(loc_x*0).astype(_np.int16)
xpixel=_np.round(loc_x)
ypixel=_np.round(loc_y)

pixel_value=_np.float32(frame_flag[:,3])

h5_filename = folder_path_save+"/"+file_name_only_dat+'_mlefit_frame_x_y_z_crlbx_crlby_crlbz_sigmax_sigma_y_N_bg_'+"PALM.hdf5"
yaml_filename = folder_path_save+"/"+file_name_only_dat+'_mlefit_frame_x_y_z_crlbx_crlby_crlbz_sigmax_sigma_y_N_bg_'+"PALM.yaml"

PS_LOCS_DTYPE = [("frame", "u4"),("x", "f4"),("y", "f4"),("z","f4"),("photons", "f4"),

                 ("xw", "f4"),("yw", "f4"),("bg", "f4"),("sx", "f4"), ("sy", "f4"), ("sz","f4"),

                 ("label", "i4"),("xpixel","i4"),("ypixel","i4"),("pixel_value","f4")]

pc = _np.squeeze(_np.rec.array((frame_number,loc_x,loc_y,loc_z,N,sigma_x,sigma_y,bg,crlb_x,crlb_y,crlb_z,label,xpixel,ypixel,pixel_value),dtype = PS_LOCS_DTYPE))

xy=_np.c_[loc_x,loc_y] 
datafile=h5_filename

with h5py.File(h5_filename, "w") as outputfile:
    outputfile.create_dataset("xy", data=xy)
    outputfile.create_dataset("pointCloud", data=pc)
    outputfile.create_dataset("datafile", data=datafile)


metadata = {
    "description": "SGiPALM-locs-data",
    "author": "Hangfeng",
    "date": now,
    "other_info": "None",
    "thickness": 481.06,
    'wavelength': 640, 'camera_pixel_size_nm': 133, 'sensor':'emccd','offset':200,'photoconversion':4.8,'QE':95,'magnification':120,
                              'peakFindRadius':7, 'gaussSigma':1.2, 'maxCount': 1000,'LowerLimitN':50,'UpperLimitN':5000,'PeakFindMode':1,
                              'peakFitDimension':5,'ExposureTime':96,'TrackSearchRadius':5,'SearchGap':0,'MinTrackLength':10,'ConfinementThreshold':250,
                              'RawColorMap':'gist_gray','TrackColorMode':2, 'Opacity':.5,'MarkerSize':30,'FitsColor':'green', 'PeaksColor':'red', 'FitsMarker':'star',
                              'TextColor':'black', 'BarMode': 0,'MovieWidth':512, 'MovieHeight':512, 'MovieLength':ch1   
}

with open(yaml_filename, "w") as yaml_file:
    yaml.dump(metadata, yaml_file)
print("save YAML file.")

print(_np.mean(points[:,7]))#Calculate the average number of photons


# points=points[points[:,6]<220,:]
