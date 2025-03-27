# -*- coding: utf-8 -*-
"""
Created on 20240521
//python code for calculting alignment parameters in SiLM 
@author: Li Hangfeng
"""

import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import tkinter as tk
from tkinter import filedialog
from ctypes import *
import ctypes as _ctypes
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
import pickle
from scipy.ndimage import map_coordinates#The corrected FOV was constructed using spline interpolation
import numba as numba
numba.config.NUMBA_DEFAULT_NUM_THREADS=4
plt.style.use("ggplot")



#Spots detection using the detection algorithm in picasso
#J. Schnitzbauer*, M.T. Strauss*, T. Schlichthaerle, F. Schueder, R. Jungmann
#Super-Resolution Microscopy with DNA-PAINT
#Nature Protocols (2017). 12: 1198-1228 DOI: https://doi.org/10.1038/nprot.2017.024
@numba.jit(nopython=True, nogil=True, cache=False)
def local_maxima(frame, box):
    """Finds pixels with maximum value within a region of interest"""
    Y, X = frame.shape
    maxima_map = np.zeros(frame.shape, np.uint8)
    box_half = int(box / 2)
    box_half_1 = box_half + 1
    for i in range(box_half, Y - box_half_1):
        for j in range(box_half, X - box_half_1):
            local_frame = frame[
                i - box_half : i + box_half + 1,
                j - box_half : j + box_half + 1,
            ]
            flat_max = np.argmax(local_frame)
            i_local_max = int(flat_max / box)
            j_local_max = int(flat_max % box)
            if (i_local_max == box_half) and (j_local_max == box_half):
                maxima_map[i, j] = 1
    y, x = np.where(maxima_map)
    return y, x


@numba.jit(nopython=True, nogil=True, cache=False)
def gradient_at(frame, y, x, i):
    gy = frame[y + 1, x] - frame[y - 1, x]
    gx = frame[y, x + 1] - frame[y, x - 1]
    return gy, gx


@numba.jit(nopython=True, nogil=True, cache=False)
def net_gradient(frame, y, x, box, uy, ux):
    box_half = int(box / 2)
    ng = np.zeros(len(x), dtype=np.float32)
    for i, (yi, xi) in enumerate(zip(y, x)):
        for k_index, k in enumerate(range(yi - box_half, yi + box_half + 1)):
            for l_index, m in enumerate(range(xi - box_half, xi + box_half + 1)):
                if not (k == yi and m == xi):
                    gy, gx = gradient_at(frame, k, m, i)
                    ng[i] += gy * uy[k_index, l_index] + gx * ux[k_index, l_index]
    return ng


@numba.jit(nopython=True, nogil=True, cache=False)
def identify_in_image(image, minimum_ng, box):
    y, x = local_maxima(image, box)
    box_half = int(box / 2)
    # Now comes basically a meshgrid
    ux = np.zeros((box, box), dtype=np.float32)
    uy = np.zeros((box, box), dtype=np.float32)
    for i in range(box):
        val = box_half - i
        ux[:, i] = uy[i, :] = val
    unorm = np.sqrt(ux**2 + uy**2)
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
    image = np.float32(frame)  # otherwise numba goes crazy
    # print('start identifying in image')
    y, x, net_gradient = identify_in_image(image, minimum_ng, box)
    # print('done identifying in image')
    if roi is not None:
        y += roi[0][0]
        x += roi[0][1]
    # print('done identifying in frame')
    return y, x, net_gradient


#Create objective function of GMM model
def objective_function(x, spot_true2, fitresult2,spot_true1):
    x1, x2, x3, x4, x5 = x
    pp=0
    jishu=0
    for ii in range(ch):
        
        xy_list = spot_true2[spot_true2[:,0]==ii,1:3]
        sigma_list = np.zeros((len(xy_list),2))
        
        sigma_list[:,0]=fitresult2[spot_true2[:,0]==ii,4]
        sigma_list[:,1]=fitresult2[spot_true2[:,0]==ii,4]
        
        xy_list2 = (spot_true1[spot_true1[:,0]==ii,1:3]).copy()
        if len(xy_list[:,0])>3 and len(xy_list2 [:,0])>3:
            jishu+=1
            gaussians = [multivariate_normal(xy, np.diag(sigma**2)) for xy, sigma in zip(xy_list, sigma_list)]#The mixed Gaussian model is constructed
            
            input_xy_0 = xy_list2.copy()
            input_xy_00 = xy_list2.copy()
            input_xy= xy_list2.copy()
            input_xy[:,0]=x3*(input_xy_00[:,0]+x1)*np.cos(x5)+x4*(input_xy_00[:,1]+x2)*np.sin(x5)
            input_xy[:,1]=-1*x3*(input_xy_00[:,0]+x1)*np.sin(x5)+x4*(input_xy_00[:,1]+x2)*np.cos(x5)
            
            pp+=sum(-1*sum([gaussian.pdf(input_xy) for gaussian in gaussians]))
    return pp



#Step1 : Extract the path to the acquired calibration file, i.e., data of the same intensities for the three channels (200frames)
# create window
root = tk.Tk()
# show window
root.deiconify()
# root.title("Selected File")
# Create the Label widget and display the prompt
label = tk.Label(root, text="Select calibration file(Cail.dat)")
label.pack()
# Open the file selection dialog box
file_path_read = filedialog.askopenfilename()
# Extract the file name and extension
file_name_read, file_extension_read = os.path.splitext(file_path_read)
# Extract file name (without suffix)
file_name_only = os.path.basename(file_name_read)
# Prints the selected file path
print("Selected File:", file_path_read)
root.destroy()





#Step 2:  Select a path to save the calibration parameters
# create window
root = tk.Tk()
root.deiconify()
label = tk.Label(root, text="Select folder path for saving(folder)")
label.pack()
folder_path_save = filedialog.askdirectory()
print("Selected Folder:", folder_path_save)
# close window
root.destroy()
# Load the acquired calibration file (read .dat image stack) 200frames
filename1  = file_path_read
# filename = "G:/tony lab/DECODE_iPALM/test_4/sgipalm/vinculin_cail03/vinculin_cail03_c2.dat"
ch=200#200frames
h=512#The size of a single frame image is 512x512
w=512
with open(filename1, "rb") as f:# Read the image data in the file(.dat) and convert it to a numpy matrix(uint16)
    stack = np.fromfile(f, dtype=np.uint16)
    stack = stack.reshape((ch,h,w))
    
#Rotate the image 45 degrees
stacked_matrix1=np.zeros((ch,726,726))
for iii in range(int(len(stack))):
    image1 = Image.fromarray(stack[iii])
    rotated_image1 = image1.rotate(45, expand=True, fillcolor="white")
    rotated_array1= np.array(rotated_image1)
    stacked_matrix1[iii,:,:]=rotated_image1   
stacked_matrix = stacked_matrix1
# plt.figure()
# plt.imshow(stacked_matrix[0,:,:])    


# Step3: Read the first image, use three circles on this image to select three FOVs, 
#achieve rough alignment, and obtain the approximate center coordinates of the three FOVs
# Initialize global variables
circle_centers = [(175, 368), (377, 369), (579, 370)]
circle_radius = 80
subtract_value = 10

# Define the initial position and radius of the three circles
def update_image():
    global first_image, image, subtract_value
    first_image = ((stacked_matrix1[0,:,:] / np.max(stacked_matrix1[0,:,:]) * 255) - subtract_value).astype(np.uint8)
    first_image = np.clip(first_image, 0, 255)  # Ensure values are within [0, 255]
    image[:,:,0] = first_image
    image[:,:,1] = first_image
    image[:,:,2] = first_image

# Define a callback function for trackbar
def trackbar_callback(x):
    global subtract_value
    subtract_value = x
    update_image()

# Define a callback function for mouse events
def mouse_callback(event, x, y, flags, param):
    global circle_centers

    if event == cv2.EVENT_LBUTTONDOWN:  # Record the center position when the left mouse button is pressed
        for i, center in enumerate(circle_centers):
            distance = np.sqrt((center[0] - x) ** 2 + (center[1] - y) ** 2)
            if distance < circle_radius:
                circle_centers[i] = (x, y)
                break

# Show reminder
root = tk.Tk()
root.deiconify()
label = tk.Label(root, text="The left mouse button moves the center of the circle, <u> increases the radius, <d> decreases the radius, and <q> quit")
label.pack()
root.mainloop()

# Initialize the first image and the image array for display
first_image = ((stacked_matrix1[0,:,:] / np.max(stacked_matrix1[0,:,:]) * 255) - subtract_value).astype(np.uint8)
first_image = np.clip(first_image, 0, 255)  # Ensure values are within [0, 255]
image = np.zeros((first_image.shape[0], first_image.shape[1], 3)).astype(np.uint8)
image[:,:,0] = first_image
image[:,:,1] = first_image
image[:,:,2] = first_image
# Show first image
cv2.namedWindow('Image1')
cv2.setMouseCallback('Image1', mouse_callback)
cv2.createTrackbar('Subtract Value', 'Image1', 10, 255, trackbar_callback)  # Trackbar to adjust subtract_value
# Loop through the image until you press the 'q' key to exit
while True:
    # Copy the image to a temporary variable so that you can draw on it
    temp_image = image.copy()

    # Draw circles over the temporary image
    for center in circle_centers:
        cv2.circle(temp_image, center, circle_radius, (255, 255, 0), 2)
    # Show temporary image
    cv2.imshow('Image1', temp_image)
    # Detect keyboard event
    key = cv2.waitKey(1)
    # Adjust the radius of the circle using the keyboard
    if key == ord('u'):  # Press 'u' to increase the radius
        circle_radius += 5
    elif key == ord('d'):  # Press 'd' to reduce the radius
        circle_radius = max(5, circle_radius - 5)

    # Press the 'q' key to exit the loop
    if key == ord('q'):
        break

# Release windows and destroy all created windows
cv2.destroyAllWindows()

# Print the center of the record
print("Recorded circle centers:")
for i, center in enumerate(circle_centers):
    print(f"Circle {i+1}: ({center[0]}, {center[1]})")

# Record the center coordinates of the three circles, that is, the approximate center coordinates of the three FOVs
if len(circle_centers) == 0:
    x0 = 175
    y0 = 368
    x1 = 377
    y1 = 369
    x2 = 579
    y2 = 370
else:
    x0 = int(circle_centers[0][0])
    y0 = int(circle_centers[0][1])
    x1 = int(circle_centers[1][0])
    y1 = int(circle_centers[1][1])
    x2 = int(circle_centers[2][0])
    y2 = int(circle_centers[2][1])
    
 
    
 
    
 
    
#Step 4: The acquired image data is converted into photon counting data
EMG=200.0#EMgain
Baseline=200.0
QE=1.0
# sensitivity=4.82#gain3
sensitivity=15.13#gain1 
ww=90#Crop three FOVs out of the image with a size of 180x180
# Crop three areas of the image
region1 = stacked_matrix1[:,y0-ww:y0+ww, x0-ww:x0+ww]-0
region2 = stacked_matrix1[:,y1-ww:y1+ww, x1-ww:x1+ww]-0
region3 = stacked_matrix1[:,y2-ww:y2+ww, x2-ww:x2+ww]-0
region1=np.float32(sensitivity*(np.float32(region1)-Baseline)/EMG/QE)  
region2=np.float32(sensitivity*(np.float32(region2)-Baseline)/EMG/QE) 
region3=np.float32(sensitivity*(np.float32(region3)-Baseline)/EMG/QE) 
region1[region1 < 0] = 0
region2[region2 < 0] = 0
region3[region3 < 0] = 0

coordinate={}
fit_results = []
box=7#The spot detection window is set to 7 pixels
window_size=7
minimum_ng = 500#Set the detection value to 800
segment_shape=region1.shape
spot1= np.empty((0, 3))
spot2= np.empty((0, 3))
spot3= np.empty((0, 3))
#The approximate center coordinates of the light spots in the three FOVs are obtained
for i1 in range(region1.shape[0]):
# for i1 in range(1):
    frame1 = region1[i1, :, :]
    frame2 = region2[i1, :, :]
    frame3 = region3[i1, :, :]
    i, j, _ = identify_in_frame(frame1, minimum_ng, box, roi=None)
    ii, jj, _ = identify_in_frame(frame2, minimum_ng, box, roi=None)
    iii, jjj, _ = identify_in_frame(frame3, minimum_ng, box, roi=None)
    
    for i6 in range(len(i)):
        if i[i6]>window_size // 2 and i[i6]<segment_shape[1]-window_size//2:
                if j[i6]>window_size // 2 and j[i6]<segment_shape[1]-window_size//2:
                    result=[i1,j[i6],i[i6]]
                    spot1 = np.vstack([spot1, result])
    for i6 in range(len(ii)):
        if ii[i6]>window_size // 2 and ii[i6]<segment_shape[1]-window_size//2:
                if jj[i6]>window_size // 2 and jj[i6]<segment_shape[1]-window_size//2:
                    result=[i1,jj[i6],ii[i6]]
                    spot2 = np.vstack([spot2, result])
    for i6 in range(len(iii)):
        if iii[i6]>window_size // 2 and iii[i6]<segment_shape[1]-window_size//2:
                if jjj[i6]>window_size // 2 and jjj[i6]<segment_shape[1]-window_size//2:
                    result=[i1,jjj[i6],iii[i6]]
                    spot3 = np.vstack([spot3, result])
#Crop the spot out of the FOV to size 7x7
segments_1=[]
segments_2=[]
segments_3=[]
for i2 in range(len(spot1)):
    x_start = int(max(0, spot1[i2,1] - window_size // 2))
    x_end = int(min(2*ww, spot1[i2,1] + window_size // 2))
    y_start = int(max(0, spot1[i2,2] - window_size // 2))
    y_end = int(min(2*ww, spot1[i2,2] + window_size // 2))
    window_data = region1[int(spot1[i2,0]), y_start:y_end, x_start:x_end]
    segments_1.append(window_data)
for i2 in range(len(spot2)):
    x_start = int(max(0, spot2[i2,1] - window_size // 2))
    x_end = int(min(2*ww, spot2[i2,1] + window_size // 2))
    y_start = int(max(0, spot2[i2,2] - window_size // 2))
    y_end = int(min(2*ww, spot2[i2,2] + window_size // 2))
    window_data = region2[int(spot2[i2,0]), y_start:y_end,x_start:x_end]
    segments_2.append(window_data)
for i2 in range(len(spot3)):
    x_start = int(max(0, spot3[i2,1] - window_size // 2))
    x_end = int(min(2*ww, spot3[i2,1] + window_size // 2))
    y_start = int(max(0, spot3[i2,2] - window_size // 2))
    y_end = int(min(2*ww, spot3[i2,2] + window_size // 2))
    window_data = region3[int(spot3[i2,0]), y_start:y_end,x_start:x_end]
    segments_3.append(window_data)
result_segments_1 = np.stack(segments_1) 
result_segments_2 = np.stack(segments_2) 
result_segments_3 = np.stack(segments_3) 
plt.figure()
plt.imshow(result_segments_1[0,:,:]) 
plt.figure()
plt.imshow(result_segments_2[0,:,:]) 
plt.figure()
plt.imshow(result_segments_3[0,:,:]) 




#Step 5: The GPU is used for fine xy localization of the spot
datanew1=result_segments_1
datanew1=np.float32(datanew1)
datanew2=result_segments_2
datanew2=np.float32(datanew2)
datanew3=result_segments_3
datanew3=np.float32(datanew3)
#Load GPU program for 2d fitting, folder's address should be changed accordingly
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
data_p1=datanew1.ctypes.data_as(POINTER(c_float))
data_p2=datanew2.ctypes.data_as(POINTER(c_float))
data_p3=datanew3.ctypes.data_as(POINTER(c_float))
psfSigma=1.1
c_psfSigma=float(psfSigma)
iterations1=50
sz1=datanew1.shape[1]
fitraw1=datanew1.shape[0]
c_fitraw1=int(fitraw1)
c_sz1=int(sz1)
c_iterations1=int(iterations1)
iterations2=50
sz2=datanew2.shape[1]
fitraw2=datanew2.shape[0]
c_fitraw2=int(fitraw2)
c_sz2=int(sz2)
c_iterations2=int(iterations2)
iterations3=50
sz3=datanew3.shape[1]
fitraw3=datanew3.shape[0]
c_fitraw3=int(fitraw3)
c_sz3=int(sz3)
c_iterations3=int(iterations3)
fitmode=1#3
c_fitmode=int(fitmode)
#define the output matrix
num_fiterror=6#7
num_fitresult=5#6
fiterror1 = np.zeros(fitraw1*num_fiterror).astype('float32')
fitresult1=np.zeros(fitraw1*num_fitresult).astype('float32')

fiterror_p1=fiterror1.ctypes.data_as(POINTER(c_float))
fitresult_p1=fitresult1.ctypes.data_as(POINTER(c_float))

fiterror2 = np.zeros(fitraw2*num_fiterror).astype('float32')
fitresult2=np.zeros(fitraw2*num_fitresult).astype('float32')

fiterror_p2=fiterror2.ctypes.data_as(POINTER(c_float))
fitresult_p2=fitresult2.ctypes.data_as(POINTER(c_float))

fiterror3 = np.zeros(fitraw3*num_fiterror).astype('float32')
fitresult3=np.zeros(fitraw3*num_fitresult).astype('float32')

fiterror_p3=fiterror3.ctypes.data_as(POINTER(c_float))
fitresult_p3=fitresult3.ctypes.data_as(POINTER(c_float))

pygpufit(data_p1,c_fitraw1,c_sz1,c_iterations1,c_psfSigma,fitresult_p1,fiterror_p1,4)#4 for mle x\y\n\bg\sigam
pygpufit(data_p2,c_fitraw2,c_sz2,c_iterations2,c_psfSigma,fitresult_p2,fiterror_p2,4)
pygpufit(data_p3,c_fitraw3,c_sz3,c_iterations3,c_psfSigma,fitresult_p3,fiterror_p3,4)

fiterror1=np.reshape(fiterror1,[fitraw1,num_fiterror])
fitresult1=np.reshape(fitresult1,[fitraw1,num_fitresult])#x\y\n\bg\sigam_x\sigam_y=sigam_x

fiterror2=np.reshape(fiterror2,[fitraw2,num_fiterror])
fitresult2=np.reshape(fitresult2,[fitraw2,num_fitresult])#x\y\n\bg\sigam_x\sigam_y=sigam_x

fiterror3=np.reshape(fiterror3,[fitraw3,num_fiterror])
fitresult3=np.reshape(fitresult3,[fitraw3,num_fitresult])#x\y\n\bg\sigam_x\sigam_y=sigam_x

print("Done")





##Step 6: Calcualte the alignment paramters using the GMM model
spot_true1=spot1.copy()
spot_true2=spot2.copy()
spot_true3=spot3.copy()
offset=window_size // 2

#The xy coordinates of the light spots in the three FOVs are obtained
spot_true1[:,1]=fitresult1[:,0]+spot_true1[:,1]-offset
spot_true1[:,2]=fitresult1[:,1]+spot_true1[:,2]-offset

spot_true2[:,1]=fitresult2[:,0]+spot_true2[:,1]-offset
spot_true2[:,2]=fitresult2[:,1]+spot_true2[:,2]-offset

spot_true3[:,1]=fitresult3[:,0]+spot_true3[:,1]-offset
spot_true3[:,2]=fitresult3[:,1]+spot_true3[:,2]-offset

#Only spots with photon numbers greater than 500 and sigma less than 1.5 are retained
spot_true1 = spot_true1[np.logical_and(fitresult1[:, 2] > 500, fitresult1[:, 4] < 1.5), :]
spot_true2 = spot_true2[np.logical_and(fitresult2[:, 2] > 500, fitresult2[:, 4] < 1.5), :]
spot_true3 = spot_true3[np.logical_and(fitresult3[:, 2] > 500, fitresult3[:, 4] < 1.5), :]

fitresult1 = fitresult1[np.logical_and(fitresult1[:, 2] > 500, fitresult1[:, 4] < 1.5), :]
fitresult2 = fitresult2[np.logical_and(fitresult2[:, 2] > 500, fitresult2[:, 4] < 1.5), :]
fitresult3 = fitresult3[np.logical_and(fitresult3[:, 2] > 500, fitresult3[:, 4] < 1.5), :]

initial_guess=np.array([0,0,1,1,0/180*np.pi])#Initial optimization parameter,dx=0,dy=0,x_scale=1,y_scale=1,rotation=0

result = minimize(objective_function, initial_guess, args=(spot_true2, fitresult2,spot_true1))
x_result1=result.x#The calibration parameters of FOV1 are obtained

result = minimize(objective_function, initial_guess, args=(spot_true2, fitresult2,spot_true3))
x_result3=result.x#The calibration parameters of FOV3 are obtained
#180 pixels are used, needs to be revised accordingly
x = np.linspace(0, 179, 180)
y = np.linspace(0, 179, 180)

new_x_0 = np.linspace(0, 179, 180)
new_y_0 = np.linspace(0, 179, 180)

nn_x_1=np.zeros((180,180))
nn_y_1=np.zeros((180,180))
nn_x_3=np.zeros((180,180))
nn_y_3=np.zeros((180,180))

for i1 in range(180):
    for j1 in range(180):
        nn_y_1[i1,j1]=(new_x_0[j1]*np.sin(x_result1[4])+new_y_0[i1]*np.cos(x_result1[4]))/x_result1[3]-x_result1[1]
        nn_x_1[i1,j1]=(new_x_0[j1]*np.cos(x_result1[4])-new_y_0[i1]*np.sin(x_result1[4]))/x_result1[2]-x_result1[0]
        nn_y_3[i1,j1]=(new_x_0[j1]*np.sin(x_result3[4])+new_y_0[i1]*np.cos(x_result3[4]))/x_result3[3]-x_result3[1]
        nn_x_3[i1,j1]=(new_x_0[j1]*np.cos(x_result3[4])-new_y_0[i1]*np.sin(x_result3[4]))/x_result3[2]-x_result3[0]
             
coordinates_1=np.column_stack((nn_x_1.flatten(), nn_y_1.flatten()))
coordinates_3=np.column_stack((nn_x_3.flatten(), nn_y_3.flatten()))
# 180* 180 pixels
h=180
w=180

cc=coordinates_1.copy()
cc[:,0]=coordinates_1[:,1].copy()
cc[:,1]=coordinates_1[:,0].copy()
i7=110
rotated_matrix_1 = map_coordinates(region1[i7,:,:], cc.T, order=3).reshape(h, w)#The corrected image of FOV1 in the first image is obtained

cc=coordinates_3.copy()
cc[:,0]=coordinates_3[:,1].copy()
cc[:,1]=coordinates_3[:,0].copy()
rotated_matrix_3 = map_coordinates(region3[i7,:,:], cc.T, order=3).reshape(h, w)#The corrected image of FOV3 in the first image is obtained

plt.figure()
plt.imshow(rotated_matrix_1)
plt.figure()
plt.imshow(region2[i7,:,:])
plt.figure()
plt.imshow(rotated_matrix_3)




#Step 7: Save calibration parameters
alin_list=(x0,y0,x1,y1,x2,y2,x_result1,x_result3)
# save file
# with open(file_name_only+'alin_list_0602_vin_cail_03_c2.pkl', 'wb') as file:
with open(folder_path_save+"/"+file_name_only+'_calibration_para.pkl', 'wb') as file:
    pickle.dump(alin_list, file)
print("Done")