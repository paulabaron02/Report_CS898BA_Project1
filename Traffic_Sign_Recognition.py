"""
Traffic Sign Recognition 
Maria Paula Baron Rodriguez 
Wsu ID: J858Q278
"""
#import sys

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
#import scipy.stats as sp
#import random
import shutil
#import re



plt.close("all")


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# Folder containing all selected traffic sign images
Input_Folder = r"C:\Users\Paula\.spyder-py3\MariaBaron-CS898BA-Project1\Report_CS898BA_Project1\Selected_Images"

# Folder where all processed images will be saved
Results_Folder = r"C:\Users\Paula\.spyder-py3\MariaBaron-CS898BA-Project1\Report_CS898BA_Project1\Results"


# Delete the previous Results folder and create a new one
if os.path.exists(Results_Folder):
    shutil.rmtree(Results_Folder)

os.makedirs(Results_Folder)


# ============================================================
# 2. CREATE OUTPUT FOLDERS
# ============================================================

Output_Folders = {
    "Resized": os.path.join(Results_Folder,"01_Resized"),
    "Grayscale": os.path.join(Results_Folder,"02_Grayscale"),
    "HSV": os.path.join(Results_Folder,"03_HSV"),
    "HLS": os.path.join(Results_Folder,"04_HLS"),
    "LAB": os.path.join(Results_Folder,"05_LAB"),
    "Equalized": os.path.join(Results_Folder,"06_Histogram_Equalization"),
    "Gaussian": os.path.join(Results_Folder,"07_Gaussian_Blur"),
    "Edges": os.path.join(Results_Folder,"08_Edge_Detection"),
    "Plots": os.path.join(Results_Folder,"09_Comparison_Plots")}


# Create every output folder
for folder in Output_Folders.values():
    os.makedirs(folder, exist_ok=True)

# All images will be resized to the same dimensions
Image_Size = (64, 64)

# Gaussian blur parameter
Gaussian_Sigma = 1.0

# Canny edge-detection thresholds
Canny_Threshold_1 = 50
Canny_Threshold_2 = 150

Image_Information = []

for filename in os.listdir(Input_Folder):

    # Only process supported image files
    if not filename.lower().endswith((".png")):
        continue

    Image_Path = os.path.join(Input_Folder,filename)
    image = cv2.imread(Image_Path)
    image_name = os.path.splitext(filename)[0]
    original_height, original_width = image.shape[:2]
    print("\nProcessing:", filename)

    Resized_Image = cv2.resize(image,Image_Size,interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(Output_Folders["Resized"],image_name + "_Resized.png"),Resized_Image)

    Gray_Image = cv2.cvtColor(Resized_Image,cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(Output_Folders["Grayscale"],image_name + "_Grayscale.png"),Gray_Image)

    HSV_Image = cv2.cvtColor(Resized_Image,cv2.COLOR_BGR2HSV)
    cv2.imwrite(os.path.join(Output_Folders["HSV"],image_name + "_HSV.png"),HSV_Image)
    
    HLS_Image = cv2.cvtColor(Resized_Image,cv2.COLOR_BGR2HLS)
    cv2.imwrite(os.path.join(Output_Folders["HLS"],image_name + "_HLS.png"),HLS_Image)
    
    LAB_Image = cv2.cvtColor(Resized_Image,cv2.COLOR_BGR2LAB)
    cv2.imwrite(os.path.join(Output_Folders["LAB"],image_name + "_LAB.png"),LAB_Image)
    
    H_Channel, S_Channel, V_Channel = cv2.split(HSV_Image) 
    V_Equalized = cv2.equalizeHist(V_Channel)
    HSV_Equalized = cv2.merge([H_Channel, S_Channel, V_Equalized])
    Equalized_BGR = cv2.cvtColor(HSV_Equalized,cv2.COLOR_HSV2BGR)
    cv2.imwrite(os.path.join(Output_Folders["Equalized"],image_name + "_Equalized.png"),Equalized_BGR)

    Gaussian_Image = cv2.GaussianBlur(Equalized_BGR,(0, 0), sigmaX=Gaussian_Sigma, sigmaY=Gaussian_Sigma)
    cv2.imwrite(os.path.join(Output_Folders["Gaussian"],image_name + "_Gaussian.png"),Gaussian_Image)

    Gaussian_Gray = cv2.cvtColor(Gaussian_Image,cv2.COLOR_BGR2GRAY)
    Sobel_X = cv2.Sobel(Gaussian_Gray,cv2.CV_64F,1,0,ksize=3)
    Sobel_Y = cv2.Sobel(Gaussian_Gray,cv2.CV_64F,0,1,ksize=3)
    Sobel_Image = cv2.magnitude(Sobel_X, Sobel_Y)
    Sobel_Image = cv2.convertScaleAbs(Sobel_Image)
    cv2.imwrite(os.path.join(Output_Folders["Edges"], image_name + "_Sobel.png"),Sobel_Image)

    Laplacian_Image = cv2.Laplacian(Gaussian_Gray,cv2.CV_64F)
    Laplacian_Image = cv2.convertScaleAbs(Laplacian_Image)
    cv2.imwrite(os.path.join(Output_Folders["Edges"],image_name + "_Laplacian.png"),Laplacian_Image)

    Canny_Image = cv2.Canny(Gaussian_Gray, Canny_Threshold_1, Canny_Threshold_2)
    cv2.imwrite(os.path.join(Output_Folders["Edges"],image_name + "_Canny.png"),Canny_Image)

    Prewitt_Kernel_X = np.array([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]],dtype=np.float32)
    Prewitt_Kernel_Y = np.array([[-1, -1, -1],[0, 0, 0],[1, 1, 1]],dtype=np.float32)

    Prewitt_X = cv2.filter2D(Gaussian_Gray.astype(np.float32),-1,Prewitt_Kernel_X)
    Prewitt_Y = cv2.filter2D(Gaussian_Gray.astype(np.float32),-1,Prewitt_Kernel_Y)
    Prewitt_Image = np.sqrt(Prewitt_X ** 2 + Prewitt_Y ** 2)

    Prewitt_Image = np.uint8(np.clip(Prewitt_Image,0,255))
    cv2.imwrite(os.path.join(Output_Folders["Edges"],image_name + "_Prewitt.png"),Prewitt_Image)

    Image_Information.append({
    "Filename": filename,
    "Original Width": original_width,
    "Original Height": original_height,
    "New Width": Image_Size[0],
    "New Height": Image_Size[1],
    "Average Gray Intensity": round(float(Gray_Image.mean()),4),
    "Gray Standard Deviation": round(float(Gray_Image.std()),4),
    "Gaussian Sigma": Gaussian_Sigma,
    "Canny Lower Threshold": Canny_Threshold_1,
    "Canny Upper Threshold": Canny_Threshold_2})

    Resized_RGB = cv2.cvtColor(Resized_Image,cv2.COLOR_BGR2RGB)
    Equalized_RGB = cv2.cvtColor(Equalized_BGR,cv2.COLOR_BGR2RGB)
    Gaussian_RGB = cv2.cvtColor(Gaussian_Image,cv2.COLOR_BGR2RGB)
    HSV_Display = cv2.cvtColor(HSV_Image,cv2.COLOR_HSV2RGB)

    fig = plt.figure(figsize=(12, 9))

    fig.suptitle(f"Traffic Sign Image Processing\n{filename}",fontsize=14)

    ax1 = fig.add_subplot(3, 3, 1)
    ax1.imshow(Resized_RGB)
    ax1.set_title("Resized")
    ax1.axis("off")

    ax2 = fig.add_subplot(3, 3, 2)
    ax2.imshow(Gray_Image, cmap="gray")
    ax2.set_title("Grayscale")
    ax2.axis("off")

    ax3 = fig.add_subplot(3, 3, 3)
    ax3.imshow(HSV_Display)
    ax3.set_title("HSV")
    ax3.axis("off")

    ax4 = fig.add_subplot(3, 3, 4)
    ax4.imshow(Equalized_RGB)
    ax4.set_title("Histogram Equalization")
    ax4.axis("off")

    ax5 = fig.add_subplot(3, 3, 5)
    ax5.imshow(Gaussian_RGB)
    ax5.set_title(f"Gaussian Blur\nSigma = {Gaussian_Sigma}")
    ax5.axis("off")

    ax6 = fig.add_subplot(3, 3, 6)
    ax6.imshow(Sobel_Image, cmap="gray")
    ax6.set_title("Sobel")
    ax6.axis("off")

    ax7 = fig.add_subplot(3, 3, 7)
    ax7.imshow(Laplacian_Image, cmap="gray")
    ax7.set_title("Laplacian")
    ax7.axis("off")

    ax8 = fig.add_subplot(3, 3, 8)
    ax8.imshow(Canny_Image, cmap="gray")
    ax8.set_title("Canny")
    ax8.axis("off")

    ax9 = fig.add_subplot(3, 3, 9)
    ax9.imshow(Prewitt_Image, cmap="gray")
    ax9.set_title("Prewitt")
    ax9.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(Output_Folders["Plots"],image_name + "_Comparison.png"),dpi=200,bbox_inches="tight")
    plt.close(fig)
    print("Completed:", filename)
    
    Results_Table = pd.DataFrame(Image_Information)
    Results_Table.to_csv(os.path.join(Results_Folder,"Image_Processing_Information.csv"),index=False)

