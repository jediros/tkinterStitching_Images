# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 21:28:00 2023

@author: jra02028
"""

import tkinter as tk
from tkinter import filedialog
import cv2
import glob
import numpy as np
import imutils
import os

def select_folder():
    folder_path = filedialog.askdirectory()
    if folder_path:
        display_images(folder_path)

def display_images(folder_path):
    image_paths = glob.glob(f'{folder_path}/*.jpg') + glob.glob(f'{folder_path}/*.png')

    for image_path in image_paths:
        img = cv2.imread(image_path)
        cv2.imshow("Image", img)
        cv2.waitKey(0)

    process_images(image_paths)

def process_images(image_paths):
    images = []

    for image_path in image_paths:
        img = cv2.imread(image_path)
        images.append(img)

    image_stitcher = cv2.Stitcher_create()
    error, stitched_img = image_stitcher.stitch(images)

    if not error:
        cv2.imshow("Stitched Img", stitched_img)
        cv2.waitKey(0)

        stitched_img = cv2.copyMakeBorder(stitched_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))
        gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)
        thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

        #cv2.imshow("Threshold Image", thresh_img)
       #cv2.waitKey(0)

        contours = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        areaOI = max(contours, key=cv2.contourArea)

        mask = np.zeros(thresh_img.shape, dtype="uint8")
        x, y, w, h = cv2.boundingRect(areaOI)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

        minRectangle = mask.copy()
        sub = mask.copy()

        while cv2.countNonZero(sub) > 0:
            minRectangle = cv2.erode(minRectangle, None)
            sub = cv2.subtract(minRectangle, thresh_img)

        contours = cv2.findContours(minRectangle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        areaOI = max(contours, key=cv2.contourArea)

        #cv2.imshow("minRectangle Image", minRectangle)
        #cv2.waitKey(0)

        x, y, w, h = cv2.boundingRect(areaOI)
        stitched_img_processed = stitched_img[y:y + h, x:x + w]

        # Save the final stitched image with the same extension as the original images
        base_name = os.path.basename(image_paths[0])
        output_path = f"stitchedOutputProcessed_{os.path.splitext(base_name)[1][1:]}.png"
        cv2.imwrite(output_path, stitched_img_processed)

        cv2.imshow("Stitched Image Processed", stitched_img_processed)
        cv2.waitKey(0)

        cv2.destroyAllWindows()
    else:
        print("Images could not be stitched!")
        print("Likely not enough keypoints being detected!")

# Create the main window
root = tk.Tk()
root.geometry("350x100") #size of window
root.title("Image Stitching Application")

# Create and pack the widgets
select_folder_button = tk.Button(root, text="Select Image Folder", command=select_folder)
select_folder_button.pack(pady=10)

# Start the main loop
root.mainloop()
