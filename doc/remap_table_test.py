import os
import numpy as np
import cv2 as cv


with open('./CalibrationData/CalibResults/CalibResults/surround-view-system/front_table.txt', 'r') as f:
    table_lines = f.readlines()

x_list = []
y_list = []

for lines in table_lines:
    x = float(lines.rstrip().split(',')[0][1:])
    y = float(lines.rstrip().split(',')[1][:-1])

    x_list.append(x)
    y_list.append(y)

map_x = np.array(x_list)[np.newaxis, :].reshape((416, 416)).astype(np.float32)
map_y = np.array(y_list)[np.newaxis, :].reshape((416, 416)).astype(np.float32)

omni_im = cv.imread('/media/silencht/c26c9997-3b3d-4de6-87c5-973b67e0c477/BeVIS/CalibrationData/extrinCalib-1/Surround-view System/surround-view system/1609492897.0_Front.jpg')
top_front = cv.remap(omni_im, map_x, map_y, interpolation=cv.INTER_LINEAR)

cv.imshow('top front', top_front)
cv.imshow('origin', omni_im)
cv.waitKey(0)

with open('./CalibrationData/CalibResults/CalibResults/surround-view-system/back_table.txt', 'r') as f:
    table_lines = f.readlines()

x_list = []
y_list = []

for lines in table_lines:
    x = float(lines.rstrip().split(',')[0][1:])
    y = float(lines.rstrip().split(',')[1][:-1])

    x_list.append(x)
    y_list.append(y)

map_x = np.array(x_list)[np.newaxis, :].reshape((416, 416)).astype(np.float32)
map_y = np.array(y_list)[np.newaxis, :].reshape((416, 416)).astype(np.float32)

omni_im = cv.imread('/media/silencht/c26c9997-3b3d-4de6-87c5-973b67e0c477/BeVIS/CalibrationData/extrinCalib-1/Surround-view System/surround-view system/1609492897.0_Back.jpg')
top_front = cv.remap(omni_im, map_x, map_y, interpolation=cv.INTER_LINEAR)

cv.imshow('top front', top_front)
cv.imshow('origin', omni_im)
cv.waitKey(0)

with open('./CalibrationData/CalibResults/CalibResults/surround-view-system/right_table.txt', 'r') as f:
    table_lines = f.readlines()

x_list = []
y_list = []

for lines in table_lines:
    x = float(lines.rstrip().split(',')[0][1:])
    y = float(lines.rstrip().split(',')[1][:-1])

    x_list.append(x)
    y_list.append(y)

map_x = np.array(x_list)[np.newaxis, :].reshape((416, 416)).astype(np.float32)
map_y = np.array(y_list)[np.newaxis, :].reshape((416, 416)).astype(np.float32)

omni_im = cv.imread('/media/silencht/c26c9997-3b3d-4de6-87c5-973b67e0c477/BeVIS/CalibrationData/extrinCalib-1/Surround-view System/surround-view system/1609492897.0_Right.jpg')
top_front = cv.remap(omni_im, map_x, map_y, interpolation=cv.INTER_LINEAR)

cv.imshow('top front', top_front)
cv.imshow('origin', omni_im)
cv.waitKey(0)

with open('./CalibrationData/CalibResults/CalibResults/surround-view-system/left_table.txt', 'r') as f:
    table_lines = f.readlines()

x_list = []
y_list = []

for lines in table_lines:
    x = float(lines.rstrip().split(',')[0][1:])
    y = float(lines.rstrip().split(',')[1][:-1])

    x_list.append(x)
    y_list.append(y)

map_x = np.array(x_list)[np.newaxis, :].reshape((416, 416)).astype(np.float32)
map_y = np.array(y_list)[np.newaxis, :].reshape((416, 416)).astype(np.float32)

omni_im = cv.imread('/media/silencht/c26c9997-3b3d-4de6-87c5-973b67e0c477/BeVIS/CalibrationData/extrinCalib-1/Surround-view System/surround-view system/1609492897.0_Left.jpg')
top_front = cv.remap(omni_im, map_x, map_y, interpolation=cv.INTER_LINEAR)

cv.imshow('top front', top_front)
cv.imshow('origin', omni_im)
cv.waitKey(0)

