import time
import numpy as np
import os
import cv2
import pandas as pd

# Defining all the functions I use here

def erodeImage(image):
    kernel = np.ones((4, 4), np.uint8)
    erosion = cv2.erode(image, kernel, iterations=1)
    return erosion

def dilateImage(image):
    kernel = np.ones((3,3), np.unit8)
    dilate = cv2.dilate(image,kernel, iterations=1)
    return dilate

# removing high pixels - input single image

#def removeHighPixel(image):
#    im = np.copy(image)
#    im[im > 220] = 0
#    return im


# calculate total number of pixels above threshold

def pixelThreshold(image):
    image = np.asarray(image)
    val = image[image > 5]
    count = len(val)
    return count


# calculate centroid, Num of pixels > threshold for a single frame

def MotionParameters(diff_image):
#    high_pix = removeHighPixel(diff_image)
    eroded = erodeImage(diff_image)
    dilated = dilateImage(eroded)
    pixel = pixelThreshold(dilated)

    mask = cv2.inRange(dilated, 5, 200)   # extract contour and the centroid of the biggest contour
    kernel = np.ones((10, 10), np.uint8)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    _, contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    if (len(contours) >= 1) & np.any(areas):

        max_index = np.argmax(areas)    # Find the index of the largest contour
        contour_basic = contours[max_index]
        contour_hull = cv2.convexHull(contour_basic)

        M = cv2.moments(contour_basic)
        centroid_basic = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        M_hull = cv2.moments(contour_hull)
        centroid_hull = (int(M_hull["m10"] / M_hull["m00"]), int(M_hull["m01"] / M_hull["m00"]))

    else:
        centroid_basic = (np.nan, np.nan)
#        contour_basic = ()
        centroid_hull = (np.nan, np.nan)
#        contour_hull = ()

    return pixel, centroid_basic, centroid_hull


# make a list of all mp4 files
def listOfVideos(path): 
#path = 'C:/Users/tanvid2/Documents/FLowerMorph-LearningExpt/Videos'
# path = 'C:\\Users\\tanvid2\\Documents\\PyCharm Projects\\videoAnalysis\\testLoopVideos'
    video_files = [(os.path.join(root, name), name[0:-4])
                   for root, dirs, files in os.walk(path)
                   for name in files
                   if name.endswith('.mp4')]
    return video_files


# Run analysis through all files

#give path to videofiles
path = "C:/Users/tanvid2/Documents/FLowerMorph-LearningExpt/Videos"    
video_files = listOfVideos(path)
for file in video_files:
    path = file[0]
    name = file[1]
    
    with open('LogFile.txt', 'a') as log_text:
        log_text.write('\n' + str(time.asctime()) + '\t' + name + ' loaded' + '\n')
    print(str(time.asctime()) + '\t' + file[1] + ' loaded')

    t0 = time.time()     # log the start time

    # declare all variables
    num_pixel = []
    centroid_basic_x = []
    centroid_basic_y = []
    centroid_hull_x = []
    centroid_hull_y = []
#    contour_basic = []
#    contour_hull = []

    cam = cv2.VideoCapture(path)  # load video as object

    back_frame = 0  # calculate background to subtract from frame
    cam.set(1, back_frame)
    ret, f = cam.read(1)  # Read the image at the first frame
    if not ret:
        with open('LogFile.txt', 'a') as log_text:
            log_text.write('\n' + str(time.asctime()) + '\t' + name + ' first image not read' + '\n')
        print(str(time.asctime()) + '\t' + name + ' image not read')
    background = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

    # loop through all frames of the video
    total_frame = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_seq = list(range(0, total_frame))

    print('analyzing ' + str(total_frame) + ' frames for ' + name)

    for frame in frame_seq:
        cam.set(1, frame)  # start processing the current frame
        ret, f = cam.read(1)  # Read the image at that frame
        if not ret:
            with open('LogFile.txt', 'a') as log_text:
                log_text.write('\n' + str(time.asctime()) + '\t' + name + ' frame ' + frame + ' not read' + '\n')
            print(str(time.asctime()) + '\t' + name + ' frame ' + frame + ' not read')
        
        
        img = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        diff_im = cv2.subtract(img, background)
        
        pixel, cent_b, cent_hull = MotionParameters(diff_im)
        num_pixel.append(pixel)
        centroid_basic_x.append(cent_b[0])
        centroid_basic_y.append(cent_b[1])
        centroid_hull_x.append(cent_hull[0])
        centroid_hull_y.append(cent_hull[1])
#        contour_basic.append(cont_b)
#        contour_hull.append(cont_hull)

    new_path = 'C:/Users/tanvid2/Documents/FLowerMorph-LearningExpt/Output/'
    # new_path = 'C:/Users/tanvid2/Documents/PyCharm Projects/videoAnalysis/OutputData/'
    full_path = new_path + name

    # collect all the variables I want to save
    df1 = pd.DataFrame({'NumPixel': num_pixel})
    df2 = pd.DataFrame({'Centroid_basic_x': centroid_basic_x})
    df3 = pd.DataFrame({'Centroid_basic_y': centroid_basic_y})
    df4 = pd.DataFrame({'Centroid_hull_x': centroid_hull_x})
    df5 = pd.DataFrame({'Centroid_hull_y': centroid_hull_y})
#    df6 = pd.DataFrame({'Contour_basic': contour_basic})
#    df7 = pd.DataFrame({'Contour_hull': contour_hull})

    df_entire = pd.concat([df1, df2, df3, df4, df5], axis=1)
    df_entire.to_csv(full_path + '.csv')

    t1 = time.time()
    with open('LogFile.txt', 'a') as log_text:
        log_text.write('\t' + '\t' + '\t' + '\t' + name + '\t' + str(total_frame) + ' frames took ' + str(t1-t0) + ' seconds' + '\n')
        log_text.write(str(time.asctime()) + '\t' + name + ' done' + '\n')

    print(name + ' took ' + str(t1-t0) + ' seconds')
