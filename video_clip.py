import cv2
import numpy as np
import os


l = [51, 63, 73, 74, 84, 90, 95, 96]
arr = [i for i in np.arange(85, 99) if i not in l]

for a in arr:
    #root = '/home/scifilab/work/rutgers-cornell-cmu/data/INAGT_clean/m0'+str(a)+'/'
    root = '/Volumes/Samsung_T5/INAGT_clean/m026/'
    path = root+'CLIPS/6_9_annotation_quad/'
    face_path = root+'face/'
    road_path = root+'road/'
    side_path = root+'side/'
    back_path = root+'back/'

    video_list = [v for v in os.listdir(path) if v.startswith('m')]

    for v in video_list:

        cameraCapture = cv2.VideoCapture(path+v)
        width = int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cameraCapture.get(cv2.CAP_PROP_FPS)


        videoWriterLeftUp = cv2.VideoWriter(face_path+v+'_face.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width//2, height//2))
        print(face_path+v+'_face.mp4')
        videoWriterLeftDown = cv2.VideoWriter(side_path+v+'_side.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width//2, height//2))
        videoWriterRightUp = cv2.VideoWriter(road_path+v+'_road.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width//2, height//2))
        videoWriterRightDown = cv2.VideoWriter(back_path+v+'_back.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width//2, height//2))

        success, frame = cameraCapture.read()
        while success:
            frameLeftUp = frame[0:height // 2, 0:width // 2, :]
            videoWriterLeftUp.write(frameLeftUp)

            frameLeftDown = frame[height // 2:height, 0:width // 2, :]
            videoWriterLeftDown.write(frameLeftDown)

            frameRightUp = frame[0:height // 2, width // 2:width, :]
            videoWriterRightUp.write(frameRightUp)

            frameRightDown = frame[height // 2:height, width // 2:width, :]
            videoWriterRightDown.write(frameRightDown)

            success, frame = cameraCapture.read()

        cameraCapture.release()
        videoWriterLeftUp.release()
        videoWriterLeftDown.release()
        videoWriterRightUp.release()
        videoWriterRightDown.release()