import json
import numpy as np
import pandas as pd
import os

path = "/Volumes/Samsung_T5/inagt_response_dataset/"
path_list = os.listdir(path)
path_list.sort()
body_op_x, lhand_op_x, rhand_op_x, face_op_x = [], [], [], []
body_op_y, lhand_op_y, rhand_op_y, face_op_y = [], [], [], []
body_op_c, lhand_op_c, rhand_op_c, face_op_c = [], [], [], []
for driver in path_list:
    event_list = os.listdir(path + driver)
    event_list.sort()
    for event in event_list:
        body_pose_frames = os.listdir(path + driver + '/' + event + '/body_openpose')  # 120 frames (6s + 6s)
        face_pose_frames = os.listdir(path + driver + '/' + event + '/face_openpose')  # 120 frames
        body_pose_frames.sort()
        face_pose_frames.sort()

        b_x, b_y, b_c = [], [], []
        lh_x, lh_y, lh_c = [], [], []
        rh_x, rh_y, rh_c = [], [], []
        f_x, f_y, f_c = [], [], []

        for i in range(60):
            # select 6s before asking
            with open(path + driver + '/' + event + '/body_openpose/' + body_pose_frames[i]) as f:
                body_pose_frame = json.load(f)
            with open(path + driver + '/' + event + '/face_openpose/' + face_pose_frames[i]) as file:
                face_pose_frame = json.load(file)

            if not body_pose_frame['people']:   # if no body detected:
                body_joints = np.empty(25*3)
                body_joints[:] = np.nan
                lhand_joints = np.empty(21*3)
                lhand_joints[:] = np.nan
                rhand_joints = np.empty(21*3)
                rhand_joints[:] = np.nan
            else:
                body_joints = body_pose_frame['people'][0]['pose_keypoints_2d']
                lhand_joints = body_pose_frame['people'][0]['hand_left_keypoints_2d']
                rhand_joints = body_pose_frame['people'][0]['hand_right_keypoints_2d']

            if not face_pose_frame['people']:   # if no face detected
                face_joints = np.empty(70*3)
                face_joints[:] = np.nan
            else:
                face_joints = face_pose_frame['people'][0]['face_keypoints_2d']

            a = 0
            while a < (len(body_joints)):
                b_x.append(body_joints[a])
                b_y.append(body_joints[a+1])
                b_c.append(body_joints[a+1])
                a += 3
            a = 0
            while a < (len(lhand_joints)):
                lh_x.append(lhand_joints[a])
                lh_y.append(lhand_joints[a + 1])
                lh_c.append(lhand_joints[a + 1])
                a += 3
            a = 0
            while a < (len(rhand_joints)):
                rh_x.append(rhand_joints[a])
                rh_y.append(rhand_joints[a + 1])
                rh_c.append(rhand_joints[a + 1])
                a += 3
            a = 0
            while a < (len(face_joints)):
                f_x.append(face_joints[a])
                f_y.append(face_joints[a + 1])
                f_c.append(face_joints[a + 1])
                a += 3

        body_op_x.append(np.array(b_x).reshape((60, -1)))
        body_op_y.append(np.array(b_y).reshape((60, -1)))
        body_op_c.append(np.array(b_c).reshape((60, -1)))
        lhand_op_x.append(np.array(lh_x).reshape((60, -1)))
        lhand_op_y.append(np.array(lh_y).reshape((60, -1)))
        lhand_op_c.append(np.array(lh_c).reshape((60, -1)))
        rhand_op_x.append(np.array(rh_x).reshape((60, -1)))
        rhand_op_y.append(np.array(rh_y).reshape((60, -1)))
        rhand_op_c.append(np.array(rh_c).reshape((60, -1)))
        face_op_x.append(np.array(f_x).reshape((60, -1)))
        face_op_y.append(np.array(f_y).reshape((60, -1)))
        face_op_c.append(np.array(f_c).reshape((60, -1)))

np.save('../data/body_op_x', np.array(body_op_x).astype(float))
np.save('../data/body_op_y', np.array(body_op_y).astype(float))
np.save('../data/body_op_c', np.array(body_op_c).astype(float))
np.save('../data/lhand_op_x', np.array(lhand_op_x).astype(float))
np.save('../data/lhand_op_y', np.array(lhand_op_y).astype(float))
np.save('../data/lhand_op_c', np.array(lhand_op_c).astype(float))
np.save('../data/rhand_op_x', np.array(rhand_op_x).astype(float))
np.save('../data/rhand_op_y', np.array(rhand_op_y).astype(float))
np.save('../data/rhand_op_c', np.array(rhand_op_c).astype(float))
np.save('../data/face_op_x', np.array(face_op_x).astype(float))
np.save('../data/face_op_y', np.array(face_op_y).astype(float))
np.save('../data/face_op_c', np.array(face_op_c).astype(float))
print()
print()