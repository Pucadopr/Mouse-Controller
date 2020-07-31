import cv2 
import os
import numpy as np
import time
import logging as log
from argparse import ArgumentParser
from face_detection import Face_Detector
from input_feeder import InputFeeder
from head_pose_estimation import Head_Pose_Estimator
from mouse_controller import MouseController
from facial_landmarks_detection import Face_Landmark_Detector
from gaze_estimation import Gaze_Estimator
from pathlib import Path


def arg_parser():
    parser = ArgumentParser()

    parser.add_argument("-i", "--input", required= True, type=str, help="path to input file. options cam, image or video")
    parser.add_argument("-e", "--extension", required=False, type=str, default=None, help="path to cpu extension")
    parser.add_argument("-d", "--device", required=False, type=str, default="CPU", help="device to run model on, options: FPGA, GPU, MYRIAD, defaults to CPU")
    parser.add_argument("-v", "--visualization", required=False, nargs='+', default=[], help="specify to view visualization from the other models. face for face detection, facel for face landmark, head for head pose, gaze for gaze estimation. for visualization of more than one model, seperate with a space")

    return parser

def main():
    args =arg_parser().parse_args()
    input_file = args.input
    visual = args.visualization

    if input_file == "cam":
        input_feeder = InputFeeder("cam")
    
    elif input_file == "image":
        input_feeder = InputFeeder("image", input_file)

    elif not input_file:
        log.error("Input file not found")
        exit(1)
    
    else:
        input_feeder = InputFeeder("video", input_file)

    face_d = Face_Detector("../intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml", "../intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.bin", args.device, args.extension)
    face_l = Face_Landmark_Detector("../intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml", "../intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.bin", args.device, args.extension)
    gaze = Gaze_Estimator("../intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml", "../intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.bin", args.device, args.extension)
    head = Head_Pose_Estimator("../intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml", "../intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.bin", args.device, args.extension)

    mouse_control = MouseController('medium', 'fast')

    input_feeder.load_data()

    face_d.load_model()
    face_l.load_model()
    gaze.load_model()
    head.load_model()

    count = 0
    f_count = 0
    inf_time = 0

    for _, frame in input_feeder.next_batch():
        if not _:
            break;

        if frame is not None:
            f_count += 1
            if f_count%5 == 0:
                cv2.imshow('video', cv2.resize(frame, (500, 500)))      
            
            key = cv2.waitKey(60)

            crop_face, face_coords = face_d.predict(frame, 0.5)
            if isinstance(crop_face, int):
                log.info("No face in frame")
                if key == 27:
                    break
                continue
            
            head_pose = head.predict(crop_face)
            le_eye, ri_eye, eye_coords = face_l.predict(crop_face)   
            new_mouse_coord, gaze_vector = gaze.predict(le_eye, ri_eye, head_pose)
            count = count + 1
            if (not len(visual) == 0):
                preview_window = frame.copy()
                
                if 'face' in visual:
                    if len(visual) != 1:
                        preview_window = crop_face
                    else:
                        cv2.rectangle(preview_window, (face_coords[0], face_coords[1]), (face_coords[2], face_coords[3]), (0, 150, 0), 3)

                if 'facel' in visual:
                    if not 'face' in visual:
                        preview_window = crop_face.copy()

                    cv2.rectangle(preview_window, (eye_coords[0][0] - 10, eye_coords[0][1] - 10), (eye_coords[0][2] + 10, eye_coords[0][3] + 10), (0,255,0), 3)
                    cv2.rectangle(preview_window, (eye_coords[1][0] - 10, eye_coords[1][1] - 10), (eye_coords[1][2] + 10, eye_coords[1][3] + 10), (0,255,0), 3)
                    
                if 'head' in visual:
                    cv2.putText(
                        preview_window, 
                        "Pose Angles: yaw:{:.2f} | pitch:{:.2f} | roll:{:.2f}".format(head_pose[0], head_pose[1], head_pose[2]), (50, 50), 
                        cv2.FONT_HERSHEY_COMPLEX, 
                        1, 
                        (0, 255, 0), 
                        1, 
                        cv2.LINE_AA
                    )

                if 'gaze' in visual:
                    if not 'face' in visual:
                        preview_window = crop_face.copy()

                    x, y, w = int(gaze_vector[0] * 12), int(gaze_vector[1] * 12), 160
                    
                    le = cv2.line(le_eye, (x - w, y - w), (x + w, y + w), (255, 0, 255), 2)
                    cv2.line(le, (x - w, y + w), (x + w, y - w), (255, 0, 255), 2)
                    
                    re = cv2.line(ri_eye, (x - w, y - w), (x + w, y + w), (255, 0, 255), 2)
                    cv2.line(re, (x - w, y + w), (x + w, y - w), (255, 0, 255), 2)
                    
                    preview_window[eye_coords[0][1]:eye_coords[0][3], eye_coords[0][0]:eye_coords[0][2]] = le
                    preview_window[eye_coords[1][1]:eye_coords[1][3], eye_coords[1][0]:eye_coords[1][2]] = re
            
            if len(visual) != 0:
                img_h = np.hstack((cv2.resize(frame, (500, 500)), cv2.resize(preview_window, (500, 500))))
            else:
                img_h = cv2.resize(frame, (500, 500))

            cv2.imshow('Visualization', img_h)

            if f_count%5 == 0:                       
                mouse_control.move(new_mouse_coord[0], new_mouse_coord[1])    
            
            if key == 27:
                break
    
    log.info("End of session.")
    cv2.destroyAllWindows()
    input_feeder.close()
        
if __name__ == '__main__':
    main()
