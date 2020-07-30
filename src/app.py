import cv2 
import os
import numpy as np
import time
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

    return parser

def main():
    args =arg_parser().parse_args()
    input_file = args.input

    if input_file == "cam":
        input_feeder = InputFeeder("cam")
    
    elif input_file == "image":
        input_feeder = InputFeeder("image", input_file)

    else:
        input_feeder = InputFeeder("video", input_file)

    face_d = Face_Detector("../intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml", "../models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.bin", args.device, args.extension)
    face_l = Face_Landmark_Detector("../intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml", "../models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.bin", args.device, args.extension)
    gaze = Gaze_Estimator("../intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml", "../models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.bin", args.device, args.extension)
    head = Head_Pose_Estimator("../intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml", "../models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.bin", args.device, args.extension)

    mouse_control = MouseController('medium', 'fast')

    input_feeder.load_data()

    face_d.load_model()
    face_l.load_model()
    gaze.load_model()
    head.load_model()

    for _, frame in input_feeder.next_batch():
        if not _:
            break;

        if frame is not None:
            cv2.imshow('video', cv2.resize(frame, (500, 500)))      
            key = cv2.waitKey(33)

            crop_face, face_coords = face_d.predict(frame, 0.5)
            if isinstance(crop_face, int):
                return print("No face in frame")
                if key == 27:
                    break
                continue
            
            head_pose = head.predict(crop_face)
            le_eye, ri_eye, eye_coords = face_l.predict(crop_face)   
            new_mouse_coord, gaze_vector = gaze.predict(le_eye, ri_eye, head_pose)
            
            mouse_control.move(new_mouse_coord[0], new_mouse_coord[1])    
            
            if key == 27:
                break

    cv2.destroyAllWindows()
    input_feeder.close()
        
if __name__ == '__main__':
    main()
