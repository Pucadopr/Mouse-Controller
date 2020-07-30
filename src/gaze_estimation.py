'''
This is the class for the gaze estimator model. 
Results from the other models are used to estimate where the person is looking.
'''
from openvino.inference_engine import IECore, IENetwork
import numpy as np
import cv2
import math

class Gaze_Estimator:
    '''
    Class for the Gaze Estimator model.
    '''
    def __init__(self, model, weights, device, extensions=None):
        self.device = device
        self.extensions = extensions
        self.model = model
        self.weights = weights
        self.plugin = None
        self.network = None
        self.input = None
        self.inputBlob = None
        self.output = None
        self.outputBlob = None
        self.execNetwork = None
        self.inferRequest = None

    def load_model(self):
        self.plugin = IECore()

        if self.extensions:
            self.plugin.add_extension(self.extensions, self.device)

        self.network = self.plugin.read_network(model=self.model, weights=self.weights)
        supported_layers = self.plugin.query_network(self.network, self.device)
        unsupported_layers = []

        for layer in self.network.layers.keys():
            if layer not in supported_layers:
                unsupported_layers.append(layer)

        if len(unsupported_layers) != 0:
            return print('Please add Extension as some unsupported layers currently exist')

        self.execNetwork = self.plugin.load_network(self.network, self.device, num_requests = 1)
        self.inputBlob = [i for i in self.network.inputs.keys()]
        self.outputBlob = [i for i in self.network.outputs.keys()]
        self.input_shape = self.network.inputs[self.inputBlob[1]].shape

    def predict(self, left_eye_image, right_eye_image, head_pose_angle):
        le_img_processed, re_img_processed = self.preprocess_input(left_eye_image, right_eye_image)
        outputs = self.execNetwork.infer({'head_pose_angles' : head_pose_angle, 'left_eye_image':le_img_processed, 'right_eye_image':re_img_processed})
        mouse_coords, gaze_vec = self.preprocess_output(outputs, head_pose_angle)

        return mouse_coords, gaze_vec

    def preprocess_input(self, left_eye_image, right_eye_image):
        le_img_resized = cv2.resize(left_eye_image, (self.input_shape[3], self.input_shape[2]))
        le_img_processed = np.transpose(np.expand_dims(le_img_resized, axis=0), (0,3,1,2))

        re_img_resized = cv2.resize(right_eye_image, (self.input_shape[3], self.input_shape[2]))
        re_img_processed = np.transpose(np.expand_dims(re_img_resized, axis=0), (0,3,1,2))
        
        return le_img_processed, re_img_processed
    
    def preprocess_output(self, outputs, head_pose_angle):
        gaze_vec = outputs[self.outputBlob[0]].tolist()[0]
        angle_r_fc = head_pose_angle[2]
        cosine = math.cos(angle_r_fc * math.pi/180)
        sine = math.sin(angle_r_fc * math.pi/180)

        x_val = gaze_vec[0] * cosine + gaze_vec[1]* sine
        y_val = gaze_vec[0] * sine + gaze_vec[1]* sine
        
        return (x_val, y_val), gaze_vec
        