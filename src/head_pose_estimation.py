'''
This is the head pose estimator model. 
Model is used to detect the pose of the head of person in frame.
'''
from openvino.inference_engine import IECore, IENetwork
import numpy as np
import cv2

class Head_Pose_Estimator:
    '''
    Class for the Head pose model.
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

        self.execNetwork = self.plugin.load_network(self.network, self.device)
        self.inputBlob = next(iter(self.network.inputs))
        self.outputBlob = next(iter(self.network.outputs))
        self.input_shape = self.network.inputs[self.inputBlob].shape
        self.output_shape = self.network.outputs[self.outputBlob].shape

    def predict(self, image):
        img_processed = self.preprocess_input(image.copy())
        outputs = self.execNetwork.infer({self.inputBlob : img_processed})
        result = self.preprocess_output(outputs)

        return result

    def preprocess_input(self, image):
        img_resized = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        img_processed = np.transpose(np.expand_dims(img_resized, axis=0), (0,3,1,2))

        return img_processed
    
    def preprocess_output(self, outputs):
        y_fc = outputs['angle_y_fc'][0][0]
        p_fc = outputs['angle_p_fc'][0][0]
        r_fc =outputs['angle_r_fc'][0][0]

        output= [y_fc, p_fc, r_fc]

        return output
    