'''
This is the class for the face detection model.
This is used to instantiate the face detection model downloaded from the Intel model zoo. 
Model is used to detect faces in Videos, Images or video streams
'''
from openvino.inference_engine import IECore, IENetwork
import numpy as np
import cv2
import time
from pathlib import Path

class Face_Detector:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model, weights, device, extensions=None):
        self.device = device
        self.extensions = extensions
        self.model = model
        self.weights = weights
        self.plugin = None
        self.network = None
        self.input_shape = None
        self.inputBlob = None
        self.output_shape = None
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


    def predict(self, image, threshold):
       img_processed = self.preprocess_input(image)
       outputs = self.execNetwork.infer({self.inputBlob : img_processed})
       coords = self.preprocess_output(outputs, threshold)

       coords = coords[0]
       height = image.shape[0]
       width = image.shape[1]
       coords = coords* np.array([width, height, width, height])
       coords = coords.astype(np.int32)

       crop_face = image[coords[1]:coords[3], coords[0]:coords[2]]
       return crop_face, coords

    def preprocess_input(self, image):
        img_resize = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        img_process = np.transpose(np.expand_dims(img_resize, axis = 0), (0,3,1,2))

        return img_process

    def preprocess_output(self, outputs, threshold):
        coords = []
        outputs = outputs[self.outputBlob][0][0]
        for output in outputs:
            if output[2] >= threshold:
                xmin = output[3]
                ymin = output[4]
                xmax = output[5]
                ymax = output[6]

                coords.append([xmin, ymin, xmax, ymax])

        return coords
   
