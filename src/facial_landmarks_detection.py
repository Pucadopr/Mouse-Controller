'''
This is the facial landmark detection model. 
Model is used to locate the eyes of the person in frame.
'''
from openvino.inference_engine import IECore, IENetwork
import numpy as np
import cv2

class Face_Landmark_Detector:
    '''
    Class for Facial landmark detection model.
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
       img_processed = self.preprocess_input(image)
       outputs = self.execNetwork.infer({self.inputBlob : img_processed})
       coords = self.preprocess_output(outputs)

       coords = coords[0]
       height = image.shape[0]
       width = image.shape[1]
       coords = coords* np.array([width, height, width, height])
       coords = coords.astype(np.int32)

       le_xmin = coords[0] -10
       le_ymin = coords[1] -10
       le_xmax = coords[0] +10
       le_ymax = coords[1] +10

       re_xmin = coords[2]-10
       re_ymin = coords[3]-10
       re_xmax = coords[2]+10
       re_ymax = coords[3]+10

       le = image[le_ymin:le_ymax, le_xmin:le_xmax]
       re = image[re_ymin:re_ymax, re_xmin:re_xmax]

       eye_coords = [[le_xmin, le_ymin, le_xmax, le_ymin], [re_xmin, re_ymin, re_xmax, re_ymax]]

       return le, re, eye_coords

    def preprocess_input(self, image):
        img_cvt = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_cvt, (self.input_shape[3], self.input_shape[2]))
        img_processed = np.transpose(np.expand_dims(img_resized, axis=0), (0, 3, 1, 2))

        return img_processed    

    def preprocess_output(self, outputs):
        outs= outputs[self.outputBlob][0]
        leeye_x = outs[0].tolist()[0][0]
        leeye_y = outs[1].tolist()[0][0]
        rieye_x = outs[2].tolist()[0][0]
        rieye_y = outs[3].tolist()[0][0]

        return (leeye_x, leeye_y, rieye_x, rieye_y)
    