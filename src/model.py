'''
This is the class for the face detection model.
This is used to instantiate the face detection model downloaded from the Intel model zoo. 
Model is used to detect faces in Videos, Images or video streams
'''
from openvino.inference_engine import IECore
import logging as log

class Model:
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
            log.info('Please add Extension as some unsupported layers currently exist')

        self.execNetwork = self.plugin.load_network(self.network, self.device)
        self.inputBlob = next(iter(self.network.inputs))
        self.outputBlob = next(iter(self.network.outputs))
        self.input_shape = self.network.inputs[self.inputBlob].shape
        self.output_shape = self.network.outputs[self.outputBlob].shape
