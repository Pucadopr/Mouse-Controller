'''
This is the class for the face detection model.
This is used to instantiate the face detection model downloaded from the Intel model zoo. 
Model is used to detect faces in Videos, Images or video streams
'''
from model import Model
import cv2 
import numpy as np

class Face_Detector(Model):
    '''
    Class for the Face Detection Model.
    '''
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
   
