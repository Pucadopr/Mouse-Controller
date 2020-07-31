'''
This is the head pose estimator model. 
Model is used to detect the pose of the head of person in frame.
'''
from model import Model 

class Head_Pose_Estimator(Model):
    '''
    Class for the Head pose model.
    '''
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
    