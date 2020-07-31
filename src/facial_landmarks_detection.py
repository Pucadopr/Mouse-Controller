'''
This is the facial landmark detection model. 
Model is used to locate the eyes of the person in frame.
'''
from model import Model

class Face_Landmark_Detector(Model):
    '''
    Class for Facial landmark detection model.
    '''

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
    