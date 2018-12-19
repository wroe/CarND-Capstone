from styx_msgs.msg import TrafficLight
import rospy
from keras.models import load_model

IMG_H = 600   # image height in pixels
IMG_W = 800  # image width in pixels
IMG_C = 3     # num of channels

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.model_dir_path =  './models/sim_model.h5'
        self.model = load_model(self.model_dir_path)
        self.model._make_predict_function()

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        colors = [TrafficLight.RED, TrafficLight.GREEN, TrafficLight.UNKNOWN]
        img = np.reshape (image,  (1, IMG_H, IMG_W, IMG_C))
        colorNum = np.argmax(self.model.predict(img))
        
        return colors[colorNum] # -1 is UNKNOWN
