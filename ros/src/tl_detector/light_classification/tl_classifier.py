from styx_msgs.msg import TrafficLight
import rospy
from keras.models import load_model
import numpy as np
from keras import backend as kbe

IMG_H = 600   # image height in pixels
IMG_W = 800  # image width in pixels
IMG_C = 3     # num of channels

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.model_dir_path =  '/home/workspace/CarND-Capstone/ros/src/tl_detector/light_classification/models/sim_model.h5'
        self.model = load_model(self.model_dir_path)
        self.model._make_predict_function()
        self.graph = kbe.tf.get_default_graph()

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        colors = [TrafficLight.RED, TrafficLight.GREEN, TrafficLight.YELLOW, TrafficLight.UNKNOWN]
        img = np.reshape (image,  (1, IMG_H, IMG_W, IMG_C))
        with self.graph.as_default():
            colorNum = np.argmax(self.model.predict(img))
        
        return colors[colorNum] # -1 is UNKNOWN
