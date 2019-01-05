from styx_msgs.msg import TrafficLight
import rospy
from keras.models import load_model
import numpy as np
from keras import backend as kbe
import yaml
import os

IMG_H = 600   # image height in pixels
IMG_W = 800  # image width in pixels
IMG_C = 3     # num of channels

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.configuration = yaml.load(rospy.get_param('/traffic_light_config'))
        #rospy.logwarn(os.getcwd())
        dir_path = os.path.dirname(os.path.realpath(__file__))
        if self.configuration['is_site']:
            self.model_dir_path =  os.path.join(dir_path, 'site_model.h5')
        else:
             self.model_dir_path =  os.path.join(dir_path, 'sim_model.h5')
             #rospy.logwarn(os.path.join(dir_path, 'sim_model.h5'))
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
        colors = [TrafficLight.GREEN, TrafficLight.RED, TrafficLight.YELLOW, TrafficLight.UNKNOWN]
        img = np.reshape (image,  (1, IMG_H, IMG_W, IMG_C))
        with self.graph.as_default():
            colorNum = np.argmax(self.model.predict(img))
            #print(str(colorNum))
        
        return colors[colorNum] # -1 is UNKNOWN
