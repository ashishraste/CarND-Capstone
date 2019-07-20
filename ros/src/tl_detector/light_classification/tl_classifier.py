from styx_msgs.msg import TrafficLight
import tensorflow as tf
from numpy import expand_dims, squeeze, int32
from datetime import datetime
from rospy import logdebug, loginfo, logwarn

CLASS_LABELS = {1: 'Green', 2: 'Red', 3: 'Yellow', 4: 'Unknown'}

class TLClassifier(object):
    def __init__(self, is_site):
        if not is_site:
            FROZEN_GRAPH_PATH = r'light_classification/model/ssd_v2_sim/frozen_inference_graph.pb'
        else:
            FROZEN_GRAPH_PATH = r'light_classification/model/ssd_v2_site/frozen_inference_graph.pb'
        self.graph = tf.Graph()
        self.classification_threshold = 0.5

        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(FROZEN_GRAPH_PATH, 'rb') as fid:
                od_graph_def.ParseFromString(fid.read())
                tf.import_graph_def(od_graph_def, name='')

            self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            self.boxes = self.graph.get_tensor_by_name('detection_boxes:0')
            self.scores = self.graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.graph.get_tensor_by_name('num_detections:0')

        self.session = tf.Session(graph=self.graph)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        with self.graph.as_default():
            img_expand = expand_dims(image, axis=0)
            start = datetime.now()
            (boxes, scores, classes, num_detections) = self.session.run(
                [self.boxes, self.scores, self.classes, self.num_detections],
                feed_dict={self.image_tensor: img_expand})
            end = datetime.now()
            inference_time = end - start
            logdebug('Inference time: {} seconds'.format(inference_time))

        boxes = squeeze(boxes)
        scores = squeeze(scores)
        classes = squeeze(classes).astype(int32)

        pred_score = scores[0]
        pred_class = classes[0]

        logwarn('Class: {}, Score: {}'.format(CLASS_LABELS[pred_class], pred_score))

        if pred_score > self.classification_threshold:
            if pred_class == 1:
                loginfo('Traffic light detected: {}'.format(CLASS_LABELS[1]))
                return TrafficLight.GREEN
            elif pred_class == 2:
                loginfo('Traffic light detected: {}'.format(CLASS_LABELS[2]))
                return TrafficLight.RED
            elif pred_class == 3:
                loginfo('Traffic light detected: {}'.format(CLASS_LABELS[3]))
                return TrafficLight.YELLOW

        return TrafficLight.UNKNOWN
