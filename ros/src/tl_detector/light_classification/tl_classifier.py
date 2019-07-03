from styx_msgs.msg import TrafficLight
from yolo3.yolo import YOLO
import numpy as np
from PIL import Image
# from ssd_inception.model import Model
import cv2

class TLClassifier(object):
    def __init__(self, is_site, config):
        self.is_site = is_site
        self.site_use_yolo = False

        if is_site:
            self.model = YOLO(model_path=config["model_path"], \
                anchors_path=config["model_anchor"], classes_path=config["model_classes"])
            self.traffic_light_class_idx = 9  # Index of traffic-light class in YOLO output
        else:
            self.model = YOLO(model_path=config["model_path"], \
                anchors_path=config["model_anchor"], classes_path=config["model_classes"])
            self.traffic_light_class_idx = 9  # Index of traffic-light class in YOLO output

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if image is not None:
            image_data = np.asarray(image)

            if self.is_site:
                return TrafficLight.UNKNOWN
                # return self.model.detect_traffic_light(image)
            else:
                pil_img = Image.fromarray(image_data)
                out_boxes, out_scores, out_classes = self.model.detect_image(pil_img)
                detected_states = []
                for i, cls_idx in reversed(list(enumerate(out_classes))):
                    box = out_boxes[i]
                    score = out_scores[i]
                    # If the detected class is of type traffic-light
                    if cls_idx == self.traffic_light_class_idx:
                        top, left, bottom, right = box
                        top = max(0, np.floor(top+0.5).astype('int32'))
                        left = max(0, np.floor(left+0.5).astype('int32'))
                        bottom = min(pil_img.size[1], np.floor(bottom+0.5).astype('int32'))
                        right = min(pil_img.size[0], np.floor(right+0.5).astype('int32'))
                        cropped_light = image_data[top:bottom, left:right, :]
                        detected_states.append(self.classify_traffic_light_color(cropped_light))

                if len(detected_states) > 0:
                    (_, idx, counts) = np.unique(detected_states, return_index=True, return_counts=True)
                    index = idx[np.argmax(counts)]
                    return detected_states[index]
                else:
                    return TrafficLight.UNKNOWN

        return TrafficLight.UNKNOWN


    def classify_traffic_light_color(self, image):
        def binary_thresh(img, thresh=(0,255)):
            # Applies binary thresholding on an image.
            binary = np.zeros_like(img)
            binary[(img > thresh[0]) & (img <= thresh[1])] = 1
            return binary 
        
        red_channel = image[:, :, 0]
        green_channel = image[:, :, 1]
        blue_channel = image[:, :, 2]

        rows = image.shape[0]

        # Binary threshold on R-channel
        r_binary = binary_thresh(red_channel, (196, 255))
        r_binary_1 = r_binary[0:int(rows / 3) - 1, :]
        r_binary_2 = r_binary[int(rows / 3):(2 * int(rows / 3)) - 1, :]
        r_binary_3 = r_binary[(2 * int(rows / 3)):rows - 1, :]
        is_red = (r_binary_1.sum() > 2 * (r_binary_2.sum())) & (r_binary_1.sum() > 2 * (r_binary_3.sum()))

        # Binary threshold on G-channel
        g_binary = binary_thresh(green_channel, (183, 255))
        g_binary_1 = g_binary[0:int(rows / 3) - 1, :]
        g_binary_2 = g_binary[int(rows / 3):(2 * int(rows / 3)) - 1, :]
        g_binary_3 = g_binary[(2 * int(rows / 3)):rows - 1, :]
        is_green = (g_binary_3.sum() > 2 * (g_binary_2.sum())) & (g_binary_3.sum() > 2 * (g_binary_1.sum()))

        # Binary threshold on G-channel and B-channel to pick Yellow colors
        g_binary = binary_thresh(green_channel, (146, 255))
        b_binary = binary_thresh(blue_channel, (143, 255))
        y_binary = g_binary + b_binary
        
        y_binary_1 = y_binary[0:int(rows / 3) - 1, :]
        y_binary_2 = y_binary[int(rows / 3):(2 * int(rows / 3)) - 1, :]
        y_binary_3 = y_binary[(2 * int(rows / 3)):rows - 1, :]
        is_yellow = (y_binary_2.sum() > 2 * (y_binary_1.sum())) & (y_binary_2.sum() > 2 * (y_binary_3.sum()))

        if is_red:
            return TrafficLight.RED
        elif is_yellow:
            return TrafficLight.YELLOW
        elif is_green:
            return TrafficLight.GREEN
        else:
            return TrafficLight.UNKNOWN


        
