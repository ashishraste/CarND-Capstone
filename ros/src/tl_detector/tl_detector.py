#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
from scipy.spatial import KDTree

from common_tools.helper import Helper

STATE_COUNT_THRESHOLD = 2  # Number of times for a light-state to be repeated before it is considered true. 
SIM_MODE = True

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        self.is_site = self.config["is_site"]
        rospy.loginfo("Is Site: {}".format(self.is_site))

        self.pose, self.waypoints, self.camera_image, self.waypoints_2d, \
            self.waypoints_tree = Helper.get_none_instances(5)

        self.lights = []

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier(self.is_site)
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.prev_light_loc = None
        self.has_image = False

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.loop_rate = 10  # in Hz. 
        self.loop()

    def loop(self):
        rate = rospy.Rate(self.loop_rate)
        while not rospy.is_shutdown():
            '''
            Publish upcoming red lights at camera frequency.
            Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
            of times before we start using it. Otherwise the previous stable state is
            used.
            '''
            if self.pose and self.waypoints and self.camera_image:
                light_wpt, state = self.process_traffic_lights()
                if self.state != state:
                    if self.state == TrafficLight.RED and state == TrafficLight.GREEN:
                        self.last_state = self.state
                        self.last_wp = -1
                        rospy.logdebug('Publishing Green light {}'.format(Int32(self.last_wp)))
                        self.upcoming_red_light_pub.publish(Int32(self.last_wp))
                    self.state_count = 0
                    self.state = state
                elif self.state_count >= STATE_COUNT_THRESHOLD-1:
                    light_wpt = light_wpt if state == TrafficLight.RED else -1
                    self.last_wp = light_wpt
                    self.upcoming_red_light_pub.publish(Int32(light_wpt))
                else:
                    self.upcoming_red_light_pub.publish(Int32(self.last_wp))
                self.state_count += 1
            rate.sleep()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] \
                for waypoint in waypoints.waypoints]
            self.waypoints_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint
        Args:
            msg (Image): image from car-mounted camera
        """
        self.has_image = True
        self.camera_image = msg

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            x: x-coordinate of the position to match a waypoint to
            y: y-coordinate of the position to match a waypoint to
        Returns:
            int: index of the closest waypoint in self.waypoints
        """
        return self.waypoints_tree.query([x, y], 1)[1]

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if not self.has_image:
            return TrafficLight.UNKNOWN
        
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        # Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color
        Returns:
            int: index of waypoint closest to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        closest_light = None
        line_wpt_index = None  # In the simulator, each light comes with a stop-line waypoint index

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']

        if self.pose and self.waypoints and self.waypoints_tree:
            car_wpt_index = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)
            diff = len(self.waypoints.waypoints)

            for i, light in enumerate(self.lights):
                # Get stop-line waypoint index
                line = stop_line_positions[i]
                tmp_wpt_index = self.get_closest_waypoint(line[0], line[1])
                d = tmp_wpt_index - car_wpt_index
                if 0 <= d < diff:
                    diff = d
                    closest_light = light
                    line_wpt_index = tmp_wpt_index

        if closest_light:
            state = self.get_light_state(closest_light)
            return line_wpt_index, state

        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
