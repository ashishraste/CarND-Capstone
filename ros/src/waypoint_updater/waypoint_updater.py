#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
import numpy as np
import math
from std_msgs.msg import Int32

from common_tools.helper import Helper

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 50 # Number of waypoints we will publish. You can change this number
MAX_DECEL = 5.0
BRAKING_DECEL = 1 / LOOKAHEAD_WPS  # Deceleration for braking
NOSE_TO_CENTRE_OFFSET = 4  # Number of waypoints that approximately amounts to distance between the nose and the centre of the car.

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)   

        self.base_waypoints, self.pose, self.waypoints_2d, self.waypoints_tree = Helper.get_none_instances(4)
        self.stopline_wpt_idx = -1

        self.loop()

    def loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints and self.waypoints_tree:
                # Get next closest waypoint index
                closest_wpt_index = self.get_closest_waypoint_index()
                self.publish_waypoints(closest_wpt_index)
            rate.sleep()

    def get_closest_waypoint_index(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_index = self.waypoints_tree.query([x, y], 1)[1]

        ## Check if the waypoint at closest-index is ahead of or behind the vehicle
        prev_coord = self.waypoints_2d[closest_index-1]
        closest_coord = self.waypoints_2d[closest_index]

        # Heading vectors between [current-position, closest-position] and
        # [closest-position, previous-position] if along the same direction
        # give us a positive dot-product, meaning the closest-position is behind the vehicle.
        closest_pos = np.array(closest_coord)
        prev_pos = np.array(prev_coord)
        cur_pos = np.array([x, y])

        dir_alignment = np.dot(closest_pos - prev_pos, cur_pos - closest_pos)
        if dir_alignment > 0:
            # Closest waypoint is behind the vehicle's current waypoint, so increment the closest-index
            closest_index = (closest_index + 1) % len(self.waypoints_2d)
        
        return closest_index

    def publish_waypoints(self, closest_idx):
        final_lane = self.generate_lane(closest_idx)
        self.final_waypoints_pub.publish(final_lane)

    def generate_lane(self, closest_idx):
        lane_msg = Lane()
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        base_waypoints = self.base_waypoints.waypoints[closest_idx : farthest_idx]

        # If traffic_waypoint index is not set, continue with base-waypoints
        if self.stopline_wpt_idx == -1 or (self.stopline_wpt_idx >= farthest_idx):
            lane_msg.waypoints = base_waypoints
        else:
            lane_msg.waypoints = self.get_decelerating_waypoints(closest_idx, base_waypoints)

        return lane_msg

    def get_decelerating_waypoints(self, closest_idx, base_waypoints):
        decel_wpts = []
        for i, wpt in enumerate(base_waypoints):
            pt = Waypoint()
            pt.pose = wpt.pose

            stop_idx = max(self.stopline_wpt_idx - closest_idx - NOSE_TO_CENTRE_OFFSET, 0)
            dist = self.distance(base_waypoints, i, stop_idx)
            vel = math.sqrt(2 * MAX_DECEL * dist) + (i * BRAKING_DECEL)
            if vel < 1.:
                vel = 0.
            
            pt.twist.twist.linear.x = min(vel, wpt.twist.twist.linear.x)
            decel_wpts.append(pt)
        
        return decel_wpts


    def pose_cb(self, pose_stamped_msg):
        self.pose = pose_stamped_msg

    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints
        if self.waypoints_2d is None:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y]
             for waypoint in waypoints.waypoints]
            self.waypoints_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.stopline_wpt_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
