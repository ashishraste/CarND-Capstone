#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
import numpy as np
import math

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

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)   
        # nones = lambda x: [None for i in range(x)]    
        self.base_waypoints, self.pose, self.waypoints_2d, self.waypoints_tree = Helper.get_none_instances(4)
    
        self.loop()

    def loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints:
                # Get next closest waypoint index
                closest_wpt_idx = self.get_closest_waypoint_index()
                self.publish_waypoints(closest_wpt_idx)
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

    def publish_waypoints(self, closest_index):
        lane_msg = Lane()
        lane_msg.header = self.base_waypoints.header
        lane_msg.waypoints = self.base_waypoints.waypoints[closest_index : closest_index+LOOKAHEAD_WPS]

        self.final_waypoints_pub.publish(lane_msg)

    def pose_cb(self, pose_stamped_msg):
        self.pose = pose_stamped_msg

    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints
        if self.waypoints_2d is None:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y]
             for waypoint in waypoints.waypoints]
            self.waypoints_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

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
