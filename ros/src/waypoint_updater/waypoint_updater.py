#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
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

LOOKAHEAD_WPS = 100 # Number of waypoints we will publish. You can change this number
NOSE_TO_CENTRE_OFFSET = 2  # Number of waypoints that approximately amounts to distance between the nose and the centre of the car.
TRAFFIC_LIGHT_STOP_DIST = 15.  # Stopping distance to decelerate when nearing a traffic-light
MIN_ACCEL_SPEED_THRESH = 0.5  # Minimum speed to maintain while accelerating
ACCEL_RATE = 0.75  # in m/s

class VehicleState(object):
    Stop, Drive = range(2)

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)   

        self.base_waypoints, self.pose, self.waypoints_2d, self.waypoints_tree, \
            self.cur_velocity, self.prev_velocity, self.decel_rate, self.stopline_wpt_idx = Helper.get_none_instances(8)
        
        self.vehicle_state = VehicleState.Drive

        self.loop_rate = 10  # in Hz
        self.loop()

    def loop(self):
        rate = rospy.Rate(self.loop_rate)
        while not rospy.is_shutdown():
            if self.pose and self.waypoints_tree and self.cur_velocity:
                # Get next closest waypoint index
                closest_wpt_index = self.get_closest_waypoint_index()
                if closest_wpt_index is not None:
                    self.publish_waypoints(closest_wpt_index)
            self.prev_velocity = self.cur_velocity
            rate.sleep()

    def pose_cb(self, pose_stamped_msg):
        self.pose = pose_stamped_msg

    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints
        if self.waypoints_2d is None:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y]
             for waypoint in waypoints.waypoints]
            self.waypoints_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        if msg.data == -1:
            self.stopline_wpt_idx = None
        else:
            self.stopline_wpt_idx = msg.data

    def velocity_cb(self, msg):
        self.cur_velocity = msg.twist.linear.x
    
    def get_closest_waypoint_index(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_index = self.waypoints_tree.query([x, y], 1)[1]

        ## Check if the waypoint at closest-index is ahead of or behind the vehicle
        prev_coord = self.waypoints_2d[closest_index-1]
        closest_coord = self.waypoints_2d[closest_index]

        # Heading vectors between [current-position, closest-position] and
        # [closest-position, previous-position] when along the same direction
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
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        lane_waypoints = self.base_waypoints.waypoints[closest_idx:farthest_idx]

        if self.stopline_wpt_idx != None and (closest_idx <= self.stopline_wpt_idx <= farthest_idx):
            self.vehicle_state = VehicleState.Stop
            lane_waypoints = self.get_decelerating_waypoints(closest_idx, lane_waypoints)
        elif self.vehicle_state == VehicleState.Stop:
            self.vehicle_state = VehicleState.Drive
            self.decel_rate = None

        if self.vehicle_state == VehicleState.Drive:
            if abs(self.cur_velocity - self.get_waypoint_velocity(lane_waypoints[0])) > 1.:
                if self.prev_velocity == None:
                    start_vel = self.cur_velocity
                else:
                    start_vel = max(self.prev_velocity + 0.2, self.cur_velocity)
                lane_waypoints = self.get_accelerating_waypoints(start_vel, lane_waypoints)
        
        lane_msg = Lane()
        lane_msg.waypoints = lane_waypoints
        return lane_msg

    def get_accelerating_waypoints(self, start_vel, lane_waypoints):
        accel_wpts = []
        for i, wpt in enumerate(lane_waypoints):
            pt = Waypoint()
            pt.pose = wpt.pose
            distance = self.distance(lane_waypoints, 0, i)
            
            target_vel = start_vel + distance * ACCEL_RATE
            if target_vel < MIN_ACCEL_SPEED_THRESH:
                target_vel = MIN_ACCEL_SPEED_THRESH
            target_vel = min(target_vel, self.get_waypoint_velocity(wpt))
            pt.twist.twist.linear.x = target_vel
            accel_wpts.append(pt)

        return accel_wpts 

    def get_decelerating_waypoints(self, closest_idx, lane_waypoints):
        decel_wpts = []
        stop_idx = self.stopline_wpt_idx - closest_idx - NOSE_TO_CENTRE_OFFSET
        for i, wpt in enumerate(lane_waypoints):
            pt = Waypoint()
            pt.pose = wpt.pose
            distance = self.distance(lane_waypoints, i, stop_idx)

            if i >= stop_idx:
                target_vel = 0.
            elif distance < TRAFFIC_LIGHT_STOP_DIST:
                if self.decel_rate is None:
                    self.decel_rate = self.cur_velocity / distance
                target_vel = self.decel_rate * distance
                if target_vel <= 1.:
                    target_vel = 0.
                target_vel = min(target_vel, self.get_waypoint_velocity(wpt))
            else:
                target_vel = self.get_waypoint_velocity(wpt)
            pt.twist.twist.linear.x = target_vel
            decel_wpts.append(pt)
        
        return decel_wpts

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
