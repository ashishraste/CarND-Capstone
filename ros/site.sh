#!/bin/sh

catkin_make
source devel/setup.sh
roslaunch launch/site.launch
