# Walkthrough Steps

## System Architecture

Below diagram shows the system architecture of this project.

![system_arch](./resources/system-arch.png)

## Waypoint Updater

![waypoint_updater_node](./resources/waypoint-updater-node.png)

Purpose : Takes the waypoints published by different nodes and translates them into a list 
of waypoints each to be hit with a target velocity. This is fed to the controller to drive
the car.

### Subscribes

* /base_waypoints : Publishes a list of all waypoints for the track, so this list includes waypoints both before and after the car, and this happens only once the simulator starts.
    * Message type : `Lane`
    * Additional info : Contains list of `Pose(s)` and `Twist(s)`, each of which are timestamped  

* /current_pose : Publishes current `Pose` of the car.
    * Message type : `Pose` timestamped

* /traffic_waypoint : Contains waypoint of a stop-line associated with a traffic light.
    * Message type : `Int32` 

* /current_velocity : Current velocity of the car published by the `styx-server` that bridges between the drive-by-wire system and our system of ROS nodes. 

### Publishes

* /final_waypoints : Contains list of waypoints for the car to follow. Is subscribed
to by the controller to drive through these waypoints.
    * Message type : `Lane`
    * Additional info : Uses the same `header` as as /base_waypoints message

### Implementation

Using the known waypoints provided on start-up (from `/base_waypoints` topic), this node find the next set of nearest waypoints and the speed at which the car must hit these waypoints. It does so by doing the following in its run-time loop.

1. Find the nearest set of waypoints ahead of the car.
2. Determine whether a red traffic-light falls in between the current-position and the nearest set of waypoints planned.
3. Calculate target velocities for each waypoint
4. Publish the target waypoints along with their velocities to `/final_waypoints` topic.

In step 2., if a red traffic light doesn't appear the waypoint-updater node continues with setting the velocities for the waypoints provided by the `waypoint-loader` node. Detection of red traffic-light triggers calculation of decelerating velocities to take such that the car stops in front of the stop-line near the detected traffic-light. In order to decelerate just before a certain distance from the stop-line, a `stopping-distance-threshold` of 15 metres is set, and the waypoints in this 15 metres length will have the decelerating velocities set. 


## Drive-by-wire Node

![dbw_node](./resources/dbw-node.png)

### Subscribes

* /current_velocity : Publishes current velocity of the car in SI units. 
    * Message type : `TwistStamped`
    * Additional info : Contains linear and angular velocity components of the car
    We only use the linear velocity components of this message to compute steering, throttle and brake values

* /twist_cmd : Publishes linear and angular velocities which must be taken by the car in order 
to hit the waypoints computed by the Waypoint Updater node.
    * Message type : `TwistStamped`
    * Additional info : We use the linear and angular velocity components of this message to compute steering, throttle and brake values

### Publishes

* Steering, throttle, and brake values on ` car/steering_cmd`, ` car/throttle_cmd`, and ` car/brake_cmd` topics respectively.


## Traffic Light Detector 

![tl_detector](./resources/tl-detector-node.png)

### Subscribes

* /base_waypoints : Complete list of waypoints for the test-track is used to stop at a waypoint in front of traffic-lights.
    * Message type: `Lane`

* /image_color : Contains frontal camera image stream in order to detect and classify the traffic lights.

* /current_pose : Used to determine car's location.

*  /vehicle/traffic_lights : Contains traffic lights' position in 3D space (x, y, z).
    * Additional info: When working on the simulator, this topic will have state of the light (red, green, yellow). We could use them for testing the classifier in the simulator, and use the classifier once its ready to publish the state of the lights

### Publishes

* /traffic_waypoint : Waypoint nearest to a red light's stop-line. Is used by the [waypoint-updater](#waypoint-updater) node to stop the car near the traffic light.

Paramater Config used

* /traffic_light_config : Contains a dictionary of permanent waypoint coordinates of traffic lights' stop-line existing in the simulated track.

### Implementation

We use [SSD Inception V2 model](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_17.tar.gz) trained on CoCo dataset for classifying the traffic lights. This model is taken from the [Tensorflow Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) and is trained on the dataset having the traffic-lights from the simulator and from the site tracks. Instructions for modifying the model configuration and the re-training / saving steps were followed from [here](https://github.com/alex-lechner/Traffic-Light-Classification).

