# Walkthrough Steps

## System Architecture

![system_arch](./resources/system-arch.png)

## Waypoint Updater

![waypoint_updater_node](./resources/waypoint-updater-node.png)

Subscribes

* /base_waypoints : Publishes a list of all waypoints for the track, so this list includes waypoints both before and after the vehicle, and this happens only once the simulator starts.
    * Message type : `Lane`
    * Additional info : Contains list of `Pose(s)` and `Twist(s)`, each of which are timestamped  

* /current_pose : Publishes current `Pose` of the vehicle.
    * Message type : `Pose` timestamped

Publishes

* /final_waypoints : Contains list of waypoints for the vehicle to follow. Is subscribed
to by the controller to drive through these waypoints.
    * Message type : `Lane`
    * Additional info : USes the same `header` as as /base_waypoints message


## Drive-by-wire Node

![dbw_node](./resources/dbw-node.png)

Subscribes

* /current_velocity : Publishes current velocity of the car in SI units. 
    * Message type : `TwistStamped`
    * Additional info : Contains linear and angular velocity components of the vehicle.
    We only use the linear velocity components of this message to compute steering, throttle and brake values

* /twist_cmd : Publishes linear and angular velocities which must be taken by the vehicle in order 
to hit the waypoints computed by the Waypoint Updater node.
    * Message type : `TwistStamped`
    * Additional info : We use the linear and angular velocity components of this message to compute steering, throttle and brake values

Publishes

* Steering, throttle, and brake values on `/vehicle/steering_cmd`, `/vehicle/throttle_cmd`, and `/vehicle/brake_cmd` topics respectively.


## Traffic Light Detector 

![tl_detector](./resources/tl-detector-node.png)

Subscribes

* /base_waypoints : Complete list of waypoints for the test-track is used to stop at a waypoint in front of traffic-lights.
    * Message type: `Lane`

* /image_color : Contains frontal camera image stream in order to detect and classify the traffic lights.

* /current_pose : Used to determine vehicle's location.

* /vehicle/traffic_lights : Contains traffic lights' position in 3D space (x, y, z).
    * Additional info: When working on the simulator, this topic will have state of the light (red, green, yellow). We could use them for testing the classifier in the simulator, and use the classifier in the real-run to publish the state of the lights

Publishes

* /traffic_waypoint : Waypoint nearest to a red light's stop-line. Is used by the [waypoint-updater](#waypoint-updater) node to stop the vehicle near the traffic light.

Paramater Config used

* /traffic_light_config : Contains a dictionary of permanent waypoint coordinates of traffic lights' stop-line existing in the simulated track.