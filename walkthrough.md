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