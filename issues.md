# Issues

## Launching Styx 

The following issue occurs when we launch `styx.launch` file under the launch/ directory.

* Message-schema mismatch
    * Latest release of `dbw_mkz_ros` drive-by-wire (DBW) repository by Dataspeed Inc. has no variable
    by the name `steering_wheel_angle_cmd` in its SteeringReport message type inside the package `dbw_mkz_msgs`. Instead there's a `steering_wheel_cmd` which I suppose replaces `steering_wheel_angle_cmd` variable.
    
    * The bridge between Udacity's simulator and the DBW node(s) looks for `steering_wheel_angle_cmd`, hence raising the following error. 

        ```bash
        AttributeError: 'SteeringReport' object has no attribute 'steering_wheel_angle_cmd'
        ```
    
    * Solution: This is resolved by downloading the older version of `dbw_mkz_msgs` package from either
    Udacity provided virtual-machine having all dependencies installed.