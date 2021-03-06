from rospy import get_time
from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter

class Controller(object):
    def __init__(self, *args, **kwargs):

        # Controller params
        self.vehicle_mass    = kwargs['vehicle_mass']
        self.decel_limit     = kwargs['decel_limit']
        self.accel_limit     = kwargs['accel_limit']
        self.wheel_radius    = kwargs['wheel_radius']
        self.wheel_base      = kwargs['wheel_base']
        self.steer_ratio     = kwargs['steer_ratio']
        self.max_lat_accel   = kwargs['max_lat_accel']
        self.max_steer_angle = kwargs['max_steer_angle']

        min_speed = 0.1
        self.yaw_controller = YawController(self.wheel_base,
         self.steer_ratio, min_speed, self.max_lat_accel, self.max_steer_angle)

        # Throttle controller params
        kp = 0.3
        ki = 0.1
        kd = 0.
        min_throttle = 0.
        max_throttle = 0.5 * self.accel_limit
        self.throttle_controller = PID(kp, ki, kd, min_throttle, max_throttle)

        tau = 0.5  # cutoff frequency = 1 / (2pi * tau)
        ts = 0.2  # sampling time
        self.vel_lpf = LowPassFilter(tau, ts)

        self.cur_linear_vel = None

        self.last_update_time = get_time()

    def control(self, linear_vel, angular_vel, cur_linear_vel, dbw_enabled):
        # Return throttle, brake, steer
        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0., 0., 0.
        
        cur_vel = self.vel_lpf.filt(cur_linear_vel)
        steering = self.yaw_controller.get_steering(linear_vel, angular_vel, cur_vel)

        # Update velocity and time params 
        vel_error = linear_vel - cur_linear_vel
        self.last_vel = cur_linear_vel

        current_time = get_time()
        sample_time = current_time - self.last_update_time
        self.last_update_time = current_time

        throttle = self.throttle_controller.step(vel_error, sample_time)
        brake = 0

        if linear_vel == 0. and cur_linear_vel < 0.1:
            throttle = 0
            brake = 400  # in N*m. Torque required to hold the car in place, e.g. near traffic lights
        elif throttle < 0.1 and vel_error < 0:
            throttle = 0
            decel = max(vel_error, self.decel_limit)
            brake = abs(decel) * self.vehicle_mass * self.wheel_radius

        return throttle, brake, steering
