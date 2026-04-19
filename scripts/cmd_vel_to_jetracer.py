#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
# cmd_vel_to_jetracer.py
#
# Bridges /cmd_vel (geometry_msgs/Twist) → JetRacer servo + ESC via I2C.
#
# Hardware:
#   JetRacer uses a Waveshare servo hat (PCA9685).
#   Channel 0 = steering servo, Channel 1 = ESC (continuous servo).
#
# Dependencies:
#   Preferred: standard Ubuntu package python-smbus (or python3-smbus)
#   Optional:  adafruit-circuitpython-servokit (if already installed)
#
# EDIT ME: adjust the hardware constants below to match your specific
# JetRacer build if steering direction or throttle direction is reversed.
# =============================================================================

from __future__ import division, print_function
import math
import time
import rospy
from geometry_msgs.msg import Twist

# ── Hardware constants — EDIT THESE to match your JetRacer ───────────────────
STEERING_CHANNEL = 0       # PCA9685 channel for steering servo
THROTTLE_CHANNEL  = 1       # PCA9685 channel for ESC (continuous servo mode)

STEERING_MID      = 90      # Servo angle (degrees) for straight-ahead
STEERING_RANGE    = 28      # ±degrees of servo travel from centre

# Negate this if your servo steers in the wrong direction
STEERING_SIGN     = 1

MAX_LINEAR_MS     = 0.5     # m/s that maps to full throttle (±1.0)
# Negate THROTTLE_SIGN if your ESC runs backwards
THROTTLE_SIGN     = 1

# PCA9685 timing defaults (50 Hz for servos/ESC)
PCA9685_I2C_BUS = 1
PCA9685_I2C_ADDR = 0x40
PWM_FREQUENCY_HZ = 50

# Steering servo pulse calibration (microseconds)
STEERING_MIN_US = 500
STEERING_MAX_US = 2500

# Continuous-servo/ESC pulse calibration (microseconds)
THROTTLE_NEUTRAL_US = 1500
THROTTLE_DELTA_US = 350


class PCA9685Driver(object):
    """Minimal PCA9685 driver using smbus; no Adafruit package required."""

    MODE1 = 0x00
    PRESCALE = 0xFE
    LED0_ON_L = 0x06

    def __init__(self, bus=1, address=0x40, freq_hz=50):
        try:
            import smbus
            self._bus = smbus.SMBus(bus)
        except Exception as e:
            rospy.logfatal(
                "[cmd_vel_bridge] Failed to open I2C bus (smbus): %s", str(e)
            )
            raise

        self._address = address
        self._freq_hz = freq_hz
        self._write8(self.MODE1, 0x00)
        self.set_pwm_freq(freq_hz)

    def _write8(self, reg, value):
        self._bus.write_byte_data(self._address, reg, value & 0xFF)

    def _read8(self, reg):
        return self._bus.read_byte_data(self._address, reg)

    def set_pwm_freq(self, freq_hz):
        prescaleval = 25000000.0
        prescaleval /= 4096.0
        prescaleval /= float(freq_hz)
        prescaleval -= 1.0
        prescale = int(math.floor(prescaleval + 0.5))

        oldmode = self._read8(self.MODE1)
        sleep = (oldmode & 0x7F) | 0x10
        self._write8(self.MODE1, sleep)
        self._write8(self.PRESCALE, prescale)
        self._write8(self.MODE1, oldmode)
        time.sleep(0.005)
        self._write8(self.MODE1, oldmode | 0xA1)

    def set_pwm(self, channel, on_count, off_count):
        reg = self.LED0_ON_L + 4 * int(channel)
        self._write8(reg + 0, on_count & 0xFF)
        self._write8(reg + 1, (on_count >> 8) & 0xFF)
        self._write8(reg + 2, off_count & 0xFF)
        self._write8(reg + 3, (off_count >> 8) & 0xFF)

    def set_pulse_us(self, channel, pulse_us):
        # ticks = pulse_us / period_us * 4096
        period_us = 1000000.0 / float(self._freq_hz)
        ticks = int(max(0, min(4095, round((pulse_us / period_us) * 4096.0))))
        self.set_pwm(channel, 0, ticks)


class JetRacerActuator(object):
    def __init__(self):
        self.backend = None
        self.kit = None
        self.pca = None

        # Prefer ServoKit when available; fallback to raw PCA9685 via smbus.
        try:
            from adafruit_servokit import ServoKit
            self.kit = ServoKit(channels=16)
            self.backend = 'servokit'
            rospy.loginfo("[cmd_vel_bridge] Using Adafruit ServoKit backend.")
            return
        except Exception as e:
            rospy.logwarn(
                "[cmd_vel_bridge] ServoKit unavailable (%s). "
                "Falling back to raw PCA9685 (smbus).", str(e)
            )

        self.pca = PCA9685Driver(
            bus=PCA9685_I2C_BUS,
            address=PCA9685_I2C_ADDR,
            freq_hz=PWM_FREQUENCY_HZ,
        )
        self.backend = 'pca9685_smbus'
        rospy.loginfo("[cmd_vel_bridge] Using PCA9685 smbus backend.")

    def set_steering_angle(self, angle_deg):
        angle_deg = max(0.0, min(180.0, float(angle_deg)))
        if self.backend == 'servokit':
            self.kit.servo[STEERING_CHANNEL].angle = angle_deg
            return

        span = float(STEERING_MAX_US - STEERING_MIN_US)
        pulse_us = STEERING_MIN_US + (angle_deg / 180.0) * span
        self.pca.set_pulse_us(STEERING_CHANNEL, pulse_us) 
    def set_throttle(self, throttle):
        throttle = max(-1.0, min(1.0, float(throttle)))
        if self.backend == 'servokit':
            self.kit.continuous_servo[THROTTLE_CHANNEL].throttle = throttle
            return

        pulse_us = THROTTLE_NEUTRAL_US + throttle * THROTTLE_DELTA_US
        self.pca.set_pulse_us(THROTTLE_CHANNEL, pulse_us)


actuator = None


def cmd_vel_cb(msg):
    """Convert Twist to servo angle and ESC throttle."""

    # ── Throttle ──────────────────────────────────────────────────────────────
    # linear.x: positive = forward, negative = reverse
    throttle = THROTTLE_SIGN * max(-1.0, min(1.0,
                                             msg.linear.x / MAX_LINEAR_MS))
    actuator.set_throttle(throttle)

    # ── Steering ──────────────────────────────────────────────────────────────
    # angular.z positive = turn left (CCW).
    # Ackermann: angular.z is used as a steering proxy here.
    # We normalise by omega_max (v_cruise * tan(max_steer) / wheelbase).
    # A simple normalisation: assume angular.z ≤ 1.5 rad/s at cruise speed.
    omega_max_approx = 1.5   # rad/s — should match controller's omega_max
    steer_frac = max(-1.0, min(1.0, msg.angular.z / omega_max_approx))
    steer_deg  = STEERING_MID - STEERING_SIGN * steer_frac * STEERING_RANGE
    steer_deg  = max(STEERING_MID - STEERING_RANGE,
                     min(STEERING_MID + STEERING_RANGE, steer_deg))
    actuator.set_steering_angle(steer_deg)


def shutdown_cb():
    """Send zero commands on ROS shutdown to prevent runaway."""
    rospy.loginfo("[cmd_vel_bridge] Shutdown — zeroing servo and ESC.")
    try:
        actuator.set_throttle(0.0)
        actuator.set_steering_angle(STEERING_MID)###########
    except Exception:
        pass


if __name__ == '__main__':
    rospy.init_node('cmd_vel_to_jetracer')
    actuator = JetRacerActuator()
    rospy.on_shutdown(shutdown_cb)
    rospy.Subscriber('/cmd_vel', Twist, cmd_vel_cb, queue_size=1)
    rospy.loginfo(
        "[cmd_vel_bridge] Ready (%s). Steering ch=%d (mid=%d°±%d°) ESC ch=%d max_speed=%.1f m/s",
        actuator.backend,
        STEERING_CHANNEL, STEERING_MID, STEERING_RANGE,
        THROTTLE_CHANNEL, MAX_LINEAR_MS)
    rospy.spin()
