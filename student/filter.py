# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np

import misc.params as params

# add project directory to python path to enable relative imports
import os
import sys

PACKAGE_PARENT = ".."
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params


class Filter:
    """Kalman filter class"""

    def __init__(self):
        pass

    def F(self):
        ############
        # Step 1: implement and return system matrix F
        ############
        dt = params.dt

        return np.matrix(
            [
                [1, 0, 0, dt, 0, 0],
                [0, 1, 0, 0, dt, 0],
                [0, 0, 1, 0, 0, dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        ############
        # END student code
        ############

    def Q(self):
        ############
        # Step 1: implement and return process noise covariance Q
        ############
        dt = params.dt
        q = params.q

        q1 = 1 / 1 * dt**1 * q
        q2 = 1 / 2 * dt**2 * q
        q3 = 1 / 3 * dt**3 * q

        return np.matrix(
            [
                [q3, 0, 0, q2, 0, 0],
                [0, q3, 0, 0, q2, 0],
                [0, 0, q3, 0, 0, q2],
                [q2, 0, 0, q1, 0, 0],
                [0, q2, 0, 0, q1, 0],
                [0, 0, q2, 0, 0, q1],
            ]
        )

        ############
        # END student code
        ############

    def predict(self, track):
        ############
        # Step 1: predict state x and estimation error covariance P to next timestep, save x and P in track
        ############

        # get state transition matrix and process noise covariance matrix
        F = self.F()
        Q = self.Q()

        # get current state and process covariance
        x = track.x
        P = track.P

        # predict new state assuming linear model and noise with expectation value 0
        x = F * x

        # calculate predicted state covariance matrix assuming linear model
        P = F * P * F.T + Q

        # save updated values in track
        track.set_x(x)
        track.set_P(P)

        ############
        # END student code
        ############

    def update(self, track, meas):
        ############
        # Step 1: update state x and covariance P with associated measurement, save x and P in track
        ############

        # get current state and process covariance
        x = track.x
        P = track.P

        # calculate the residual
        gamma = self.gamma(track, meas)

        # get the Jacobian of the measurement function
        H = meas.sensor.get_H(x)

        # calculate the covariance of the residual
        S = self.S(track, meas, H)

        # calculate the Kalman gain
        K = P * H.T * np.linalg.inv(S)

        # calculate the updated state
        x_update = x + K * gamma

        # calculate the updated covariance matrix
        I = np.eye(params.dim_state)
        P_update = (I - K * H) * P

        # save update in track
        track.set_x(x_update)
        track.set_P(P_update)

        ############
        # END student code
        ############
        track.update_attributes(meas)

    def gamma(self, track, meas):
        ############
        # Step 1: calculate and return residual gamma
        ############

        # get measured position values
        z = meas.z

        # get current state
        x = track.x

        # get the measurement expectation value
        hx = meas.sensor.get_hx(x)

        # calculate the residual assuming noise with expectation value 0
        gamma = z - hx

        return gamma

        ############
        # END student code
        ############

    def S(self, track, meas, H):
        ############
        # Step 1: calculate and return covariance of residual S
        ############

        # get current process covariance
        P = track.P

        # get measurement noise covariance matrix / uncertainty
        R = meas.R

        # calculate covariance of residual
        S = H * P * H.T + R

        return S

        ############
        # END student code
        ############
