# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Classes for track and track management
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
import collections

# add project directory to python path to enable relative imports
import os
import sys

PACKAGE_PARENT = ".."
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params


class Track:
    """Track class with state, covariance, id, score"""

    def __init__(self, meas, id):
        print("creating track no.", id)
        M_rot = meas.sensor.sens_to_veh[
            0:3, 0:3
        ]  # rotation matrix from sensor to vehicle coordinates

        ############
        # Step 2: initialization:
        # - replace fixed track initialization values by initialization of x and P based on
        # unassigned measurement transformed from sensor to vehicle coordinates
        # - initialize track state and track score with appropriate values
        ############

        # initialize with measured position and zero speed
        self.x = np.concatenate((M_rot * meas.z, np.zeros((3, 1))))

        # intialize position uncertainty with transformed sensor position covariance and defined velocity

        # calculate position covariance submatrix
        R = meas.R
        P_pos = M_rot * R * M_rot.T

        # create velocity covariance submatrix from params file
        p44 = params.sigma_p44**2
        p55 = params.sigma_p55**2
        p66 = params.sigma_p66**2

        P_vel = np.diag((p44, p55, p66))

        # Create P-Matrix in the form
        # P_pos |   0
        # --------------
        #   0   | P_vel
        Zero = np.zeros_like(P_pos)
        P = np.concatenate(
            (
                np.concatenate((P_pos, Zero), axis=0),
                np.concatenate((Zero, P_vel), axis=0),
            ),
            axis=1,
        )

        self.P = P

        self.state = "initialized"
        self.score = 1.0 / params.window

        ############
        # END student code
        ############

        # other track attributes
        self.id = id
        self.width = meas.width
        self.length = meas.length
        self.height = meas.height
        self.yaw = np.arccos(
            M_rot[0, 0] * np.cos(meas.yaw) + M_rot[0, 1] * np.sin(meas.yaw)
        )  # transform rotation from sensor to vehicle coordinates
        self.t = meas.t

    def set_x(self, x):
        self.x = x

    def set_P(self, P):
        self.P = P

    def set_t(self, t):
        self.t = t

    def update_attributes(self, meas):
        # use exponential sliding average to estimate dimensions and orientation
        if meas.sensor.name == "lidar":
            c = params.weight_dim
            self.width = c * meas.width + (1 - c) * self.width
            self.length = c * meas.length + (1 - c) * self.length
            self.height = c * meas.height + (1 - c) * self.height
            M_rot = meas.sensor.sens_to_veh
            self.yaw = np.arccos(
                M_rot[0, 0] * np.cos(meas.yaw) + M_rot[0, 1] * np.sin(meas.yaw)
            )  # transform rotation from sensor to vehicle coordinates


###################


class Trackmanagement:
    """Track manager with logic for initializing and deleting objects"""

    def __init__(self):
        self.N = 0  # current number of tracks
        self.track_list = []
        self.last_id = -1
        self.result_list = []

    def manage_tracks(self, unassigned_tracks, unassigned_meas, meas_list):
        ############
        # Step 2: implement track management:
        # - decrease the track score for unassigned tracks
        # - delete tracks if the score is too low or P is too big (check params.py for parameters that might be helpful, but
        # feel free to define your own parameters)
        ############

        # decrease score for unassigned tracks
        for i in unassigned_tracks:
            track = self.track_list[i]
            track.score -= 1.0 / params.window

        # delete old tracks
        for track in self.track_list:
            if (
                track.state == "confirmed" and track.score <= params.delete_threshold
            ) or (
                (track.state == "initialized" or track.state == "tentative")
                and (track.P[0, 0] > params.max_P or track.P[1, 1] > params.max_P)
            ):
                self.delete_track(track)

        ############
        # END student code
        ############

        # initialize new track with unassigned measurement
        for j in unassigned_meas:
            if (
                meas_list[j].sensor.name == "lidar"
            ):  # only initialize with lidar measurements
                self.init_track(meas_list[j])

    def addTrackToList(self, track):
        self.track_list.append(track)
        self.N += 1
        self.last_id = track.id

    def init_track(self, meas):
        track = Track(meas, self.last_id + 1)
        self.addTrackToList(track)

    def delete_track(self, track):
        print("deleting track no.", track.id)
        self.track_list.remove(track)

    def handle_updated_track(self, track):
        ############
        # Step 2: implement track management for updated tracks:
        # - increase track score
        # - set track state to 'tentative' or 'confirmed'
        ############

        # increase track score by 1/params.window and limit it's value to [0,1]
        track.score = max(0, min(track.score + 1.0 / params.window, 1))

        if track.score > params.confirmed_threshold:
            track.state = "confirmed"
        else:
            track.state = "tentative"

        ############
        # END student code
        ############
