# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Data association class with single nearest neighbor association and gating based on Mahalanobis distance
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
from scipy.stats.distributions import chi2

# add project directory to python path to enable relative imports
import os
import sys

PACKAGE_PARENT = ".."
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import misc.params as params


class Association:
    """Data association class with single nearest neighbor association and gating based on Mahalanobis distance"""

    def __init__(self):
        self.association_matrix = np.matrix([])
        self.unassigned_tracks = []
        self.unassigned_meas = []

    def associate(self, track_list, meas_list, KF):

        ############
        # Step 3: association:
        # - replace association_matrix with the actual association matrix based on Mahalanobis distance (see below) for all tracks and all measurements
        # - update list of unassigned measurements and unassigned tracks
        ############

        # initialize the association matrix and unassigned lists
        n_tracks = len(track_list)
        m_meas = len(meas_list)

        self.association_matrix = np.inf * np.ones((n_tracks, m_meas))
        self.unassigned_tracks = list(range(n_tracks))
        self.unassigned_meas = list(range(m_meas))

        # calculate Mahalanobis distance between each track and measurement and apply gating
        for i, track in enumerate(track_list):
            for j, meas in enumerate(meas_list):
                dist = self.MHD(track, meas, KF)

                if self.gating(dist, meas.sensor):
                    self.association_matrix[i, j] = dist
                else:
                    self.association_matrix[i, j] = np.inf

        ############
        # END student code
        ############

    def get_closest_track_and_meas(self):
        ############
        # Step 3: find closest track and measurement:
        # - find minimum entry in association matrix
        # - delete row and column
        # - remove corresponding track and measurement from unassigned_tracks and unassigned_meas
        # - return this track and measurement
        ############

        A = self.association_matrix

        # check if the association matrix only contains inf then return nan, nan
        if np.min(A) == np.inf:
            return np.nan, np.nan

        # find index of nearest measurement, convert to tuple of array coordinates and unpack it
        # see: https://numpy.org/doc/stable/reference/generated/numpy.argmin.html
        ind_track, ind_meas = np.unravel_index(np.argmin(A, axis=None), A.shape)

        # get and remove the nearest track and measurement from unassigned lists
        update_track = self.unassigned_tracks[ind_track]
        update_meas = self.unassigned_meas[ind_meas]

        self.unassigned_tracks.remove(update_track)
        self.unassigned_meas.remove(update_meas)

        # delete the matching row and column from the association matrix
        A = np.delete(A, ind_track, axis=0)
        A = np.delete(A, ind_meas, axis=1)
        self.association_matrix = A

        ############
        # END student code
        ############
        return update_track, update_meas

    def gating(self, MHD, sensor):
        ############
        # Step 3: return True if measurement lies inside gate, otherwise False
        ############

        limit = chi2.ppf(params.gating_threshold, df=sensor.dim_meas)
        if MHD < limit:
            return True
        else:
            return False

        ############
        # END student code
        ############

    def MHD(self, track, meas, KF):
        ############
        # Step 3: calculate and return Mahalanobis distance
        ############
        H = meas.sensor.get_H(track.x)
        gamma = KF.gamma(track, meas)
        S = KF.S(track, meas, H)
        MHD = gamma.T * np.linalg.inv(S) * gamma  # Mahalanobis distance formula
        return MHD

        ############
        # END student code
        ############

    def associate_and_update(self, manager, meas_list, KF):
        # associate measurements and tracks
        self.associate(manager.track_list, meas_list, KF)

        # update associated tracks with measurements
        while (
            self.association_matrix.shape[0] > 0
            and self.association_matrix.shape[1] > 0
        ):

            # search for next association between a track and a measurement
            ind_track, ind_meas = self.get_closest_track_and_meas()
            if np.isnan(ind_track):
                print("---no more associations---")
                break
            track = manager.track_list[ind_track]

            # check visibility, only update tracks in fov
            if not meas_list[0].sensor.in_fov(track.x):
                continue

            # Kalman update
            print(
                "update track",
                track.id,
                "with",
                meas_list[ind_meas].sensor.name,
                "measurement",
                ind_meas,
            )
            KF.update(track, meas_list[ind_meas])

            # update score and track state
            manager.handle_updated_track(track)

            # save updated track
            manager.track_list[ind_track] = track

        # run track management
        manager.manage_tracks(self.unassigned_tracks, self.unassigned_meas, meas_list)

        for track in manager.track_list:
            print("track", track.id, "score =", track.score)
