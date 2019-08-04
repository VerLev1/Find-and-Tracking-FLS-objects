# -*- coding: utf-8 -*-


import cv2
# import pandas as pd
import numpy as np
import munkres
# import importlib
import objDetect
import track


class tracking:

    def __init__(self):
        self.tracks = []  # list of track objects
        self.unassignedTracks = np.array([])  # array of the id-s of all the unsigned tracks
        self.unassignedDetection = np.array([])
        self.id = 1
        self.proccesNum = 0  # this will tell us how many times the whole proccess ocoured.

    def addNextFrameFeatures(self, mask, centroids, bboxes, circles, targetRate):
        self.bboxes = bboxes
        self.mask = mask
        self.centroids = centroids
        self.circles = circles
        self.targetRate = targetRate

    def predictNewLocations(self):
        for t in self.tracks:
            t.kalman.predict()
            predictedState = t.kalman.x
            t.bbox = np.array([predictedState[0][0], predictedState[2][0], t.bbox[2], t.bbox[3]])

    def costToAssignment(self, cost, costOfNonAssignments):
        # this function get nXm matrix of cost, and the maximum cost, and return assignments of tracks to detections
        # inputs: cost - matrix of nXm (num of tracks X num of detections)
        #        costOfNonAssignments- scalar
        # outputs: assignments- LX2 np.array, where L is the number of detections assigned to tracks
        #                      assignments[:][0] the detections, assignments[:][1] the tracks
        #         unassignedTracks - np array of all the tracks that no detection assigned to them
        #         unassignedDetections - np array of all the detections that not assigned to tracks

        assignments = munkres.Munkres().compute(cost)
        unassignedTracks = np.empty((0, 1), dtype=int)
        unassignedDetection = np.empty((0, 1), dtype=int)
        unAssignedIdx = np.empty((0, 1), dtype=int)
        for i in range(len(assignments)):
            track = assignments[i][0]
            detection = assignments[i][1]
            if cost[track][detection] >= costOfNonAssignments:  # and self.proccesNum>5:
                unassignedTracks = np.vstack((unassignedTracks, track))
                unassignedDetection = np.vstack((unassignedDetection, detection))
                # index from assignments
                unAssignedIdx = np.vstack((unAssignedIdx, i))
        numUnassigned = unAssignedIdx.shape
        for i in range(numUnassigned[0]):
            j = unAssignedIdx[numUnassigned[0] - i - 1]
            del assignments[j[0]]
        return assignments, unassignedTracks, unassignedDetection

    def detectionToTracksAssignment(self):

        self.assignments = np.array([])
        self.unassignedTracks = np.array([])

        nTracks = len(self.tracks)
        nDetections = self.centroids.shape
        nDetections = nDetections[0]
        self.unassignedDetection = np.array(range(nDetections))

        if nDetections == 0:
            return
        if nTracks == 0:
            return

        cost = [[0] * nDetections for i in range(nTracks)]  # zeros-matrice list
        costOfNonAssignment = 20
        if nDetections > nTracks:
            d = nDetections - nTracks
            for i in range(d):
                cost.append([costOfNonAssignment + 1] * nDetections)
        if nTracks > nDetections:
            d = nTracks - nDetections
            cost = [x + [costOfNonAssignment + 1] * d for x in cost]
        for i in range(nTracks):
            for j in range(nDetections):
                cost[i][j] = np.sqrt(((self.tracks[i].bbox[0] - self.centroids[j][0]) ** 2) + \
                                     ((self.tracks[i].bbox[1] - self.centroids[j][
                                         1]) ** 2))  # centroids need to be np array, as well as tracks['kalmanFilter']
        self.assignments, self.unassignedTracks, self.unassignedDetection = \
            self.costToAssignment(cost, costOfNonAssignment)

        # delete the "zero-padding":
        self.unassignedDetection = self.unassignedDetection[self.unassignedDetection < nDetections]
        self.unassignedTracks = self.unassignedTracks[self.unassignedTracks < nTracks]

    def updateTracks(self):
        invisibleForTooLong = 1
        ageThreshold = 0
        lost = []

        for a in self.assignments:
            trackIdx = a[0]
            detectionIdx = a[1]
            centroid = self.centroids[detectionIdx][:]
            bbox = self.bboxes[detectionIdx][:]
            targetRate = self.targetRate[detectionIdx]
            circles = self.circles[detectionIdx]
            self.tracks[trackIdx].updateTrack(1, circles, targetRate, bbox, centroid)

        for u in self.unassignedTracks:
            self.tracks[u].updateTrack(0)
            if self.tracks[u].age > ageThreshold:
                if self.tracks[u].invisibleCount > invisibleForTooLong:
                    lost.append(u)
            elif self.tracks[u].visibleCount / self.tracks[u].age > 0.6:
                lost.append(u)

        for l in range(len(lost) - 1, -1, -1):
            del self.tracks[lost[l]]

    def createNewTracks(self):
        unasDet = self.unassignedDetection.shape
        if unasDet[0] == 0:
            return

        # create properties-arrays of the unassigned detections only:
        centroids = self.centroids[self.unassignedDetection]
        bboxes = self.bboxes[self.unassignedDetection]
        targetRate = self.targetRate[self.unassignedDetection]
        circle = []
        for i in self.unassignedDetection:
            circle.append(self.circles[i])

        # create the track objects:
        for i in range(unasDet[0]):
            self.tracks.append(track.track(circles[i], self.id, bboxes[i], centroids[i], targetRate[i]))
            self.id += 1

        self.proccesNum = self.proccesNum + 1

    def showTracks(self):
        for t in self.tracks:
            p1 = (int(t.bbox[0]), int(t.bbox[1]))
            p2 = (int(t.bbox[0]) + int(t.bbox[2]), int(t.bbox[1]) + int(t.bbox[3]))
            cv2.rectangle(img1, p1, p2, t.color, 2, 1)
        cv2.imshow('tracking', img1)
        cv2.waitKey(0)


### the procces starts here ###
t = tracking()
for i in range(136, 152):
    # reading the frame
    name = 'images/try' + str(i) + '.jpg'
    img1 = cv2.imread(name)
    img = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)

    # find ROIs in the frame
    rois = objDetect.ROIfind(img)
    mask, centroids, bboxes, circles, targetRate = rois.apply()

    # tracking and updating
    t.addNextFrameFeatures(mask, centroids, bboxes, circles, targetRate)
    t.predictNewLocations()
    t.detectionToTracksAssignment()
    t.updateTracks()
    t.createNewTracks()
    t.showTracks()