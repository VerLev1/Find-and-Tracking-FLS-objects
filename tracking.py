# -*- coding: utf-8 -*-


import cv2
import numpy as np
import munkres
import track
import matplotlib.pyplot as plt
import imagehash
from PIL import Image


class Tracking:

    def __init__(self):
        self.tracks = []  # list of track objects
        self.unassignedTracks = np.array([])  # array of the id-s of all the unsigned tracks
        self.unassignedDetection = np.array([])
        self.assignments=[]
        self.id = 1
        self.processNum = 0  # this will tell us how many times the whole process ocoured.
        self.circles = []
        self.bboxes = []
        self.mask = []
        self.centroids = []
        self.tiny_masks = []
        self.img = []

    def add_next_frame_features(self,img, mask, centroids, bboxes, tiny_masks):
        self.bboxes = bboxes
        self.mask = mask
        self.centroids = centroids
        self.tiny_masks = tiny_masks
        self.img = img

    def cost_to_assignment(self, cost, max_cost):
        # get nXm matrix of cost, and the maximum cost allowed, and return assignments of tracks to detections
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
            if cost[track][detection] >= max_cost:  # and self.processNum>5:
                unassignedTracks = np.vstack((unassignedTracks, track))
                unassignedDetection = np.vstack((unassignedDetection, detection))
                # index from assignments
                unAssignedIdx = np.vstack((unAssignedIdx, i))
        numUnassigned = unAssignedIdx.shape
        for i in range(numUnassigned[0]):
            j = unAssignedIdx[numUnassigned[0] - i - 1]
            del assignments[j[0]]
        return assignments, unassignedTracks, unassignedDetection

    def detection_to_tracks_assignment(self):

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

        # making cost matrix
        deb = 0
        cost = [[0] * nDetections for i in range(nTracks)]  # zeros-matrice list
        max_cost = 12
        if nDetections > nTracks:
            d = nDetections - nTracks
            for i in range(d):
                cost.append([max_cost + 1] * nDetections)
        if nTracks > nDetections:
            d = nTracks - nDetections
            cost = [x + [max_cost + 1] * d for x in cost]
        for i in range(nTracks):
            for j in range(nDetections):
                distance = np.sqrt(((self.tracks[i].bbox[0] - self.bboxes[j][0]) ** 2) + \
                                     ((self.tracks[i].bbox[1] - self.bboxes[j][1]) ** 2))
                size_dif = np.sqrt(((self.tracks[i].bbox[2] - self.bboxes[j][2]) ** 2) + \
                                     ((self.tracks[i].bbox[3] - self.bboxes[j][3]) ** 2)) #difference between sizes
                tiny_mask_track = self.tracks[i].tiny_mask
                if len(tiny_mask_track.shape)<2:
                    sim1 = imagehash.average_hash(Image.fromarray(np.zeros((8,8), dtype=np.float)))
                else:
                    sim1 = imagehash.average_hash(Image.fromarray(tiny_mask_track))
                if len(self.tiny_masks[j].shape)<2:
                    sim2 = imagehash.average_hash(Image.fromarray(np.zeros((8,8), dtype=np.float)))
                else:
                    sim2 = imagehash.average_hash(Image.fromarray(self.tiny_masks[j]))
                similarity = np.abs(sim1 - sim2)
                if deb:
                    fig, ax = plt.subplots(1,2, sharex=True, sharey=True)
                    ax[0].imshow(tiny_mask_track)
                    ax[1].imshow(self.tiny_masks[j])
                    deb = not deb
                cost[i][j] = 0.2*distance + 0.45*size_dif + 0.35*similarity
        self.assignments, self.unassignedTracks, self.unassignedDetection = \
            self.cost_to_assignment(cost, max_cost)

        # delete the "zero-padding":
        self.unassignedDetection = self.unassignedDetection[self.unassignedDetection < nDetections]
        self.unassignedTracks = self.unassignedTracks[self.unassignedTracks < nTracks]

    def update_tracks(self):
        invisibleForTooLong = 2
        ageThreshold = 0
        lost = []

        for a in self.assignments:
            trackIdx = a[0]
            detectionIdx = a[1]
            centroid = self.centroids[detectionIdx][:]
            bbox = self.bboxes[detectionIdx][:]
            tiny_mask = self.tiny_masks[detectionIdx]
            self.tracks[trackIdx].updateTrack(1, bbox, centroid, tiny_mask)

        for u in self.unassignedTracks:
            self.tracks[u].updateTrack(0)
            if self.tracks[u].age > ageThreshold:
                if self.tracks[u].invisibleCount > invisibleForTooLong:
                    lost.append(u)
            elif self.tracks[u].visibleCount / self.tracks[u].age > 0.6:
                lost.append(u)

        for l in range(len(lost) - 1, -1, -1):
            del self.tracks[lost[l]]

    def create_new_tracks(self):
        unasDet = self.unassignedDetection.shape
        if unasDet[0] == 0:
            return

        # create properties-arrays of the unassigned detections only:
        centroids = self.centroids[self.unassignedDetection]
        bboxes = self.bboxes[self.unassignedDetection]

        # create the track objects:
        for i in range(unasDet[0]):
            self.tracks.append(track.track(self.id, bboxes[i], centroids[i], self.tiny_masks[i]))
            self.id += 1

        self.processNum = self.processNum + 1

    def show_tracks(self):
        plt.imshow(self.img)
        for t in self.tracks:
            ax = plt.gca()
            if (not t.invisibleCount)and (t.age>1):
                p1 = (int(t.bbox[0]), int(t.bbox[1]))
                p2 = (int(t.bbox[0]) + int(t.bbox[2]), int(t.bbox[1]) + int(t.bbox[3]))
                rect = plt.Rectangle((t.bbox[0],t.bbox[1]),t.bbox[2],t.bbox[3],edgecolor=t.color, fill=False, linewidth=2)
                ax.add_patch(rect)

    def get_circle(self, t, r_max=20.0):

        rows, cols, _ = self.img.shape
        # mask = np.zeros((rows, cols), dtype=int)
        ys, xs = t.tiny_mask.shape
        tiny1 = t.tiny_mask
        tiny = np.zeros((ys, xs, 3))
        tiny[:, :, 0] = tiny1
        tiny[:, :, 1] = tiny1
        tiny[:, :, 2] = tiny1
        circ = np.array([])

        radius = r_max*0.75

        x, y = np.meshgrid(np.arange(xs), np.arange(ys), sparse=True)

        #fig, ax = plt.subplots(2, 1)
        #ax[0].imshow(t.tiny_mask)

        for i in range(int(-0.5 * ys / radius), int(ys / radius)+1):
            for j in range(int(xs / radius)+1):
                x0 = 2 * radius * (i + 0.5 * j)
                y0 = 2 * radius * np.sqrt(3) / 2 * j

                r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)

                circ_indicator = r < radius
                img_indicator = tiny1 > 0

                if np.any(tiny[circ_indicator, 0] != 0):

                    mask = np.zeros((ys,xs))
                    mask[circ_indicator & img_indicator] = 1
                    #ax[1].imshow(mask)
                    #plt.show()
                    im, contours, h = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        if cnt.size >= 10:
                            (x1, y1), (ma, MA), _ = cv2.fitEllipse(cnt)

                    if circ.size:
                        circ = np.vstack([circ, [x1 + t.bbox[0] , y1 + t.bbox[1], int(MA/2)]])
                    else:
                        circ = np.array([x1 + t.bbox[0], y1 + t.bbox[1], int(MA/2)])

        # check if there are circles which contained by another one:
        contained = []
        for c1 in circ:
            ind = 0
            for c2 in circ:
                distance = np.sqrt(((c1[0]-c2[0])**2)+(c1[1]-c2[1])**2)
                if (c2[2]+distance) < c1[2]:
                    contained.append(ind)
                ind += 1
        contained = np.array(contained)
        circ = np.delete(circ, contained, 0)

        return circ

    def return_circles(self, num_of_images):
        ind = 0
        circles = []
        for t in self.tracks:
            condition = (t.age >= 0.8*num_of_images) and (t.visibleCount >= 0.8*t.age)
            if condition:
                circle = self.get_circle(t)
                circles.append(circle)
            ind += 1

        a = 2
        self.circles.append(circles)

    def show_circles(self):
        plt.imshow(self.img)
        for c in self.circles:
            if c:
                for one_c in c[0]:
                    ax = plt.gca()
                    circle = plt.Circle((one_c[0], one_c[1]), one_c[2], fill=False)
                    ax.add_patch(circle)
        plt.show()
