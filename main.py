

import cv2
import objDetect
import tracking


t = tracking.Tracking()
for i in range(136, 152):
    # reading the frame
    name = 'images/try' + str(i) + '.jpg'
    img1 = cv2.imread(name)
    img = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)

    # find ROIs in the frame
    rois = objDetect.ROIfind(img)
    mask, centroids, bboxes, circles, targetRate = rois.apply()

    # tracking and updating
    t.addNextFrameFeatures(img1, mask, centroids, bboxes, circles, targetRate)
    t.predictNewLocations()
    t.detectionToTracksAssignment()
    t.updateTracks()
    t.createNewTracks()
    t.showTracks()