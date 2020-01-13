
import cv2
import objDetect
import tracking
# import matplotlib.pyplot as plt
# import time

# t1 = time.time()

t = tracking.Tracking()
i1 = 250; i2 = 262
for i in range(i1, i2):
    # reading the frame
    name = 'images/Try' + str(i) + '.jpg'
    # name = 'images/Swimmer/Test' + str(i) + '.jpg'
    img1 = cv2.imread(name)
    img = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)

    # clean and find ROIs in the frame
    rois = objDetect.ROIfind(img)

    mask, centroids, bboxes, tiny_masks = rois.apply()

    # cleaning by tracking
    t.add_next_frame_features(img1, mask, centroids, bboxes, tiny_masks)
    # t.predictNewLocations()
    t.detection_to_tracks_assignment()
    t.update_tracks()
    t.create_new_tracks()
#     t.show_tracks()
# plt.show()
t.return_circles(i2-i1+1) # this makes the list to return
t.show_circles()
# e = time.time()-t1
