# Find-and-Tracking-FLS-objects
The algorithm for object detection from FLS has 2 parts:
The first one is detection from one frame. In this part, the image is scanned by two windows, one is bigger than the others, for creating an image with much less noise. The mean of the outer window is taken, and for not losing the information of the pixel’s environment, the mean of the inner window is subtracted from the mean of the outer one. The result is the value of the pixel in the new image. From this step, it’s easy to recognize the objects by threshold.
The second part is false-alarms reducing, by tracking the objects found through some frames. It’s done by Kalman-filter. The matrices of the Kalman Filter still need some work for optimal using. If object found in one frame and than not found on the next one, it’s reducing from the objects list.

# The files
"objDetect.py" is where the algorithm of the detection from one frame.
"track.py" makes one track after one object.
"tracking.py" contains the class "tracking" which makes the algorithm of tracking after some objects. 
"main.py" is the main. You should start from there.
I wrote this code using the images in the directory "images". You can try it on another batch of data, just don't forget to change the name in "tracking.py".
