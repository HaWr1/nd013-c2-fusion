# Writeup: Track 3D-Objects Over Time

Please use this starter template to answer the following questions:

### 1. Write a short recap of the four tracking steps and what you implemented there (filter, track management, association, camera fusion). Which results did you achieve? Which part of the project was most difficult for you to complete, and why?

- 1st step: Implementation of the extended Kalman Filter with 6 states (3d position and speed) inside the predict and update functions
- 2nd step: Implementation of track management (init new tracks, increment/decrement track score, dismiss tracks)
- 3d step: Calculate Mahalanobis distance and associate measurements and tracks according to it.
- 4th step: Implementation of non linear measurement function for the camera and fov check which are needed for succesfull camera fusion

The fusion algorithm provides a reliable and stable detection of objects with a low dropout and false positive rate.

### 2. Do you see any benefits in camera-lidar fusion compared to lidar-only tracking (in theory and in your concrete results)? 
- More sensors add robustness to the system
- A camera can increase the accuracy of the tracking result (can be seen in RSME plot by comparing step 3 and 4)
- The camera helps to reduce FP rate of Lidar
- Camera can see objects much farer away

### 3. Which challenges will a sensor fusion system face in real-life scenarios? Did you see any of these challenges in the project?
- Fusion has a high ressource consumption
- Trade-Of between low FP rate and fast detection of objects

### 4. Can you think of ways to improve your tracking results in the future?
- Use also the other sensors in the Waymo dataset
- implement a more sophisticated tracking model (e.g. bicycle model)
- Use a different association algorithm
