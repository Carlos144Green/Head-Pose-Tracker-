README
Runs on Python 3.7
Requirement file included called "requirements.txt"
This is a head Pose Tracker. It takes an image, finds all faces, assigns points to them, and calculates a vector to show
where the user is facing. If there are multiple faces it will assign a red box to all faces except the most central one.

REPORT
I used openCV, mediapipe, and NumPy for my imports in order to detect faces, apply mesh points, and do calculations. The
order I solved the problems was 5,3,1,2,4. I tested only using a web camera with no stand-alone videos(5). I then applied
the face mesh which automatically detects the location of the face(3). I then found the left uppermost and the right lowermost corner of the face in order to bound the face(1). I then calculated the distance from the center of the video feed and
the center of the detected faces with the Pythagorean Theorem formula(2). Lastly, in order to get a vector properly representing
the direction the user is looking, I parse through all mesh points and handpick the outer eye corners, the lip corners,
the nose, and the chin. This is because I found a 3D face template with those depths all calculated. Having a template with the
depths helps with the stability of the output. With the points, depths, and a few other camera settings a proper head position
vector is calculated and displayed.

GIT
https://github.com/Carlos144Green/Head-Pose-Tracker-
