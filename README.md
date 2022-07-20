README

Runs on Python 3.7
Requirement file included called "requirements.txt"
This is a head Pose Tracker. It takes an image, finds all faces, assigns points to them, and calculates a visual vector to show
where the user is facing. If there are multiple faces it will assign a red box to all faces except the most central one.
I used openCV, mediapipe, and NumPy for my imports in order to detect faces, apply mesh points, and do calculations. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/60107217/157765900-b004541a-abdf-4ed5-a304-fddc66c6e609.gif" />
</p>

