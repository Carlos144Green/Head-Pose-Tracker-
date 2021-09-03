import cv2
import numpy as np
import mediapipe as mp

def x_element(elem):
    return elem[0]
def y_element(elem):
    return elem[1]

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
# Here I increased the default confidence level for detection and dropped the level for tracking for better occlusion results
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=5, min_detection_confidence=.9, min_tracking_confidence=.01)
drawSpec = mpDraw.DrawingSpec((0,255,0),1,1)

cap = cv2.VideoCapture(0)
success, img = cap.read()
height, width = img.shape[:2]

# 3D model points, I could use the estimated 3D points mediapipe guesses but that has more noise
face3Dmodel = np.array([
    (0.0, 0.0, 0.0),            # Nose tip
    (0.0, -330.0, -65.0),       # Chin
    (-225.0, 170.0, -135.0),    # Left eye left corner
    (225.0, 170.0, -135.0),     # Right eye right corner
    (-150.0, -150.0, -125.0),   # Left Mouth corner
    (150.0, -150.0, -125.0)     # Right mouth corner
    ], dtype=np.float64)

dist_coeffs = np.zeros((4, 1))  # Pretending we have no lens distortion
camera_matrix = np.array([
    [width, 0, width / 2],
    [0, width, height / 2],
    [0, 0, 1]], dtype="double")

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:                                            # if faces found
        dist = []                                                               # stores distance from center, max and min values
        for faceLms in results.multi_face_landmarks:                            # loop through all matches
            # mpDraw.draw_landmarks(img, faceLms, landmark_drawing_spec=drawSpec) # draw every match
            faceXY = []                                                         # resets value for every face
            for id,lm in enumerate(faceLms.landmark):                           # loop over all land marks of one face
                x,y = int(lm.x*width), int(lm.y*height)                         # scale to img
                faceXY.append((x, y))                                           # put all xy points in neat array
            image_points = np.array([
                faceXY[1],      # "nose"
                faceXY[152],    # "chin"
                faceXY[226],    # "left eye"
                faceXY[446],    # "right eye"
                faceXY[57],     # "left mouth"
                faceXY[287]     # "right mouth"
                ], dtype="double")
            for i in image_points:
                cv2.circle(img, (int(i[0]), int(i[1])), 3, (255, 0, 0), -1)
            maxXY = max(faceXY, key=x_element)[0], max(faceXY, key=y_element)[1]
            minXY = min(faceXY, key=x_element)[0], min(faceXY, key=y_element)[1]

            xcenter = (maxXY[0] + minXY[0]) / 2
            ycenter = (maxXY[1] + minXY[1]) / 2

            dist.append(((int(((xcenter-width/2)**2+(ycenter-height/2)**2)**.4)), maxXY, minXY))     # distance, maxXY, minXY

            (success, rotation_vector, translation_vector) = cv2.solvePnP(face3Dmodel, image_points,  camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)
            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

            p1 = (int(image_points[0][0]), int(image_points[0][1]))
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

            cv2.line(img, p1, p2, (255, 0, 0), 2)

        dist.sort(key=x_element)
        for i, (_,x,y) in enumerate(dist):
            if i == 0:
                cv2.rectangle(img,x,y,(0,255,0),2)
            else:
                cv2.rectangle(img, x, y, (0, 0, 255), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)