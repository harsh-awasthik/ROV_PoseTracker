import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor
import time

def detect_lines(image_path):
    """
    Detects lines in an image using Hough Transform.
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
    
    if lines is None:
        print(f"No lines detected in {image_path}")
        return None, None, img, edges
    
    angles = []
    points = []

    for rho, theta in lines[:, 0]:
        angle = np.rad2deg(theta)  # Convert radians to degrees
        angles.append(angle)

        # Convert (rho, theta) to (x1, y1, x2, y2) for regression
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        points.append((x1, y1, x2, y2))

    return angles, points, img, edges

def fit_average_lines(points):
    """
    Fits regression lines separately for vertical and horizontal components.
    """
    if not points:
        return None, None
    
    x_coords = []
    y_coords = []
    
    for x1, y1, x2, y2 in points:
        x_coords.extend([x1, x2])
        y_coords.extend([y1, y2])
    
    x_coords = np.array(x_coords).reshape(-1, 1)
    y_coords = np.array(y_coords).reshape(-1, 1)

    # Fit RANSAC Regression to find the dominant orientation
    ransac = RANSACRegressor()
    ransac.fit(x_coords, y_coords)
    slope = ransac.estimator_.coef_[0][0]
    
    avg_angle = np.rad2deg(np.arctan(slope))
    
    return avg_angle

# Process two images
index = 2497
img_path1 = "imgdatapool/img" + str(index) + ".jpg"
net_angle_diff = 0
while index < 2592:
    index += 1
    img_path2 = "imgdatapool/img" + str(index) + ".jpg"

    angles1, points1, img1, edges1 = detect_lines(img_path1)
    angles2, points2, img2, edges2 = detect_lines(img_path2)

    if points1 and points2:
        avg_angle1 = fit_average_lines(points1)
        avg_angle2 = fit_average_lines(points2)
        
        angle_difference = (avg_angle2 - avg_angle1)

        
        net_angle_diff += angle_difference
        
        print(f"Average Grill Angle ({index-1}): {avg_angle1:.2f}°")
        print(f"Average Grill Angle ({index}): {avg_angle2:.2f}°")
        print(f"Angle Difference: {angle_difference:.2f}°")

        # Display results
        cv2.imshow("Edges Image 1", edges1)
        cv2.imshow("Edges Image 2", edges2)
        cv2.waitKey(1)
        

    img_path1 = img_path2

cv2.waitKey(0)
print("Net Yaw: ", net_angle_diff)
cv2.destroyAllWindows()
