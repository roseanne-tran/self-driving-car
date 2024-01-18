import cv2
import numpy as np

def preprocess_image(original_image):
    copy = np.copy(original_image)
    gray_scale = cv2.cvtColor(copy, cv2.COLOR_RGB2GRAY) #convert the image to grayscale
    smooth = cv2.GaussianBlur(gray_scale, (3,3), 0) #smooth th edges
    edge = cv2.Canny(smooth, 10, 150) #edge detection
    return edge

def draw_driving_region(original_image, processed_image):
    lane_region = np.array([[(180, processed_image.shape[0]), (450, 350), (1200, processed_image.shape[0])]], np.int32)
    mask = np.zeros_like(processed_image)
    cv2.fillPoly(mask, lane_region, 255)
    driving_lane = cv2.bitwise_and(processed_image, mask)

  #find lines that fit best
    lines = cv2.HoughLinesP(driving_lane, 2, np.pi/180, 100, np.array([]), minLineLength=40,maxLineGap=5)
    return lines
  # left_right_lines = smooth_lines(frame, lines)

def smooth_lines(original_image, driving_region):
    left_lines = []
    right_lines = []
    if driving_region is None:
        return None
    for lines in driving_region:
        for x1, y1, x2, y2 in lines:
            line_parameters = np.polyfit((x1,x2), (y1,y2), 1)
            slope = line_parameters[0]
            intercept= line_parameters[1]
            if slope < 0:
                left_lines.append((slope, intercept))
            else:
                right_lines.append((slope, intercept))

    if len(left_lines) and len(right_lines):
        left_average = np.average(left_lines, axis=0)
        right_average = np.average(right_lines, axis=0)
        left = find_coordinates(original_image, left_average)
        right = find_coordinates(original_image, right_average)
        lines = [left, right]
        return lines


def find_coordinates(original_image, line):
    slope = line[0]
    intercept = line[1]
    y1 = int(original_image.shape[0])
    y2 = int(y1*3/5)
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return [[x1, y1, x2, y2]]

def merge_driving_region_to_original(original_image, lines):
    lane_lines = np.zeros_like(original_image)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
#                print(x1, ' ', y1, ' ', x2, ' ', y2)
                cv2.line(lane_lines,(x1,y1),(x2,y2),(255,0,0),10)

    merged_lines= cv2.addWeighted(frame, 0.8, lane_lines, 1, 1)

    return merged_lines


cap = cv2.VideoCapture("driving.MOV")

while(cap.isOpened()):
    _, frame = cap.read()
    processed_image = preprocess_image(frame)
    driving_region = draw_driving_region(frame, processed_image)
    smooth_driving_region = smooth_lines(frame, driving_region)
    lines_on_frame = merge_driving_region_to_original(frame, smooth_driving_region)

    cv2.imshow('lane detection',lines_on_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
