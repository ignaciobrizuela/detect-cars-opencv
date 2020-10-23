import cv2
import numpy as np

# Read grayscale template
template = cv2.imread('patente.jpg', 0)
z, q = template.shape[::-1]
# Video capture
cap = cv2.VideoCapture('trafico.mp4')
# Choice method
meth = 'cv2.TM_CCOEFF_NORMED'
method = eval(meth)
# Delete background
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
# Matrix filters
kernel1 = np.ones((3, 3), np.uint8)
kernel2 = np.ones((15, 15), np.uint8)
# Car count variable
i = 0

while(1):
    # Snapshot
    ret, img = cap.read()
    # Copy of frame
    frame = img.copy()
    # Mask
    fgmask = fgbg.apply(frame)
    # Threshold with mask
    ret, th1 = cv2.threshold(fgmask, 150, 255, cv2.THRESH_BINARY)
    # Delete noise
    opening = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel1)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2)
    # Find contours
    a, contours, hierarchy = cv2.findContours(
        closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        # Get area
        area = cv2.contourArea(cnt)
        # If the areas ar big as a car
        frame = cv2.line(frame, (500, 400), (1280, 400), (255, 0, 0), 4)
        if area > 27000:
            # Moments
            M = cv2.moments(cnt)
            # Center coordinates
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            # Draw a circle in the middle of the rectangle
            cv2.circle(frame, (cx, cy), 5, (255, 0, 216), -1)
            # Draw a rectangle
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 120, 255), 3)
            # If the center reach the line
            if cy in range(390, 395):
                # Cut the car area
                crop_img = img[y:(y+h), x:(x+w)]
                # Convert to grayscale
                img_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                # Matching to find template
                res = cv2.matchTemplate(img_gray, template, method)
                # Threshold
                threshold = 0.58
                loc = np.where(res >= threshold)
                for pt in zip(*loc[::-1]):
                    cv2.rectangle(crop_img, pt, (pt[0] + z, pt[1] + q), (255, 0, 0), 2)
                # Save image
                cv2.imwrite('auto_'+str(i)+'.png', crop_img)
                i = i + 1

    cv2.imshow('frame', frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()
