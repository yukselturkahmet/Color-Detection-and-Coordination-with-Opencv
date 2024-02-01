import cv2
from PIL import Image
import numpy as np

#Teacher please download to your library these three: opencv-python, numpy, Pillow

#We Defined color thresholds in HSV space
color_ranges = {
    "red": ([0, 100, 100], [10, 255, 255]),
    "blue": ([100, 100, 100], [120, 255, 255]),
    "green": ([40, 100, 100], [80, 255, 255]),
}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for color_name, (lower_limit, upper_limit) in color_ranges.items():
        lower_limit = np.array(lower_limit, dtype=np.uint8)
        upper_limit = np.array(upper_limit, dtype=np.uint8)

        mask = cv2.inRange(hsvImage, lower_limit, upper_limit)

        mask_ = Image.fromarray(mask)

        bbox = mask_.getbbox()

        if bbox is not None:
            x1, y1, x2, y2 = bbox

            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

            # We calculated text position inside the bounding box
            text_x = x1 + 5
            text_y = y1 + 25

            # We added the color name and coordinates text
            cv2.putText(frame, f"{color_name.capitalize()} ({x1},{y1},{x2},{y2})", (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

