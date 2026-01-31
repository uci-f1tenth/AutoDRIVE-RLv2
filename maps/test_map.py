import cv2
import numpy as np
import os

def create_technical_track():
    img = np.zeros((800, 1200), dtype=np.uint8)
    
    # 1. Main Track Waypoints
    pts = np.array([[100, 400], [300, 150], [900, 150], [1100, 400], 
                    [900, 650], [600, 500], [300, 650], [150, 600]], np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=255, thickness=80, lineType=cv2.LINE_AA)

    # 2. Add a Shortcut (Narrower than the main track)
    # This cuts through the middle chicane
    cv2.line(img, (900, 150), (600, 500), 255, 30) 

    # 3. Add "Sensor Noise" (Spurs)
    cv2.line(img, (300, 150), (350, 50), 255, 10) # Sharp spur out of a corner
    cv2.rectangle(img, (1050, 350), (1150, 450), 255, -1) # A 'pillar' blob near the wall

    # Smooth for organic edges
    img = cv2.GaussianBlur(img, (21, 21), 0)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    if not os.path.exists("maps"): os.makedirs("maps")
    cv2.imwrite("maps/closed_track.pgm", img)
    print("✅ Technical track with shortcuts and spurs created.")

if __name__ == "__main__":
    create_technical_track()