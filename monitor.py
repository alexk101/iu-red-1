import matplotlib.pyplot as plt
from output import RTSPOutput
import numpy.random as random
import numpy as np
import cv2 as cv
import sys
import os


dir_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(dir_path)


# target = 'rtsp://{USERNAME}:{PASSWORD}@192.168.1.232:554/Src/MediaInput/h264/stream_1/ch_'
source = 'rtsp://192.168.1.150:8554/camera-15'
output = 'rtsp://192.168.1.150:8554/red-team'


def monitor():
    cap = cv.VideoCapture(source)
    cap2 = cv.VideoCapture(output)
    # if not cap.isOpened() or not cap2.isOpened():
    #     print("Cannot open camera")
    #     exit()


    # loop
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            continue

        cv.imshow('source', frame)


        ret2, frame2 = cap2.read()
        if not ret2:
            print("Can't receive frame (stream end?). Exiting ...")
            continue

        cv.imshow('output', frame2)
        if cv.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture

    cap.release()
    cap2.release()
    cv.destroyAllWindows()


def main() -> int:
    # test()
    monitor()
    # basic()
    return 0


if __name__ == "__main__":
    sys.exit(main())