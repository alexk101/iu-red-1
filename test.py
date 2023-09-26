from output import RTSPOutput
import numpy.random as random
import numpy as np
import cv2 as cv
import sys
import os
import torch
import polars as pl
import yaml

# While running the model and focused on the video feed, the level of noise intensity
# and blurring can be adjusted by modifying the param.yml file and pressing 'u' to 
# update the stream live

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)  # or yolov5n - yolov5x6 or custom

dir_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(dir_path)


# target = 'rtsp://{USERNAME}:{PASSWORD}@192.168.1.232:554/Src/MediaInput/h264/stream_1/ch_'
target = 'rtsp://192.168.1.150:8554/camera-15'
print(target)

# slider current value
STRENGTH = 16
BLUR = 7
noise_n = 20
NOISE = []

def gen_noise():
    global STRENGTH
    global NOISE
    NOISE = []
    for x in range(noise_n):
        frame_noise = random.normal(64,STRENGTH,(480,640,3))
        frame_noise = (frame_noise*0.5).astype(np.uint8)
        NOISE.append(frame_noise)


def predict(frame):
    global BLUR
    global STRENGTH
    results = model(frame)  # inference
    crops = pl.from_pandas(results.pandas().xyxy[0])
    people: pl.DataFrame = crops.filter(pl.col('name')=='person')
    if people.height:
        for row in people.iter_rows(named=True):
            x_min = int(row['xmin'])
            x_max = int(row['xmax'])
            y_min = int(row['ymin'])
            y_max = int(row['ymax'])
            sub = frame[y_min:y_max,x_min:x_max].copy()
            x_size = x_max-x_min
            y_size = y_max-y_min
            frame_noise = random.normal(64,STRENGTH,(y_size,x_size,3)).astype(np.uint8)
            sub = cv.addWeighted(sub, 0.5, frame_noise, 0.5, 0)
            frame[y_min:y_max,x_min:x_max] = cv.GaussianBlur(sub, (BLUR,BLUR), 0)
    
    return frame


def cv_test():
    global STRENGTH
    global NOISE
    global BLUR
    gen_noise()
    # print(cv.getBuildInformation())
    # 192.168.1.150
    out = RTSPOutput(30, 'rtsp://192.168.1.150:8554/red-team')
    out.start()
    cap = cv.VideoCapture(target)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    

    # loop
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame = np.asarray(frame, np.uint8)

        predict(frame)
        gauss_noise = NOISE[random.randint(0,noise_n)]
        frame = cv.addWeighted(frame, 0.5, gauss_noise, 0.5, 0)

        out.update(frame)
        # out._stream.write(np.asarray(frame))
        cv.imshow('frame', frame)
        key = cv.waitKey(1)

        if key == ord('q'):
            break
        elif key == ord('u'):
            with open('./params.yml', 'r') as fp:
                params = yaml.safe_load(fp)
                STRENGTH = params['noise']
                BLUR = params['blur']
            print(STRENGTH)
            gen_noise()
    # When everything done, release the capture
    out.stop()
    cap.release()
    cv.destroyAllWindows()


def main() -> int:
    cv_test()
    return 0


if __name__ == "__main__":
    sys.exit(main())