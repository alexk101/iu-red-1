import time
from threading import Thread
from vidgear.gears import WriteGear
import numpy as np

class RTSPOutput(Thread):
    def __init__(self, fps: int, location: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._should_exit = False
        self._latest_frame: np.ndarray = np.zeros((480,640,3), dtype=np.uint8)
        self._timestep = 1.0 / fps
        output_params = {
            "-f": "rtsp",
            "-rtsp_transport": "tcp",
            '-c:v': 'h264_nvenc',
            '-input_framerate': fps,
            # '-vsync': '1',
            '-r': fps
        }
        
        self._stream = WriteGear(
            output = location, 
            logging = True, 
            compression_mode = True, 
            **output_params
        )
        # StreamGear(output = "output_foo.m3u8", format="hls")
    
    def update(self, frame):
        self._latest_frame = frame.copy()
        
    def run(self) -> None:
        while not self._should_exit:
            start = time.time()

            if self._latest_frame is not None:
                self._stream.write(self._latest_frame)
            
            diff = time.time() - start
            if diff < self._timestep:
                time.sleep(self._timestep - diff)

        self._stream.close()

    def stop(self):
        self._should_exit = True