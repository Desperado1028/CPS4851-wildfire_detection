import cv2
import time
import threading
from collections import deque
from typing import Union, Optional, Tuple
import numpy as np

class VideoStream:
    """线程化视频流捕获，减少 I/O 阻塞"""
    def __init__(self, src: Union[int, str] = 0, width: int = 640, height: int = 480):
        self.src = src
        self.width = width
        self.height = height
        self.cap = None
        self.grabbed = False
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        self._open()

    def _open(self):
        if isinstance(self.src, str) and self.src.isdigit():
            self.src = int(self.src)
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.grabbed, self.frame = self.cap.read()

    def start(self):
        threading.Thread(target=self._update, daemon=True).start()
        return self

    def _update(self):
        while not self.stopped:
            if self.cap is not None and self.cap.isOpened():
                grabbed, frame = self.cap.read()
                with self.lock:
                    self.grabbed = grabbed
                    if grabbed:
                        self.frame = frame
            else:
                time.sleep(0.01)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if self.frame is None:
                return False, None
            return self.grabbed, self.frame.copy()

    def stop(self):
        self.stopped = True
        if self.cap:
            self.cap.release()

    def __del__(self):
        self.stop()