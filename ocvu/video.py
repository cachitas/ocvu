# -*- coding: utf-8 -*-

"""
Video object
"""

import logging

import numpy as np
import cv2

from .utils import _mean_squared_error


logger = logging.getLogger(__name__)


class Video:

    """
    OpenCV Video.
    """

    def __init__(self, filepath, grayscale=False):
        # OpenCV VideoCapture object
        self.filepath = filepath
        self._capture = cv2.VideoCapture(filepath)
        self.grayscale = grayscale
        self.bgmodel = None

    def __iter__(self):
        for i in self.frames:
            yield self.read_frame(number=i)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self._frame_generator(key)
        elif isinstance(key, int):
            return self.read_frame(number=key)
        else:
            raise TypeError

    def __len__(self):
        return self.nframes

    def __str__(self):
        return "Video: size={s}, nframes={n}, fps={fps}".format(
            s=self.size,
            n=self.nframes,
            fps=self.fps
        )

    def __del__(self):
        self._capture.release()

    def _frame_generator(self, slice):
        """Auxiliary generator to return specific frames."""
        for i in self.frames[slice]:
            yield self.read_frame(number=i)

    @property
    def fourcc(self):
        """4-character code of codec."""
        fourcc = int(self._capture.get(cv2.CAP_PROP_FOURCC))
        return "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    @property
    def nframes(self):
        """Returns the total number of frames."""
        return int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def size(self):
        """Returns the size of the video frames: (width, height)."""
        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)

    @property
    def fps(self):
        """Frames per second."""
        return int(self._capture.get(cv2.CAP_PROP_FPS))

    @property
    def frame_number(self):
        """Number of the frame that will be read next."""
        return int(self._capture.get(cv2.CAP_PROP_POS_FRAMES))

    @frame_number.setter
    def frame_number(self, value):
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, value)

    @property
    def frames(self):
        """Returns an iterator with all frames."""
        return range(self.nframes)

    def generate_background_model(self, step=None, end=None, mse_min=50):
        """Generates a background model using the median.

        Only sufficiently different frames are considered, using the
        mean squared error method.

        Parameters:
            step: Step to iterate through the video. Default is video FPS rate.
            end: Last frame to consider. Default is 2/3 of video length.
            mse_min: The minimum error at wich the frame is selected. The
                lower the error, the more *similar* the two images are.
        """

        step = step or self.fps
        end = end or int(self.nframes * (2 / 3))

        # Select the good frames to compute the background model
        logger.info(
            "Selecting frames (step={}, end={}, mse_min={})".format(
                step, end, mse_min)
        )

        first_frame = self.read_frame(number=0, grayscale=True)
        selected_frames = [first_frame.image]

        for i in range(1, end, step):
            frame = self.read_frame(number=i, grayscale=True)
            mse = _mean_squared_error(frame.image, selected_frames[-1])
            if mse < mse_min:
                continue
            else:
                selected_frames.append(frame.image)

        logger.info(
            "Generating the background model using {} frames".format(
                len(selected_frames))
        )

        bgmodel = np.median(
            np.dstack(selected_frames), axis=2).astype(np.uint8)

        return bgmodel

    def read_frame(self, number=None, grayscale=False):
        """Reads the current frame and returns it.
        You can also ask for a specific frame.
        Returns a Frame object.

        :param int number: Number of the frame desired. If None, reads
                           the current one.
        :param bool grayscale: Convert the frame read to grayscale.
        """
        assert(self._capture.isOpened())

        if number is not None:
            self.frame_number = number
        else:
            number = self.frame_number

        logger.debug('Reading frame %d' % number)

        reading_success, image = self._capture.read()

        if reading_success is True:
            if grayscale or self.grayscale:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return Frame(number, image)
        else:
            raise Exception("Failed to read frame.")

    def show_frame(self, number=None, window=None, resize=False):
        """Shows frame `number` in an OpenCV window.
        Returns the frame read.
        """

        if number is not None:
            self.frame_number = number
        else:
            number = self.frame_number

        if window is None:
            window = 'Frame {}'.format(number)

        frame = self.read_frame(number)

        if resize:
            image = cv2.resize(frame.image,
                               dsize=(0, 0),
                               fx=0.5,
                               fy=0.5,
                               interpolation=cv2.INTER_AREA)
        else:
            image = frame.image

        cv2.imshow(window, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return frame

    def play(self, begin=None, end=None, step=1, window=None, wait_time=None):
        if begin is None:
            begin = 0
        if end is None:
            end = self.nframes
        if window is None:
            window = 'Playing Video'
        if wait_time is None:
            wait_time = int(1000 / self.fps)

        for i in self.frames[begin:end:step]:
            frame = self.read_frame(i)
            cv2.putText(frame.image,
                        "Frame " + str(i),
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        1,
                        cv2.LINE_AA)
            cv2.imshow(window, frame.image)
            key = cv2.waitKey(wait_time) & 0xff
            if key == 27:
                break
        cv2.destroyAllWindows()


# 0 - CV_CAP_PROP_POS_MSEC
#   Current position of the video file in milliseconds or video capture
#   timestamp.

# 1 - CV_CAP_PROP_POS_FRAMES
#   0-based index of the frame to be decoded/captured next.

# 2 - CV_CAP_PROP_POS_AVI_RATIO
#   Relative position of the video file: 0 - start of the film, 1 - end
#   of the film.

# 3 - CV_CAP_PROP_FRAME_WIDTH
#   Width of the frames in the video stream.

# 4 - CV_CAP_PROP_FRAME_HEIGHT
#   Height of the frames in the video stream.

# 5 - CV_CAP_PROP_FPS
#   Frame rate.

# 6 - CV_CAP_PROP_FOURCC
#   4-character code of codec.

# 7 - CV_CAP_PROP_FRAME_COUNT
#   Number of frames in the video file.

# 8 - CV_CAP_PROP_FORMAT
#   Format of the Mat objects returned by retrieve() .

# 9 - CV_CAP_PROP_MODE
#   Backend-specific value indicating the current capture mode.

# 10 - CV_CAP_PROP_BRIGHTNESS
#   Brightness of the image (only for cameras).

# 11 - CV_CAP_PROP_CONTRAST
#   Contrast of the image (only for cameras).

# 12 - CV_CAP_PROP_SATURATION
#   Saturation of the image (only for cameras).

# 13 - CV_CAP_PROP_HUE
#   Hue of the image (only for cameras).

# 14 - CV_CAP_PROP_GAIN
#   Gain of the image (only for cameras).

# 15 - CV_CAP_PROP_EXPOSURE
#   Exposure (only for cameras).

# 16 - CV_CAP_PROP_CONVERT_RGB
#   Boolean flags indicating whether images should be converted to RGB.

# 17 - CV_CAP_PROP_WHITE_BALANCE
#   Currently not supported

# 18 - CV_CAP_PROP_RECTIFICATION
#   Rectification flag for stereo cameras (note: only supported by
#   DC1394 v 2.x backend currently)


class Frame:

    def __init__(self, number, image=None):
        self._number = number
        self._image = image

    def __repr__(self):
        return('Frame({})'.format(self.number))

    @property
    def number(self):
        return self._number

    @property
    def image(self):
        return self._image
