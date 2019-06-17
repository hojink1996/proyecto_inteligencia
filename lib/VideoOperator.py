import cv2
import numpy as np


class VideoOperator:
    """
    Class for the operations done to the video.
    """
    def __init__(self, video_path, root_path='C:/Users/Hojin/PycharmProjects/proyecto_inteligencia/'):
        """
        Initialize the VideoOperator Class

        :param video_path:      The path to the video over which we are operating
        """
        self._video = cv2.VideoCapture(f'{root_path}{video_path}')
        self._frame_operator = FrameOperator()

    def detect_faces(self) -> list:
        """
        Detect the faces in each frame of the video.

        :return:        List with only the faces cropped from each frame.
        """
        frames = []
        # While the video is opened
        while True:
            ret, frame = self._video.read()

            # Detect the face in the frame
            if ret:
                frames.append(self._frame_operator.detect_face(frame))

            # We have no more images
            else:
                break

        return frames

    @staticmethod
    def play_sequence(sequence: list):
        """
        Plays a sequence of frames

        :param sequence:    The sequence to be played
        """
        for frame in sequence:
            cv2.imshow('Image', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


class FrameOperator:
    """
    Class for the operations done to a single frame of the video.
    """
    def __init__(self):
        """
        Initializes the FrameOperator Class
        """
        self._face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def rotate_image(self, mat, angle):
        """
        Rotates an image (angle in degrees) and expands image to avoid cropping
        """

        height, width = mat.shape[:2]  # image shape has 3 dimensions
        image_center = (width / 2,
                        height / 2)

        rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

        # Rotation calculates the cos and sin, taking absolutes of those.
        abs_cos = abs(rotation_mat[0, 0])
        abs_sin = abs(rotation_mat[0, 1])

        # Find the new width and height bounds
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        # Subtract old image center (bringing image back to origo) and adding the new image center coordinates
        rotation_mat[0, 2] += bound_w / 2 - image_center[0]
        rotation_mat[1, 2] += bound_h / 2 - image_center[1]

        # Rotate image with the new bounds and translated rotation matrix
        rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
        return rotated_mat

    def detect_face(self, frame):
        """
        Detects the face in a single frame of a video.

        :param frame:   The frame from which to detect the face.
        :return:        The cropped image of only the face.
        """
        frame = self.rotate_image(frame, 270)
        # Turn the image to GrayScale so we can use Viola-Jones
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect the face
        faces = self._face_cascade.detectMultiScale(gray_frame)
        (column, row, width, height) = faces[0]

        # Get cropped image
        cropped_img = frame[row:row + height, column:column+width]

        return cropped_img


video_operator = VideoOperator('Videos/usuario_1_1.mp4')
faces = video_operator.detect_faces()
VideoOperator.play_sequence(faces)
print('Done')



