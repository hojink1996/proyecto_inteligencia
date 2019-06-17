"""
Proyecto Final de Imagenes del Ramo EL5206-1 Laboratorio de Inteligencia Computacional.

@authors: Hojin Kang and Eduardo Salazar
"""

import cv2
import numpy as np

from skimage.measure import compare_ssim, shannon_entropy
from scipy.stats import skew


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
        self._frame_operator = FrameOperator('imagen_base.jpeg', root_path)

        # Lists where we will save the values
        self._luminance = []
        self._faces = []
        self._quality_check = []
        self._ssim = []
        self._energies = []
        self._entropies = []
        self._mean_r = []
        self._mean_g = []
        self._mean_b = []
        self._mean_y = []
        self._mean_cb = []
        self._mean_cr = []
        self._skewness = []

    def obtain_values(self) -> (list, list, list, list, list, list, list, list, list, list, list):
        """
        Calculates all the values needed for the classification of the video.

        :return:    A tuple of lists needed for the classification. In order we have the list of luminance, ssim,
                    energy, entropy, mean for the r, g, b, y, cb and cr channels and skewness. The lists contain
                    the values for each of the frames.
        """
        self._calculate_values()
        return (self._luminance, self._ssim, self._energies, self._entropies, self._mean_r, self._mean_g, self._mean_b,
                self._mean_y, self._mean_cb, self._mean_cr, self._skewness)

    def _calculate_values(self):
        """
        Calculates the parameters we need for the evaluation.
        """
        self._calculate_luminance()
        self._calculate_ssim()
        self._calculate_energy()
        self._calculate_entropy()
        self._calculate_skewness()
        self._calculate_mean_ycbcr()

    def _calculate_skewness(self) -> list:
        """
        Calculate the skewness of each frame of the video.

        :return:    A list with the skewness of each frame.
        """
        if len(self._faces) == 0:
            self._calculate_luminance()

        # Get the YCbCr values from our RGB values
        if len(self._skewness) == 0:
            for face in self._quality_check:
                self._skewness.append(self._frame_operator.calculate_skewness(face))

        return self._skewness

    def _calculate_mean_ycbcr(self) -> (list, list, list):
        """
        Calculates the mean values for the YCbCr channels of each frame.

        :return:    A tuple with the mean values of the Y, Cb and Cr channels respectively.
        """
        if len(self._mean_r) == 0:
            self._calculate_luminance()

        # Get the YCbCr values from our RGB values
        if len(self._mean_y) == 0:
            for r, g, b in zip(self._mean_r, self._mean_g, self._mean_b):
                self._mean_y.append(0.258*r + 0.504*g + 0.098*b + 16)
                self._mean_cb.append(-0.148*r - 0.291*g + 0.439*b + 128)
                self._mean_cr.append(0.439*r - 0.368*g - 0.071*b + 128)

        return self._mean_y, self._mean_cb, self._mean_cr

    def _detect_faces(self) -> list:
        """
        Detect the faces in each frame of the video.

        :return:        List with only the faces cropped from each frame.
        """
        # While the video is opened
        while True:
            ret, frame = self._video.read()

            # Detect the face in the frame
            if ret:
                imgs = self._frame_operator.detect_face(frame)

                # No face was detected
                if imgs is None:
                    continue
                else:
                    cropped_img, base_img = imgs

                self._faces.append(cropped_img)
                self._quality_check.append(base_img)

            # We have no more images
            else:
                break

        return self._quality_check

    def _calculate_luminance(self) -> (list, list, list, list):
        """
        Calculates the luminance and the average value per channel of the frames.

        :return:    A tuple containing the list of the values for the luminance per frame in the first position,
                    and the average of the R, G and B channels respectively in the rest of the channels.
        """
        if len(self._faces) == 0:
            self._detect_faces()

        if len(self._luminance) == 0:
            for face in self._faces:
                luminance_value, r, g, b = self._frame_operator.calculate_luminance(face)
                self._luminance.append(luminance_value)
                self._mean_r.append(r)
                self._mean_g.append(g)
                self._mean_b.append(b)

        return self._luminance, self._mean_r, self._mean_g, self._mean_b

    def _calculate_ssim(self) -> list:
        """
        Calculates the ssim for the frames

        :return:    A list with the ssim of each of the frames of the video
        """
        if len(self._faces) == 0:
            self._detect_faces()

        if len(self._ssim) == 0:
            for face in self._quality_check:
                self._ssim.append(self._frame_operator.calculate_ssim(face))

        return self._ssim

    def _calculate_energy(self) -> list:
        """
        Calculate the energies for the frames

        :return:    A list with the energy of each of the frames of the video
        """
        if len(self._faces) == 0:
            self._detect_faces()

        if len(self._energies) == 0:
            for face in self._quality_check:
                self._energies.append(self._frame_operator.calculate_energy(face))

        return self._energies

    def _calculate_entropy(self) -> list:
        """
        Calculate the entropy for the frames.

        :return:    A list with the entropy of each of the frames of the video
        """
        if len(self._faces) == 0:
            self._detect_faces()

        if len(self._entropies) == 0:
            for face in self._quality_check:
                self._entropies.append(self._frame_operator.calculate_entropy(face))

        return self._entropies

    @staticmethod
    def play_sequence(sequence: list):
        """
        Plays a sequence of frames. To show the next frame of the sequence just press a button.

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
    def __init__(self, base_image_path, root_path):
        """
        Initializes the FrameOperator Class
        """
        self._face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self._detect_initial_face(f'{root_path}{base_image_path}')
        self._base_width, self._base_height = self._base_image.shape

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

        # Subtract old image center (bringing image back to origin) and adding the new image center coordinates
        rotation_mat[0, 2] += bound_w / 2 - image_center[0]
        rotation_mat[1, 2] += bound_h / 2 - image_center[1]

        # Rotate image with the new bounds and translated rotation matrix
        rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
        return rotated_mat

    def calculate_ssim(self, frame) -> float:
        """
        Calculates the SSIM between the frame and the base image. Note that the size of the base image and the
        frame must be the same.

        :param frame:   The frame from which to calculate the SSIM. The frame must be in gray scale.
        :return:        A float value corresponding to the SSIM
        """

        score, _ = compare_ssim(self._base_image, frame, full=True)

        return score

    def calculate_energy(self, frame) -> float:
        """
        We calculate the energy of the image. For them to be directly compareable we need them to have
        the same size, so we should use the images that are the same size as the base image.

        :param frame:   The frame from which to calculate the image (in gray scale)
        :return:        The energy of the frame
        """
        # Calculate the fft of the frame in gray scale
        fft_values = np.fft.fft2(frame)

        # Frequency array
        f = np.log(np.absolute(fft_values))

        # Get the energy
        return float(np.sum(np.square(f)))

    def _detect_initial_face(self, path):
        """
        Detect the initial face of the base image.

        :param path:    The path to the image
        """
        image = cv2.imread(path)
        width, height = image.shape[:2]
        image = cv2.resize(image, (int(height - 100), int(width - 100)))

        # Turn the image to GrayScale so we can use Viola-Jones
        gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect the face
        faces = self._face_cascade.detectMultiScale(gray_frame)

        # If any face was detected
        if (faces.shape[0]) > 1:
            vals = [faces[index, 2] + faces[index, 3] for index in range(faces.shape[0])]
            index = vals.index(max(vals))
        else:
            index = 0

        # If any face was detected
        (column, row, width, height) = faces[index]

        # Get cropped image
        self._base_image = gray_frame[row:row + height, column:column + width]

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

        # There were no faces detected
        if type(faces) is tuple:
            return None
        # If any face was detected
        if (faces.shape[0]) > 1:
            vals = [faces[index, 2] + faces[index, 3] for index in range(faces.shape[0])]
            index = vals.index(max(vals))
        else:
            index = 0

        (column, row, width, height) = faces[index, :]

        # Get cropped image
        cropped_img = frame[row:row + height, column:column+width]

        # Make it so we can't get out of the image index
        height, width = gray_frame.shape
        if row + self._base_height >= height:
            row -= (row + self._base_height - height)
        if column + self._base_width >= width:
            column -= (column + self._base_width - width)
        cropped_base_img = np.array(gray_frame)[row:row+self._base_height, column:column+self._base_width]

        return cropped_img, cropped_base_img

    def calculate_entropy(self, frame) -> float:
        """
        Calculates the Shannon's Entropy of an image.

        :param frame:   The frame from which to calculate the entropy. It must be in gray scale.
        :return:        A float corresponding to the Shannon Entropy of the image.
        """
        return shannon_entropy(frame)

    def calculate_luminance(self, frame) -> (float, float, float, float):
        """
        Calculates the luminance of the frame. This is done with the following equation:

            Luminance = 0.299 * R + 0.587 * G + 0.114 * B,

        Where R, G and B are the average values of the R, G and B channels respectively.

        This equation is from D. Garud. (2016). Face Liveliness Detection. (1)

        Also returns the mean of each channel of the image

        :param frame:   Frame from which to calculate the luminance (considered in BGR format)
        :return:        A tuple containing the luminance of the image in the first value, and the means
                        of the R, G and B channels respectively in the rest of the channels
        """
        R = frame[:, :, 2]
        G = frame[:, :, 1]
        B = frame[:, :, 0]

        return 0.299*np.mean(R) + 0.587*np.mean(G) + 0.114*np.mean(B), np.mean(R), np.mean(G), np.mean(B)

    def calculate_skewness(self, frame) -> float:
        """
        Calculate the skewness of a frame.

        :param frame:   The frame from which to calculate the skewness. The image must be in gray scale.
        :return:        A float value corresponding to the skewness of the frame.
        """
        skew_value = skew(np.array(frame).flatten())

        return skew_value

    @staticmethod
    def show_frame(frame):
        """
        Shows the frame until a button is pressed.

        :param frame:   The frame to show
        """
        cv2.imshow('Image', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

video_operator = VideoOperator('Videos/usuario_1_1.mp4')
luminance, ssim, energy, entropy, r, g, b, y, cb ,cr ,skewness = video_operator.obtain_values()
print('Done')
