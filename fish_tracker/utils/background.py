import cv2

from fish_tracker.utils.logger import get_logger


class BackgroundSubtractor:

    def __init__(
        self,
        reference_frame=None,
        gaussian_blur_size=21,
        binary_threshold=30,
        kernel_size=5,  # for morphological transformations
        dilatation_iter=5,
        morphological_gradient_iter=1,
        log_level="INFO",
    ):

        self.gaussian_blur_size = gaussian_blur_size
        self.binary_threshold = binary_threshold
        self.kernel_size = kernel_size
        self.dilatation_iter = dilatation_iter
        self.morphological_gradient_iter = morphological_gradient_iter
        self.logger = get_logger("BackgroundSubstractor")
        self.logger.info("BackgroundSubstractor Initialization")
        if isinstance(reference_frame, str):
            self.ref = cv2.imread(reference_frame)
        else:
            self.ref = reference_frame
        self.logger.debug(type(self.ref))
        self.delta_frame = None

        self.set_reference_frame()

    def convert_to_grayscale(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(
            gray, (self.gaussian_blur_size, self.gaussian_blur_size), 0
        )
        return gray

    def set_reference_frame(self):
        self.ref = self.convert_to_grayscale(self.ref)

    def set_delta_frame(self, frame):

        if self.ref is None:
            raise ValueError("Reference frame not set. Call set_reference_frame first.")

        # Grayscale conversion
        gray = self.convert_to_grayscale(frame)

        # Difference
        delta_frame = cv2.absdiff(self.ref, gray)

        # Binary
        delta_frame = cv2.threshold(
            delta_frame,
            self.binary_threshold,
            255,
            cv2.THRESH_BINARY,
            # 255 -> white
        )[1]

        # Kernel for morpholofical transformations
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size)
        )
        # Dilatation
        delta_frame = cv2.dilate(delta_frame, kernel, iterations=self.dilatation_iter)
        # Morphological Gradient to get contours
        self.delta_frame = cv2.morphologyEx(
            delta_frame,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=self.morphological_gradient_iter,
        )
