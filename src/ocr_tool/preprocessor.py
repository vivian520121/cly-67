import os
import cv2
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.heic', '.heif', '.bmp', '.tiff', '.webp'}


class ImagePreprocessor:
    def __init__(
        self,
        contrast_alpha: float = 1.5,
        denoise_strength: int = 10,
        sharpen: bool = True,
        threshold: bool = False,
        deskew: bool = True,
        resize_max_width: int = 2000,
    ):
        self.contrast_alpha = contrast_alpha
        self.denoise_strength = denoise_strength
        self.sharpen = sharpen
        self.threshold = threshold
        self.deskew = deskew
        self.resize_max_width = resize_max_width

    def load_image(self, image_path: str) -> np.ndarray:
        ext = os.path.splitext(image_path)[1].lower()
        if ext in {'.heic', '.heif'}:
            pil_img = Image.open(image_path)
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            img_array = np.array(pil_img)
            return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"无法读取图片: {image_path}")
            return img

    def resize_image(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if w > self.resize_max_width:
            scale = self.resize_max_width / w
            new_h = int(h * scale)
            img = cv2.resize(img, (self.resize_max_width, new_h), interpolation=cv2.INTER_AREA)
        return img

    def enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        img = cv2.convertScaleAbs(img, alpha=self.contrast_alpha, beta=0)
        return img

    def denoise_image(self, img: np.ndarray) -> np.ndarray:
        if self.denoise_strength <= 0:
            return img
        strength = min(self.denoise_strength, 20)
        template_window_size = 7
        search_window_size = 21
        denoised = cv2.fastNlMeansDenoisingColored(
            img, None, strength, strength, template_window_size, search_window_size
        )
        return denoised

    def sharpen_image(self, img: np.ndarray) -> np.ndarray:
        if not self.sharpen:
            return img
        kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ])
        sharpened = cv2.filter2D(img, -1, kernel)
        return sharpened

    def apply_threshold(self, img: np.ndarray) -> np.ndarray:
        if not self.threshold:
            return img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    def deskew_image(self, img: np.ndarray) -> np.ndarray:
        if not self.deskew:
            return img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 100:
            return img
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) < 0.5:
            return img
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    def detect_blur(self, img: np.ndarray) -> tuple[float, bool]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        threshold = 100.0
        is_blurry = laplacian_var < threshold
        return laplacian_var, is_blurry

    def process(self, image_path: str) -> tuple[np.ndarray, dict]:
        img = self.load_image(image_path)
        img = self.resize_image(img)
        blur_score, is_blurry = self.detect_blur(img)
        img = self.enhance_contrast(img)
        img = self.denoise_image(img)
        img = self.sharpen_image(img)
        img = self.apply_threshold(img)
        img = self.deskew_image(img)
        metadata = {
            'blur_score': blur_score,
            'is_blurry': is_blurry,
            'original_size': (img.shape[1], img.shape[0]),
        }
        return img, metadata

    def save_preview(self, img: np.ndarray, output_path: str):
        cv2.imwrite(output_path, img)


def get_image_files(input_path: str) -> list[str]:
    image_files = []
    if os.path.isfile(input_path):
        ext = os.path.splitext(input_path)[1].lower()
        if ext in SUPPORTED_FORMATS:
            image_files.append(input_path)
    elif os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in SUPPORTED_FORMATS:
                    image_files.append(os.path.join(root, file))
    image_files.sort()
    return image_files
