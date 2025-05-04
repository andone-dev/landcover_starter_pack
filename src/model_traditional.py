import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


def extract_features(img: np.ndarray) -> np.ndarray:
    """
    img: (H,W,3) np.uint8, BGR 혹은 RGB
    return: (H,W,F) float - 각 픽셀마다 F차원 특징
    """
    # 1) RGB -> HSV
    #   opencv의 cvtColor가 BGR 전제이므로, 필요한 경우 img[..., ::-1]로 BGR->RGB 교정
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # shape(H,W,3)

    # 2) Sobel 에지
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    # shape( H, W, F=8 ) => (R, G, B, H, S, V, SobelX, SobelY)
    #   R/G/B/H/S/V: float in [0..255] or [0..1], sobel in some range
    #   실제 구현에서 scaling/normalization할 수 있음
    R = img[:, :, 0].astype(np.float32)
    G = img[:, :, 1].astype(np.float32)
    B = img[:, :, 2].astype(np.float32)

    H = hsv[:, :, 0].astype(np.float32)
    S = hsv[:, :, 1].astype(np.float32)
    V = hsv[:, :, 2].astype(np.float32)

    F = np.stack([R, G, B, H, S, V, sobelx, sobely], axis=-1)
    return F


class KMeansSegmentation:
    """
    색상(RGB) 기반 K-Means 군집화 세그멘테이션.
    딥러닝 이전 자주 사용되던 방식으로 CPU만 있는 환경에서도 빠르게 동작 가능.
    """

    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.model = None

    def fit(self, pixel_samples):
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.model.fit(pixel_samples)

    def predict(self, pixel_data):
        return self.model.predict(pixel_data)
