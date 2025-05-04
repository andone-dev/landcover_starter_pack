import random

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier


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


class RandomForestSegmentation:
    """
    특성 기반 RandomForest 세그멘테이션 모델.
    픽셀별 특성을 추출하여 RandomForest로 학습 및 예측합니다.
    """

    def __init__(self, n_estimators=50, max_depth=20, class_weight="balanced"):
        """
        Args:
            n_estimators: 랜덤 포레스트의 트리 개수
            max_depth: 트리의 최대 깊이
            class_weight: 클래스 가중치 설정 방식
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.model = None

    def collect_pixel_data(self, train_dataset, sample_pixels=5000, balanced=True):
        """Dataset 객체에서 특성과 라벨 수집

        Args:
            train_dataset: (이미지, 마스크) 쌍을 포함하는 Dataset 객체
            sample_pixels: 이미지당 샘플링할 픽셀 수
            balanced: 클래스 균형 샘플링 사용 여부

        Returns:
            X: 특성 행렬 (samples, features)
            y: 라벨 벡터 (samples,)
        """
        X_list, y_list = [], []

        for i in range(len(train_dataset)):
            img_t, mask_t = train_dataset[i]

            # 텐서를 NumPy 배열로 변환
            img_np = img_t.permute(1, 2, 0).cpu().numpy()  # (H,W,3)
            mask_np = mask_t.cpu().numpy()  # (H,W)

            # 255 값 처리 (발견된 경우)
            if np.any(mask_np == 255):
                print(f"이미지 {i}: 255 값을 0으로 변환합니다.")
                mask_np[mask_np == 255] = 0  # 배경으로 처리

            # 특성 추출
            feats = extract_features(img_np)  # shape(H,W,F)
            H, W, F = feats.shape

            # Flatten
            feats_2d = feats.reshape(-1, F)  # (H*W, F)
            label_1d = mask_np.flatten()  # (H*W,)

            if balanced:
                # 클래스별 균형 있는 샘플링
                unique_classes = np.unique(label_1d)
                samples_per_class = sample_pixels // len(unique_classes)

                for cls in unique_classes:
                    cls_indices = np.where(label_1d == cls)[0]
                    if len(cls_indices) == 0:
                        continue

                    # 각 클래스에서 동일한 수의 샘플 선택 (가능한 경우)
                    cls_count = min(samples_per_class, len(cls_indices))
                    if cls_count > 0:
                        selected_indices = np.random.choice(
                            cls_indices, cls_count, replace=False
                        )
                        X_list.append(feats_2d[selected_indices, :])
                        y_list.append(label_1d[selected_indices])
            else:
                # 기존 랜덤 샘플링 방식
                total_px = H * W
                s_count = min(sample_pixels, total_px)
                px_indices = random.sample(range(total_px), s_count)

                X_sub = feats_2d[px_indices, :]
                y_sub = label_1d[px_indices]

                X_list.append(X_sub)
                y_list.append(y_sub)

        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)

        # 클래스 분포 출력
        classes, counts = np.unique(y, return_counts=True)
        print("클래스 분포:")
        for c, count in zip(classes, counts):
            print(f"  클래스 {c}: {count}개 ({count/len(y)*100:.1f}%)")

        return X, y

    def fit(self, train_dataset, sample_pixels=5000, balanced=True):
        """
        train_dataset을 사용하여 모델 학습

        Args:
            train_dataset: (이미지, 마스크) 쌍을 포함하는 Dataset 객체
            sample_pixels: 이미지당 샘플링할 픽셀 수
            balanced: 클래스 균형 샘플링 사용 여부
        """
        # 균형 잡힌 데이터 수집
        X, y = self.collect_pixel_data(train_dataset, sample_pixels, balanced)
        print(f"수집된 학습 샘플: {X.shape[0]}개, 특성 차원: {X.shape[1]}")

        print("RandomForest 모델 학습 시작...")
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=5,  # 노드 분할을 위한 최소 샘플 수
            min_samples_leaf=2,  # 리프 노드의 최소 샘플 수
            class_weight=self.class_weight,
            random_state=42,
            n_jobs=-1,
            verbose=1,
        )
        self.model.fit(X, y)

        # 특성 중요도 출력
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]

        print("특성 중요도 순위:")
        for f in range(min(X.shape[1], 10)):  # 상위 10개 특성만 출력
            print(f"{f+1}. 특성 {indices[f]} ({importances[indices[f]]:.4f})")

        print("RandomForest 모델 학습 완료!")
        return self

    def predict(self, pixel_data):
        """
        픽셀 데이터 특성에 대한 클래스 예측

        Args:
            pixel_data: 특성 행렬 (samples, features)

        Returns:
            예측된 클래스 라벨
        """
        if self.model is None:
            raise ValueError(
                "모델이 학습되지 않았습니다. fit() 메서드를 먼저 호출하세요."
            )
        return self.model.predict(pixel_data)
