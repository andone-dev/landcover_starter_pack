from sklearn.cluster import KMeans


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
