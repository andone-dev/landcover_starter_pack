import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .evaluate import compute_iou  # mIoU 계산 함수 import
from .model_lightweight import MiniUNet
from .model_traditional import KMeansSegmentation


def train_kmeans(train_dataset, n_clusters=5):
    """KMeans 세그멘테이션 모델 학습"""
    # 픽셀 샘플링
    pixel_samples = []
    for i in range(len(train_dataset)):
        img_t, _ = train_dataset[i]
        img_np = img_t.permute(1, 2, 0).numpy()  # (H,W,3)
        H, W, C = img_np.shape
        s_count = min(H * W, 5000)
        coords = random.sample(range(H * W), s_count)
        sub_px = img_np.reshape(-1, 3)[coords]
        pixel_samples.append(sub_px)

    pixel_samples = np.concatenate(pixel_samples, axis=0)
    print(f"KMeans 학습용 픽셀 샘플: {pixel_samples.shape}")

    # 모델 학습
    kmeans = KMeansSegmentation(n_clusters=n_clusters)
    kmeans.fit(pixel_samples)
    return kmeans


def validate_miniunet(model, val_loader, criterion, device, max_batches=None):
    """Mini U-Net 모델 검증 및 mIoU 계산"""
    model.eval()
    val_loss = 0
    val_batch_count = 0
    all_preds = []
    all_masks = []

    with torch.no_grad():
        for i, (val_images, val_masks) in enumerate(val_loader):
            if max_batches and i >= max_batches:
                break

            val_images = val_images.to(device)
            val_masks = torch.clamp(val_masks, 0, 8).long().to(device)

            val_outputs = model(val_images)
            val_batch_loss = criterion(val_outputs, val_masks)
            val_loss += val_batch_loss.item()

            val_preds = torch.argmax(val_outputs, dim=1)
            all_preds.extend(val_preds.cpu().numpy())
            all_masks.extend(val_masks.cpu().numpy())

            val_batch_count += 1

    # mIoU 계산
    avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else 0

    miou_sum = 0
    for pred, mask in zip(all_preds, all_masks):
        miou_sum += compute_iou(pred, mask, num_classes=9)
    mean_iou = miou_sum / len(all_preds) if all_preds else 0

    return {"loss": avg_val_loss, "miou": mean_iou, "batch_count": val_batch_count}


def train_miniunet(train_dataset, val_dataset=None, epochs=5, max_batches=None):
    """Mini U-Net 세그멘테이션 모델 학습"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MiniUNet(num_classes=9).to(device)

    # 옵티마이저 및 손실 함수 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # 데이터 로더 설정
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    print(f"학습 데이터: {len(train_dataset)}개, 배치 수: {len(train_loader)}")

    if val_dataset:
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
        print(f"검증 데이터: {len(val_dataset)}개, 배치 수: {len(val_loader)}")

    if max_batches:
        print(f"배치 제한: {max_batches}개")

    # 최적 모델 관리
    best_miou = 0.0
    best_model_state = None

    # 에폭 단위 학습
    for epoch in range(epochs):
        print(f"[Epoch {epoch+1}/{epochs}] 학습 시작")

        # 학습 단계
        model.train()
        total_loss = 0
        batch_count = 0

        for i, (images, masks) in enumerate(train_loader):
            if max_batches and i >= max_batches:
                break

            images = images.to(device)
            masks = torch.clamp(masks, 0, 8).long().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        # 학습 결과 출력
        train_loss = total_loss / batch_count if batch_count > 0 else 0
        print(
            f"[Epoch {epoch+1}/{epochs}] 학습 손실={train_loss:.4f} (배치: {batch_count}개)"
        )

        # 검증 단계
        if val_dataset:
            val_results = validate_miniunet(
                model, val_loader, criterion, device, max_batches
            )

            print(
                f"[Epoch {epoch+1}/{epochs}] 검증 손실={val_results['loss']:.4f}, "
                f"mIoU={val_results['miou']:.4f} (배치: {val_results['batch_count']}개)"
            )

            # 최적 모델 저장
            if val_results["miou"] > best_miou:
                best_miou = val_results["miou"]
                best_model_state = model.state_dict().copy()
                print(
                    f"[Epoch {epoch+1}/{epochs}] 개선된 모델 저장 (mIoU: {best_miou:.4f})"
                )

    # 최적 모델 복원
    if best_model_state and val_dataset:
        model.load_state_dict(best_model_state)
        print(f"최적 모델 복원 완료 (mIoU: {best_miou:.4f})")

    return model


def train_model(
    model_type,
    train_dataset,
    val_dataset=None,
    n_clusters=5,
    epochs=5,
    max_batches=None,
):
    """모델 타입에 따라 적절한 학습 함수 호출

    Args:
        model_type: 'kmeans' 또는 'miniunet'
        train_dataset: 학습 데이터셋
        val_dataset: 검증 데이터셋 (optional)
        n_clusters: KMeans 클러스터 수
        epochs: 학습 에폭 수
        max_batches: 배치 수 제한 (테스트용)

    Returns:
        학습된 모델
    """
    print(f"모델 타입: {model_type} 학습 시작")

    if model_type == "kmeans":
        return train_kmeans(train_dataset, n_clusters)

    elif model_type == "miniunet":
        return train_miniunet(train_dataset, val_dataset, epochs, max_batches)

    else:
        raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
