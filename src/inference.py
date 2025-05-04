import csv
import os

import numpy as np
import torch
from PIL import Image

from src.data_preprocessing import GRAY_TO_INDEX
from src.evaluate import compute_iou
from src.model_traditional import extract_features


def inference_kmeans(model, img_tensor):
    """
    - model: 학습된 KMeansSegmentation 인스턴스
    - img_tensor: (3,H,W) float
    return: (H,W) np array (pixel label)
    """
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()  # (H,W,3)
    H, W, C = img_np.shape
    pred = model.predict(img_np.reshape(-1, 3)).reshape(H, W)
    return pred


def inference_rf(model, img_tensor):
    """
    - model: 학습된 RandomForestSegmentation 인스턴스
    - img_tensor: (3,H,W) torch.Tensor
    return: (H,W) np array (pixel label)
    """
    # 텐서를 NumPy 배열로 변환
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()  # (H,W,3)

    # 특성 추출
    feats = extract_features(img_np)  # shape(H,W,F)

    H, W, F = feats.shape
    feats_2d = feats.reshape(-1, F)

    # 분류기 예측 (RandomForestSegmentation 클래스 호출)
    pred_1d = model.predict(feats_2d)
    pred_2d = pred_1d.reshape(H, W)
    return pred_2d


def inference_miniunet(model, img_tensor):
    """
    - model: 학습된 MiniUNet 인스턴스
    - img_tensor: (3,H,W) torch.Tensor
    return: (H,W) np array (pixel label)
    """
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        inp = img_tensor.unsqueeze(0).to(device)  # (1,3,H,W)
        out = model(inp)  # (1,9,H,W)
        pred = out.argmax(dim=1).cpu().squeeze(0).numpy()
    return pred


def generate_predictions(
    model,
    test_images,
    model_type="miniunet",
    output_dir="inference_outputs",
    return_preds=False,
):
    """
    - model: 학습된 모델 객체 (K-Means, RandomForest 또는 MiniUNet)
    - test_images: 추론하고자 하는 image set
    - model_type: "kmeans", "rf" 또는 "miniunet"
    - output_dir: 예측 마스크를 저장할 폴더
    - return_preds: 예측 마스크를 반환할지 여부 (inference_and_evaluate에서 사용)

    이 함수는 예측된 마스크(픽셀값=클래스 인덱스)를 .png 파일로 저장하고,
    필요시 예측 결과를 리스트로 반환합니다.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 반환할 예측 마스크와 파일 경로 리스트 초기화
    pred_masks = []
    pred_filepaths = []

    # 더 잘 구분되는 그레이스케일 값 정의 (0->10, 1->20, ... 8->100)
    gray_values = [10, 20, 30, 40, 50, 60, 70, 80, 100]

    for i, img_tensor in enumerate(test_images):
        # 모델 추론
        if model_type == "kmeans":
            pred_mask = inference_kmeans(model, img_tensor)
        elif model_type == "rf":
            pred_mask = inference_rf(model, img_tensor)
        else:
            pred_mask = inference_miniunet(model, img_tensor)

        # 원본 예측 마스크 저장 (반환용)
        if return_preds:
            pred_masks.append(pred_mask)

        # 시각화를 위해 클래스 값을 더 잘 구분되는 그레이스케일로 변환
        vis_mask = np.zeros_like(pred_mask, dtype=np.uint8)
        for class_idx, gray_value in enumerate(gray_values):
            vis_mask[pred_mask == class_idx] = gray_value

        # 시각화용 마스크 저장
        pred_img = Image.fromarray(vis_mask)
        filename = f"pred_{i:03d}.png"
        filepath = os.path.join(output_dir, filename)
        pred_img.save(filepath)

        # 파일 경로 저장 (반환용)
        if return_preds:
            pred_filepaths.append(filepath)

    print(
        f"[generate_predictions] 총 {len(test_images)}장의 예측 마스크를 '{output_dir}' 폴더에 저장했습니다."
    )

    if return_preds:
        return pred_masks, pred_filepaths


def inference_and_evaluate(
    model,
    test_dataset,
    model_type="miniunet",
    save_pred_dir="predictions",
    csv_file="detailed_results.csv",
):
    """
    - test_dataset: (이미지, 정답 마스크) Dataset
    - model_type: "kmeans", "rf" 또는 "miniunet"
    - save_pred_dir: 예측 마스크를 저장할 폴더
    - csv_file: 이미지별 IoU 등 기록할 CSV 파일 이름

    1) generate_predictions 함수를 활용하여 예측 마스크 생성
    2) 정답 마스크와 비교하여 IoU 계산
    3) 결과를 CSV에 기록
    4) 전체 평균 mIoU 출력 및 반환
    """
    # 테스트 이미지 리스트 준비
    test_images = [img for img, _ in test_dataset]

    # generate_predictions 함수를 활용하여 예측 마스크 생성
    pred_masks, pred_filepaths = generate_predictions(
        model,
        test_images,
        model_type=model_type,
        output_dir=save_pred_dir,
        return_preds=True,
    )

    # 클래스 개수 (예: 9) - GRAY_TO_INDEX 딕셔너리 이용
    num_classes = len(GRAY_TO_INDEX)

    # IoU 계산 및 결과 기록
    results = []
    sum_iou = 0.0

    for idx, (pred_mask, pred_path) in enumerate(zip(pred_masks, pred_filepaths)):
        # 정답 마스크 가져오기
        _, gt_mask = test_dataset[idx]
        gt_np = gt_mask.numpy()

        # IoU 계산
        iou = compute_iou(pred_mask, gt_np, num_classes=num_classes)
        sum_iou += iou

        # 결과 기록
        results.append({"iou": iou, "pred_mask_path": pred_path})

    # 평균 mIoU 계산
    avg_iou = sum_iou / len(test_dataset)
    print(f"[Evaluation] Total {len(test_dataset)} images, mean IoU={avg_iou:.4f}")

    # CSV로 저장 (이미지별 IoU, 예측 마스크 경로)
    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["iou", "pred_mask_path"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"Saved detailed results to {csv_file}.")
    return avg_iou
