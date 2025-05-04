from src.data_preprocessing import LandcoverDataset, split_dataset
from src.inference import generate_predictions, inference_and_evaluate
from src.train import train_model


def main():
    # 1) 데이터셋 분할
    split_dataset(
        raw_img_dir="data/raw/images",
        raw_mask_dir="data/raw/masks",
        output_dir="data",
        train_ratio=0.7,
        val_ratio=0.2,
        seed=42,
        overwrite=False,
    )

    # 2) 모델 타입 ('kmeans' or 'rf' or 'miniunet')
    model_type = "miniunet"

    # 3) Dataset 준비
    #   - use_clahe=True -> train 데이터에 CLAHE(히스토그램 평활화) 일부 적용
    train_dataset = LandcoverDataset(
        "data/train/images", "data/train/masks", phase="train", use_clahe=True
    )
    val_dataset = LandcoverDataset(
        "data/val/images", "data/val/masks", phase="val", use_clahe=False
    )
    test_dataset = LandcoverDataset(
        "data/test/images", "data/test/masks", phase="test", use_clahe=False
    )

    print("모델 학습 시작")

    # 4) 모델 학습
    model = train_model(
        model_type=model_type,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        # n_clusters=5,
        epochs=2,
        # max_batches=500,
        # n_estimators=100,
    )

    print("모델 학습 완료")

    # 5) 추론 & 평가
    inference_and_evaluate(
        model,
        test_dataset,
        model_type,
        save_pred_dir="predictions",
        csv_file="detailed_results.csv",
    )

    # 5) 추론 결과 저장 (평가 없이 저장만 하려고 할 때 사용. AI 해커톤 결과 제출 시 사용)
    """
    generate_predictions(
        model,
        test_dataset,
        model_type,
        output_dir="predictions",
    )
    """


if __name__ == "__main__":
    main()
