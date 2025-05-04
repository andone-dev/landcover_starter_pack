# src/data_preprocessing.py

import glob
import os
import random
import shutil

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset


def split_dataset(
    raw_img_dir,
    raw_mask_dir,
    output_dir,
    train_ratio=0.7,
    val_ratio=0.2,
    seed=42,
    overwrite=False,
):
    """
    1) raw 폴더 내 이미지/마스크 파일을 train/val/test로 분할
    2) 분할 비율(70/20/10)로 복사

    Args:
        raw_img_dir: 원본 이미지 디렉토리
        raw_mask_dir: 원본 마스크 디렉토리
        output_dir: 출력 디렉토리
        train_ratio: 학습 데이터 비율
        val_ratio: 검증 데이터 비율
        seed: 랜덤 시드
        overwrite: True일 경우 기존 데이터 덮어쓰기, False일 경우 이미 존재하면 스킵
    """
    # 이미 데이터가 나눠져 있는지 확인
    train_dir = os.path.join(output_dir, "train", "images")
    val_dir = os.path.join(output_dir, "val", "images")
    test_dir = os.path.join(output_dir, "test", "images")

    if (
        os.path.exists(train_dir)
        and os.path.exists(val_dir)
        and os.path.exists(test_dir)
    ):
        # 디렉토리가 모두 존재하면 파일이 있는지 확인
        train_files = glob.glob(os.path.join(train_dir, "*"))
        val_files = glob.glob(os.path.join(val_dir, "*"))
        test_files = glob.glob(os.path.join(test_dir, "*"))

        if len(train_files) > 0 and len(val_files) > 0 and len(test_files) > 0:
            if not overwrite:
                print(
                    f"[split_dataset] 이미 데이터가 나눠져 있습니다. train={len(train_files)}, val={len(val_files)}, test={len(test_files)}"
                )
                print(
                    f"[split_dataset] 데이터를 다시 나누려면 overwrite=True로 설정하세요."
                )
                return
            else:
                print(f"[split_dataset] 기존 데이터를 제거하고 다시 나눕니다.")
                # 기존 데이터 제거
                for phase in ["train", "val", "test"]:
                    img_dir = os.path.join(output_dir, phase, "images")
                    mask_dir = os.path.join(output_dir, phase, "masks")
                    if os.path.exists(img_dir):
                        shutil.rmtree(img_dir)
                    if os.path.exists(mask_dir):
                        shutil.rmtree(mask_dir)

    random.seed(seed)
    img_paths = sorted(glob.glob(os.path.join(raw_img_dir, "*")))
    mask_paths = sorted(glob.glob(os.path.join(raw_mask_dir, "*")))

    data_pairs = list(zip(img_paths, mask_paths))
    random.shuffle(data_pairs)

    n_total = len(data_pairs)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_set = data_pairs[:n_train]
    val_set = data_pairs[n_train : n_train + n_val]
    test_set = data_pairs[n_train + n_val :]

    for phase, subset in zip(["train", "val", "test"], [train_set, val_set, test_set]):
        os.makedirs(os.path.join(output_dir, phase, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, phase, "masks"), exist_ok=True)
        for img, msk in subset:
            shutil.copy(img, os.path.join(output_dir, phase, "images"))
            shutil.copy(msk, os.path.join(output_dir, phase, "masks"))

    print(
        f"[split_dataset] train={len(train_set)}, val={len(val_set)}, test={len(test_set)}"
    )


#####################################
# 전처리 / 증강
#####################################

# graysacle -> ann_code
GRAY_TO_INDEX = {
    10: 0,  # 건물
    20: 1,  # 주차장
    30: 2,  # 도로
    40: 3,  # 가로수
    50: 4,  # 논
    60: 5,  # 밭
    70: 6,  # 산림
    80: 7,  # 나지
    100: 8,  # 비대상지
}


def grayscale_to_class(mask_arr):
    """
    그레이스케일 마스크를 클래스 인덱스로 변환합니다.

    Args:
        mask_arr: (H,W) np.array, 픽셀값 {10,20,30,40,50,60,70,80,100}

    Returns:
        (H,W) np.array, 클래스 인덱스 (0~8)

    Note:
        - 10: 건물 -> 0
        - 20: 주차장 -> 1
        - 30: 도로 -> 2
        - 40: 가로수 -> 3
        - 50: 논 -> 4
        - 60: 밭 -> 5
        - 70: 산림 -> 6
        - 80: 나지 -> 7
        - 100: 비대상지 -> 8
    """
    H, W = mask_arr.shape
    result = np.zeros((H, W), dtype=np.uint8)

    # 유효하지 않은 값 확인
    unique_values = np.unique(mask_arr)
    unknown_values = [v for v in unique_values if v not in GRAY_TO_INDEX]
    if unknown_values:
        print(f"경고: 알 수 없는 그레이스케일 값이 발견되었습니다: {unknown_values}")

    for gval, idx in GRAY_TO_INDEX.items():
        result[mask_arr == gval] = idx

    return result


class LandcoverDataset(Dataset):
    """
    - CLAHE 알고리즘 등 전처리를 on-the-fly로 시연
    - phase='train'일 때만 CLAHE나 히스토그램 평활화가 일부 적용
    """

    def __init__(self, img_dir, mask_dir, phase="train", use_clahe=False, img_size=512):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*")))

        # 이미지와 마스크 파일명 매칭 확인
        img_basenames = [os.path.basename(p) for p in self.img_paths]
        mask_basenames = [os.path.basename(p) for p in self.mask_paths]
        assert len(self.img_paths) == len(
            self.mask_paths
        ), "이미지와 마스크 개수가 일치하지 않습니다."
        assert all(
            [
                i.split(".")[0] == m.split("_FGT.")[0]
                for i, m in zip(sorted(img_basenames), sorted(mask_basenames))
            ]
        ), "이미지와 마스크 파일명이 일치하지 않습니다."

        self.phase = phase
        self.use_clahe = use_clahe
        self.img_size = img_size

        if phase == "train":
            # Albumentations 파이프라인 (CLAHE 사용)
            # p=0.3 => 30% 확률 적용
            self.transform = A.Compose(
                [
                    A.Resize(img_size, img_size),
                    A.CLAHE(clip_limit=(1, 4), p=0.3) if use_clahe else A.NoOp(),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=0.3
                    ),
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3
                    ),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
        else:
            # Albumentations 파이프라인 (CLAHE 사용 X)
            self.transform = A.Compose(
                [
                    A.Resize(img_size, img_size),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        msk_path = self.mask_paths[idx]

        try:
            # 이미지: RGB
            img = Image.open(img_path).convert("RGB")
            img_arr = np.array(img)

            # 마스크: 그레이 -> np
            mask_gray = Image.open(msk_path).convert("L")
            mask_arr = np.array(mask_gray)
            mask_index = grayscale_to_class(mask_arr)  # 클래스 인덱스로 변환

            # Albumentations transform
            transformed = self.transform(image=img_arr, mask=mask_index)
            return transformed["image"], transformed["mask"]

        except Exception as e:
            print(f"데이터 로딩 오류: {img_path} / {msk_path}")
            print(f"오류 내용: {str(e)}")
            # 오류 발생 시 대체 데이터 반환
            if idx > 0:
                return self.__getitem__(idx - 1)  # 이전 인덱스로 시도
            else:
                # 더미 데이터 생성
                dummy_img = torch.zeros(3, self.img_size, self.img_size)
                dummy_mask = torch.zeros(self.img_size, self.img_size, dtype=torch.long)
                return dummy_img, dummy_mask
