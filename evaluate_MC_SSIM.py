import os
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

# 경로 설정
real_dir = '/data/Anime/test_data/reference'
generated_dir = './result_same'

# 모든 이미지의 MC-SSIM을 저장할 리스트
mc_ssim_scores = []

# 이미지 파일 리스트 정렬
real_files = sorted(os.listdir(real_dir))
generated_files = sorted(os.listdir(generated_dir))

# SSIM 계산
for real_file, gen_file in zip(real_files, generated_files):
    real_path = os.path.join(real_dir, real_file)
    gen_path = os.path.join(generated_dir, gen_file)

    # 이미지 불러오기 (RGB)
    img_real = cv2.imread(real_path)  # BGR 형식
    img_gen = cv2.imread(gen_path)  # BGR 형식

    # 크기 맞추기
    img_gen = cv2.resize(img_gen, (img_real.shape[1], img_real.shape[0]))

    # BGR → RGB 변환 (OpenCV는 기본적으로 BGR을 사용)
    img_real = cv2.cvtColor(img_real, cv2.COLOR_BGR2RGB)
    img_gen = cv2.cvtColor(img_gen, cv2.COLOR_BGR2RGB)

    # 채널별 SSIM 계산
    ssim_per_channel = []
    for i in range(3):  # R, G, B 채널
        score, _ = ssim(img_real[:, :, i], img_gen[:, :, i], full=True)
        ssim_per_channel.append(score)

    # MC-SSIM: 각 채널 SSIM의 평균값
    mc_ssim = np.mean(ssim_per_channel)
    mc_ssim_scores.append(mc_ssim)

    print(f"{real_file} vs {gen_file} ➔ MC-SSIM: {mc_ssim:.4f}")

# 평균 MC-SSIM 계산
average_mc_ssim = np.mean(mc_ssim_scores)
print(f"📊 평균 MC-SSIM Score: {average_mc_ssim:.4f}")
