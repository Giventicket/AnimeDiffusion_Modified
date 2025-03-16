from pytorch_fid import fid_score
import torch

# 이미지 폴더 경로 설정
same_images_path = '/data/Anime/test_data/reference'
diff_images_path = '/data/Anime/test_data_shuffled/reference'
same_generated_images_path = './result_same_mse+lpips'
diff_generated_images_path = './result_diff_mse+lpips'

# GPU 사용 가능 여부 확인
device = 1

# FID 계산
same_fid_value = fid_score.calculate_fid_given_paths(
    [same_images_path, same_generated_images_path],
    batch_size=50,
    device=device,
    dims=2048
)

# FID 계산
diff_fid_value = fid_score.calculate_fid_given_paths(
    [diff_images_path, diff_generated_images_path],
    batch_size=50,
    device=device,
    dims=2048
)

print(f"✅ sketch = reference FID Score: {same_fid_value}")
print(f"✅ sketch != reference FID Score: {diff_fid_value}")
