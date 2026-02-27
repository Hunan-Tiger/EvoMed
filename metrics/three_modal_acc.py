import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


from utils.dataset import get_omni_rad_closed

data = get_omni_rad_closed(truncate=1000, split='test')

# 计算CT，XRay MRI的问题索引有哪些
ct_indices = []
xray_indices = []
mri_indices = []

for i, item in enumerate(data):
    if 'ct' in item['modality_type']:
        ct_indices.append(i+1)
    elif 'x-ray' in item['modality_type']:
        xray_indices.append(i+1)
    else:
        mri_indices.append(i+1)

# 给一个failed_index列表，计算三个模态的acc
failed_indices =[4, 15, 26, 32, 41, 48, 52, 54, 59, 62, 74, 82, 89, 92, 98, 103, 132, 138, 140, 149, 151, 158, 169, 175, 181, 184, 189, 193, 199, 204, 208, 243, 247, 250, 254, 258, 264, 273, 275, 287, 301, 306, 319, 326, 334, 336, 339, 352, 354, 359, 362, 367, 369, 377, 383, 385, 386, 402, 415, 424, 433, 440, 448, 459, 465, 469, 472, 493, 494, 495, 497, 513, 515, 517, 526, 537, 539, 541, 549]


ct_failed = len(set(ct_indices) & set(failed_indices))
xray_failed = len(set(xray_indices) & set(failed_indices))
mri_failed = len(set(mri_indices) & set(failed_indices))

ct_acc = 1 - ct_failed / len(ct_indices) if ct_indices else 0
xray_acc = 1 - xray_failed / len(xray_indices) if xray_indices else 0
mri_acc = 1 - mri_failed / len(mri_indices) if mri_indices else 0

print(f"CT Accuracy: {ct_acc:.4f}")
print(f"X-Ray Accuracy: {xray_acc:.4f}")
print(f"MRI Accuracy: {mri_acc:.4f}")

print(f"Overall Accuracy: {(ct_acc + xray_acc + mri_acc) / 3:.4f}")