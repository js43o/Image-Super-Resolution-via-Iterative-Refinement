import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from itertools import combinations

EXPRESSION_COND = ["E01"]
LIGHT_COND = ["L1", "L2", "L3", "L4", "L8", "L9", "L10", "L13"]

ANGLES_EXTREME = [1, 2, 3, 14, 19, 20, 18, 11, 12, 13]
ANGLES_MODERATE = [4, 5, 6, 15, 16, 17, 8, 9, 10]

GT_ANGLES_MODERATE = [4, 10]
GT_ANGLES_FRONTAL = [7, 7]

dataroot = "../../datasets/kface_crop_patch_v2/test"

MODEL_TYPE = "e2f"  # 포즈 지정

if MODEL_TYPE == "m2f":
    angles = ANGLES_MODERATE
    gt_angles = GT_ANGLES_FRONTAL
elif MODEL_TYPE == "e2f":
    angles = ANGLES_EXTREME
    gt_angles = GT_ANGLES_FRONTAL
else:
    raise "pose error"

new_dataroot_lr = "./dataset/kface/%s/test/lr_32" % MODEL_TYPE
new_dataroot_sr = "./dataset/kface/%s/test/sr_32_128" % MODEL_TYPE
new_dataroot_hr = "./dataset/kface/%s/test/hr_128" % MODEL_TYPE

os.makedirs(new_dataroot_lr, exist_ok=True)
os.makedirs(new_dataroot_sr, exist_ok=True)
os.makedirs(new_dataroot_hr, exist_ok=True)

pids = sorted(os.listdir(dataroot))


pids = sorted(os.listdir(dataroot))

for pid in pids:
    print("processing id %s..." % pid)
    for light in LIGHT_COND:
        for expression in EXPRESSION_COND:
            for idx, angle in enumerate(angles):
                sr_path = os.path.join(
                    dataroot,
                    pid,
                    "S001",
                    light,
                    expression,
                    "C%s.png" % angle,
                )
                gt_angle = gt_angles[0] if idx < len(angles) // 2 else gt_angles[1]
                gt_path = os.path.join(
                    dataroot,
                    pid,
                    "S001",
                    light,
                    expression,
                    "C%s.png" % gt_angle,
                )
                if os.path.exists(sr_path) and os.path.exists(gt_path):
                    lr = (
                        Image.open(sr_path)
                        .convert("RGB")
                        .resize((32, 32), Image.Resampling.BICUBIC)
                    )
                    sr = lr.resize((128, 128), Image.Resampling.BICUBIC)
                    hr = (
                        Image.open(gt_path)
                        .convert("RGB")
                        .resize((128, 128), Image.Resampling.BICUBIC)
                    )

                    lr.save(
                        os.path.join(
                            new_dataroot_lr,
                            "%s_%s_%s.png" % (pid, light, angle),
                        )
                    )
                    sr.save(
                        os.path.join(
                            new_dataroot_sr,
                            "%s_%s_%s.png" % (pid, light, angle),
                        )
                    )
                    hr.save(
                        os.path.join(
                            new_dataroot_hr,
                            "%s_%s_%s.png" % (pid, light, angle),
                        )  # the same name as input image, only different folder
                    )
