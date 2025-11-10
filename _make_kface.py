from PIL import Image
import os

EXPRESSION_COND = ["E01"]
LIGHT_COND = ["L1", "L2", "L3", "L4", "L8", "L9", "L10", "L13"]

ANGLES_EXTREME = [1, 2, 3, 14, 19, 20, 18, 11, 12, 13]
ANGLES_MODERATE = [4, 5, 6, 15, 16, 17, 8, 9, 10]

GT_ANGLES_MODERATE = [4, 10]
GT_ANGLES_FRONTAL = [7, 7]

dataroot = "../../datasets/kface_crop_patch_v2/train"
angles = list(range(1, 21))
gt_angles = GT_ANGLES_FRONTAL

new_dataroot_lr = "./dataset/kface/train/lr_32"
new_dataroot_sr = "./dataset/kface/train/sr_32_128"
new_dataroot_hr = "./dataset/kface/train/hr_128"

os.makedirs(new_dataroot_lr, exist_ok=True)
os.makedirs(new_dataroot_sr, exist_ok=True)
os.makedirs(new_dataroot_hr, exist_ok=True)

pids = sorted(os.listdir(dataroot))

for idx, pid in enumerate(pids):
    print("🔥 Processing #ID %s/%s ..." % (idx + 1, len(pids)))
    for light in LIGHT_COND:
        for expression in EXPRESSION_COND:
            for idx, angle in enumerate(angles):
                img_path = os.path.join(
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

                if os.path.exists(img_path) and os.path.exists(gt_path):
                    lr = (
                        Image.open(img_path)
                        .convert("RGB")
                        .resize((32, 32), Image.Resampling.BICUBIC)  # degrade to 32x32
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
                            "%s_%s_%s.png" % (pid, angle, light),
                        )
                    )
                    sr.save(
                        os.path.join(
                            new_dataroot_sr,
                            "%s_%s_%s.png" % (pid, angle, light),
                        )
                    )
                    hr.save(
                        os.path.join(
                            new_dataroot_hr,
                            "%s_%s_%s.png" % (pid, angle, light),
                        )  # the same name as input image, only different folder
                    )
