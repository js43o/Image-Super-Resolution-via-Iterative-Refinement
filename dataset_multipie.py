from PIL import Image
import os

RECORDING_COND = ["01"]
LIGHT_COND = ["%02d" % i for i in range(20)]

ANGLES_EXTREME = ["11_0", "12_0", "09_0", "19_1", "08_1", "20_0", "01_0", "24_0"]
ANGLES_MODERATE = ["08_0", "13_0", "14_0", "05_0", "04_1", "19_0"]

GT_ANGLES_MODERATE = ["08_0", "19_0"]
GT_ANGLES_FRONTAL = ["05_1", "05_1"]

dataroot = "../../datasets/multi-pie_crop_patch_v2/test"

MODEL_TYPE = "m2f"  # 포즈 지정

if MODEL_TYPE == "m2f":
    angles = ANGLES_MODERATE
    gt_angles = GT_ANGLES_FRONTAL
elif MODEL_TYPE == "e2f":
    angles = ANGLES_EXTREME
    gt_angles = GT_ANGLES_FRONTAL
else:
    raise "pose error"

new_dataroot_lr = "./dataset/multipie/%s/test/lr_32" % MODEL_TYPE
new_dataroot_sr = "./dataset/multipie/%s/test/sr_32_128" % MODEL_TYPE
new_dataroot_hr = "./dataset/multipie/%s/test/hr_128" % MODEL_TYPE

os.makedirs(new_dataroot_lr, exist_ok=True)
os.makedirs(new_dataroot_sr, exist_ok=True)
os.makedirs(new_dataroot_hr, exist_ok=True)

pids = sorted(os.listdir(dataroot))

for pid in pids:
    print("processing id %s..." % pid)
    for recording in RECORDING_COND:
        for idx, angle in enumerate(angles):
            for light in LIGHT_COND:
                sr_path = os.path.join(
                    dataroot,
                    pid,
                    recording,
                    angle,
                    "%s.png" % light,
                )
                gt_angle = gt_angles[0] if idx < len(angles) // 2 else gt_angles[1]
                gt_path = os.path.join(
                    dataroot,
                    pid,
                    recording,
                    gt_angle,
                    "%s.png" % light,
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
                            "%s_%s_%s_%s.png" % (pid, recording, angle, light),
                        )
                    )
                    sr.save(
                        os.path.join(
                            new_dataroot_sr,
                            "%s_%s_%s_%s.png" % (pid, recording, angle, light),
                        )
                    )
                    hr.save(
                        os.path.join(
                            new_dataroot_hr,
                            "%s_%s_%s_%s.png" % (pid, recording, angle, light),
                        )  # the same name as input image, only different folder
                    )
