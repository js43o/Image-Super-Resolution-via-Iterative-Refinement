from PIL import Image
import os

"""
with open("multi-pie_sep_list.txt", "w") as f:
    filenames = os.listdir("/4tb/datasets/multi-pie_sep_v2/test/gt")
    for filename in filenames:
        gt_path = os.path.join("/4tb/datasets/multi-pie_sep_v2/test/gt", filename)
        image_path = os.path.join(
            "/4tb/datasets/multi-pie_sep_v2/test/input", filename
        )
        if os.path.exists(gt_path) and os.path.exists(image_path):
            f.write("%s, %s\n" % (gt_path, image_path))
"""

RECORDING_COND = ["01"]
LIGHT_COND = ["%02d" % i for i in range(20)]

ANGLES_EXTREME = ["11_0", "12_0", "09_0", "19_1", "08_1", "20_0", "01_0", "24_0"]
ANGLES_MODERATE = ["08_0", "13_0", "14_0", "05_0", "04_1", "19_0"]

GT_ANGLES_MODERATE = ["08_0", "19_0"]
GT_ANGLES_FRONTAL = ["05_1", "05_1"]

dataroot = "../../datasets/multi-pie_crop_v2/test"
angles = [*ANGLES_EXTREME, *ANGLES_MODERATE]
gt_angles = GT_ANGLES_FRONTAL

new_dataroot_lr = "../..//datasets/multi-pie_sr_v2/test/lr_16"
new_dataroot_sr = "../..//datasets/multi-pie_sr_v2/test/sr_16_128"
new_dataroot_hr = "../..//datasets/multi-pie_sr_v2/test/hr_128"

os.makedirs(new_dataroot_lr, exist_ok=True)
os.makedirs(new_dataroot_sr, exist_ok=True)
os.makedirs(new_dataroot_hr, exist_ok=True)

ids = sorted(os.listdir(dataroot))

"""
with open("multi-pie_sep_list.txt", "w") as f:
    for id in ids:
        print("processing id %s..." % id)
        for recording in RECORDING_COND:
            for idx, angle in enumerate(angles):
                for light in LIGHT_COND:
                    gt_path = os.path.join(
                        new_dataroot_hr,
                        "%s_%s_%s_%s.png" % (id, recording, angle, light),
                    )
                    image_path = os.path.join(
                        new_dataroot_input,
                        "%s_%s_%s_%s.png" % (id, recording, angle, light),
                    )
                    f.write("%s, %s\n" % (gt_path, image_path))
"""

for id in ids:
    print("processing id %s..." % id)
    for recording in RECORDING_COND:
        for idx, angle in enumerate(angles):
            for light in LIGHT_COND:
                image_path = os.path.join(
                    dataroot,
                    id,
                    recording,
                    angle,
                    "%s.png" % light,
                )
                gt_angle = gt_angles[0] if idx < len(angles) // 2 else gt_angles[1]
                gt_path = os.path.join(
                    dataroot,
                    id,
                    recording,
                    gt_angle,
                    "%s.png" % light,
                )
                if os.path.exists(image_path) and os.path.exists(gt_path):
                    lr = (
                        Image.open(image_path)
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
                            "%s_%s_%s_%s.png" % (id, recording, angle, light),
                        )
                    )
                    sr.save(
                        os.path.join(
                            new_dataroot_sr,
                            "%s_%s_%s_%s.png" % (id, recording, angle, light),
                        )
                    )
                    hr.save(
                        os.path.join(
                            new_dataroot_hr,
                            "%s_%s_%s_%s.png" % (id, recording, angle, light),
                        )  # the same name as input image, only different folder
                    )
