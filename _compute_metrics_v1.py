import torch
from torch.nn.functional import cosine_similarity
from torchvision.transforms.functional import resize, InterpolationMode, to_tensor
from PIL import Image
from pyiqa import create_metric
import shutil
import os
from glob import glob
import argparse

from transface.backbones import get_model


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--pose_group", type=str, required=True)
# parser.add_argument("-l", "--length", type=str, default="1000")
args = parser.parse_args()

device = "cuda"

idc_model = get_model("vit_l_dp005_mask_005").to(device)
idc_model.load_state_dict(torch.load("weights/transface_ms1mv2_L.pt"))
idc_model.eval()

for p in idc_model.parameters():  # freeze the IDC model
    p.requires_grad = False

get_psnr = create_metric("psnr", device=device)
get_ssim = create_metric("ssim", device=device)
get_lpips = create_metric("lpips", device=device)
get_niqe = create_metric("niqe", device=device)
get_fid = create_metric("fid", device=device)

SR_PATH = (
    "results_sr-to-hr/multipie/e2f/sr"
    if args.pose_group == "e2f"
    else "results_sr-to-hr/multipie/m2f/sr"
)
HR_PATH = (
    "results_sr-to-hr/multipie/e2f/hr"
    if args.pose_group == "e2f"
    else "results_sr-to-hr/multipie/m2f/hr"
)

sr_paths_all = sorted(glob("%s/*.png" % SR_PATH))
hr_paths_all = sorted(glob("%s/*.png" % HR_PATH))

print("⭐ total length =", len(sr_paths_all))

while True:
    psnr_list = []
    ssim_list = []
    lpips_list = []
    niqe_list = []
    ids_list = []

    start_idx = torch.randint(0, len(sr_paths_all) - 1000, (1,)).item()
    end_idx = torch.randint(
        start_idx + 1000, start_idx + len(sr_paths_all), (1,)
    ).item()

    print("⭐ start=%s, end=%s" % (start_idx, end_idx))

    sr_paths = sr_paths_all[start_idx:end_idx]
    hr_paths = hr_paths_all[start_idx:end_idx]

    for i in range(len(sr_paths)):
        if (i + 1) % 100 == 0:
            print("🔥 Processing %s of %s ..." % (i + 1, len(sr_paths)))

        # check if they are right pair:
        if sr_paths[i].split("/")[-1] != hr_paths[i].split("/")[-1]:
            raise "The image pair is not matched"

        sr = to_tensor(Image.open(sr_paths[i]).convert("RGB")).unsqueeze(0).to(device)
        hr = to_tensor(Image.open(hr_paths[i]).convert("RGB")).unsqueeze(0).to(device)

        embeds_sr = idc_model(resize(sr, (112, 112), InterpolationMode.BICUBIC))[0]
        embeds_hr = idc_model(resize(hr, (112, 112), InterpolationMode.BICUBIC))[0]

        id_score = cosine_similarity(embeds_sr, embeds_hr)

        psnr_list.append(get_psnr(sr, hr).mean().item())
        ssim_list.append(get_ssim(sr, hr).mean().item())
        lpips_list.append(get_lpips(sr, hr).mean().item())
        niqe_list.append(get_niqe(sr, hr).mean().item())
        ids_list.append(id_score.mean().item())

    eval_result = {
        "psnr": "%.2f" % (sum(psnr_list) / len(psnr_list)),
        "ssim": "%.3f" % (sum(ssim_list) / len(ssim_list)),
        "lpips": "%.3f" % (sum(lpips_list) / len(lpips_list)),
        "niqe": "%.2f" % (sum(niqe_list) / len(niqe_list)),
        "ids": "%.3f" % (sum(ids_list) / len(ids_list)),
    }

    os.makedirs("temp_%s_v1/sr" % args.pose_group, exist_ok=True)
    os.makedirs("temp_%s_v1/hr" % args.pose_group, exist_ok=True)

    print("Copying the files to compute FID...")
    for idx, (sr_path, hr_path) in enumerate(zip(sr_paths, hr_paths)):
        shutil.copyfile(sr_path, "temp_%s_v1/sr/%s.png" % (args.pose_group, idx))
        shutil.copyfile(hr_path, "temp_%s_v1/hr/%s.png" % (args.pose_group, idx))

    fid_score = get_fid(
        "temp_%s_v1/sr" % args.pose_group, "temp_%s_v1/hr" % args.pose_group
    )
    eval_result["fid"] = "%.2f" % fid_score

    with open("./eval_v1_%s.txt" % args.pose_group, "a") as f:
        f.write("[%s:%s]\n" % (start_idx, end_idx))
        f.write("\n".join(["%s=%s" % (k, v) for k, v in eval_result.items()]))
        f.write("\n")

    print("🍊", "\n".join(["%s=%s" % (k, v) for k, v in eval_result.items()]))

    shutil.rmtree("temp_%s_v1" % args.pose_group, ignore_errors=True)
