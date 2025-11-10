import torch
from torch.nn.functional import cosine_similarity
from torchvision.transforms.functional import resize, InterpolationMode, to_tensor
from PIL import Image
from pyiqa import create_metric

from transface.backbones import get_model

from glob import glob

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

psnr_list = []
ssim_list = []
lpips_list = []
niqe_list = []
ids_list = []

NAME = "kface/m2f"

sr_paths = sorted(glob("results/%s/sr/*.png" % NAME))
hr_paths = sorted(glob("results/%s/hr/*.png" % NAME))

for i in range(len(sr_paths)):
    print("🔥 Processing %s of %s" % (i + 1, len(sr_paths)))
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

fid_score = get_fid("results/%s/sr/" % NAME, "results/%s/hr/" % NAME)
eval_result["fid"] = "%.2f" % fid_score

with open("./results/%s/eval.txt" % NAME, "w") as f:
    f.write("\n".join(["%s=%s" % (k, v) for k, v in eval_result.items()]))

print("🍊", "\n".join(["%s=%s" % (k, v) for k, v in eval_result.items()]))
