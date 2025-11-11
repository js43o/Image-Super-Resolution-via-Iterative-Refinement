import os
import torch
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor, resize, InterpolationMode
from torchvision.utils import save_image
from torchvision.models import resnet50
from safetensors.torch import load_file
from pyiqa import create_metric
from tqdm.auto import tqdm
import argparse
import subprocess
import shutil
import gc
import re

from dataset_kface import KfaceDataset
from dataset_multipie import MultiPIEDataset, MultiPIEDatasetWithSingleView
from models.cr.model import CoarseRestorer, CoarseRestorerWithDoubleEncoder
from models.idc.transface.backbones import get_model
from utils import seed_everything

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument(
    "--name",
    type=str,
    required=True,
    help="A name of the trained CR module to decide checkpoint and output directory",
)
parser.add_argument(
    "--epoch",
    type=str,
    default="47",
    help="A epoch of the trained CR module",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=5e-4,
    help="An initial learning rate for the optimizer",
)
parser.add_argument(
    "--batch_size", type=int, default=256, help="A batch size of testing"
)
parser.add_argument(
    "--dataset",
    type=str,
    default="multipie",
    help='A name of the training dataset ("kface" | "multipie")',
)
parser.add_argument(
    "--image_res",
    type=int,
    default=128,
    help="A resolution of the images",
)
parser.add_argument(
    "--model_type",
    type=str,
    required=True,
    help='A type of model and dataset for testing ("uni" | "uni_e" | "uni_m" | "e2m" | "m2f" | "e2f")',
)
parser.add_argument(
    "--angle",
    type=str,
    help="ANGLE",
)
parser.add_argument(
    "--fuse_type",
    type=str,
    default="concat",
    help='How to fuse inputs of multi-reference CR module ("concat" | "addition") (ONLY WORKING WITH e2f MODEL)',
)
parser.add_argument(
    "--ckpt_e2m",
    type=str,
    help="A checkpoint path of extreme-to-moderate model (ONLY WORKING WITH e2f MODEL)",
)
parser.add_argument(
    "--save_image_batch",
    type=int,
    default=15,
    help="A number of batch to save sample images during the test",
)
parser.add_argument(
    "--use_fid",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Whether to include FID calculation",
)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
seed_everything(42)
os.makedirs("./output/cr/%s/test" % args.name, exist_ok=True)


DEFAULT_POS_PROMPT = "high-quality portrait, sharp and detailed face, realistic skin texture, natural lighting, symmetrical face, well-defined facial features, clear eyes, smooth skin, photorealistic, front-facing, professional photography."
DEFAULT_NEG_PROMPT = "blurry, low resolution, deformed face, distorted features, extra eyes, extra nose, asymmetry, unnatural skin, over-smoothing, cartoon, painting, text, watermark, jpeg artifacts, unreal, uncanny, bad anatomy."


if args.use_fid:
    shutil.rmtree("./output/cr/%s/temp" % args.name, ignore_errors=True)
    os.makedirs("./output/cr/%s/temp/real" % args.name, exist_ok=True)
    os.makedirs("./output/cr/%s/temp/fake" % args.name, exist_ok=True)


def start_test(dataloader, model, e2m_model, idc_model):
    progress_bar = tqdm(total=len(dataloader))
    progress_bar.set_description(f"Testing")

    get_psnr = create_metric("psnr", device=device)
    get_ssim = create_metric("ssim", device=device)
    get_lpips = create_metric("lpips", device=device)
    get_niqe = create_metric("niqe", device=device)

    psnr_list = []
    ssim_list = []
    lpips_list = []
    niqe_list = []
    ids_list = []

    model.eval()
    e2m_model.eval()
    idc_model.eval()

    with torch.no_grad():
        for batch, (x, y, _y_patch, _angle) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            if args.model_type == "e2f":
                prev_cr = e2m_model(x)
                pred = model(prev_cr, x)
            else:
                pred = model(x)

            embeds_pred = idc_model(resize(pred, 112, InterpolationMode.BICUBIC))[0]
            embeds_y = idc_model(resize(y, 112, InterpolationMode.BICUBIC))[0]

            ids = cosine_similarity(embeds_pred, embeds_y)

            psnr_list.append(get_psnr(pred, y).mean().item())
            ssim_list.append(get_ssim(pred, y).mean().item())
            lpips_list.append(get_lpips(pred, y).mean().item())
            niqe_list.append(get_niqe(pred, y).mean().item())
            ids_list.append(ids.mean().item())

            progress_bar.update(1)
            logs = {
                "psnr": psnr_list[-1],
                "ssim": ssim_list[-1],
                "lpips": lpips_list[-1],
                "niqe": niqe_list[-1],
                "ids": ids_list[-1],
                "batch": batch,
            }
            progress_bar.set_postfix(**logs)

            if batch % args.save_image_batch == 0:
                if args.model_type == "e2f" and prev_cr is not None:
                    result = torch.cat([x[0], prev_cr[0], pred[0], y[0]], dim=-1)
                else:
                    result = torch.cat([x[0], pred[0], y[0]], dim=-1)

                save_image(
                    result,
                    os.path.join(
                        "output/cr/%s/test/%s_%s.png"
                        % (args.name, args.epoch, batch + 1),
                    ),
                )

            if args.use_fid:
                for idx in range(x.shape[0]):
                    save_image(
                        y[idx],
                        os.path.join(
                            "output/cr/%s/temp/real/%s.png"
                            % (args.name, batch * args.batch_size + idx),
                        ),
                    )
                    save_image(
                        pred[idx],
                        os.path.join(
                            "output/cr/%s/temp/fake/%s.png"
                            % (args.name, batch * args.batch_size + idx),
                        ),
                    )

    eval_result = {
        "psnr": "%.2f" % (sum(psnr_list) / len(psnr_list)),
        "ssim": "%.3f" % (sum(ssim_list) / len(ssim_list)),
        "lpips": "%.3f" % (sum(lpips_list) / len(lpips_list)),
        "niqe": "%.2f" % (sum(niqe_list) / len(niqe_list)),
        "ids": "%.3f" % (sum(ids_list) / len(ids_list)),
    }

    if args.use_fid:
        result = subprocess.check_output(
            "python -m pytorch_fid output/cr/%s/temp/real output/cr/%s/temp/fake --device cuda"
            % (args.name, args.name),
            stderr=subprocess.STDOUT,
            text=True,
            shell=True,
        )
        p = re.compile(r"(?<=FID:).+")
        fid_score = float(p.search(result)[0])
        eval_result["fid"] = "%.2f" % fid_score

    with open("./output/cr/%s/test/eval.txt" % args.name, "w") as f:
        f.write("\n".join(["%s=%s" % (k, v) for k, v in eval_result.items()]))

    print("🍊", eval_result)

    gc.collect()
    torch.cuda.empty_cache()


if args.model_type == "uni_e":
    data_type = "e2f"
elif args.model_type == "uni_m":
    data_type = "m2f"
else:
    data_type = args.model_type

if args.angle is not None:  # for single view
    test_dataset = MultiPIEDatasetWithSingleView(
        dataroot="../../datasets/multi-pie_crop_patch_v2",
        angle=args.angle,
        use="test",
        res=args.image_res,
    )
elif args.dataset == "kface":
    test_dataset = KfaceDataset(
        dataroot="../../datasets/kface_crop_patch_v2",
        use="test",
        res=args.image_res,
        model_type=data_type,
    )
elif args.dataset == "multipie":
    test_dataset = MultiPIEDataset(
        dataroot="../../datasets/multi-pie_crop_patch_v2",
        use="test",
        res=args.image_res,
        model_type=data_type,
    )

test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size)

e2m_model = CoarseRestorer(res=args.image_res, in_channels=3).to(device)
if args.model_type == "e2f" and args.ckpt_e2m is not None:
    # multi-reference model
    model = CoarseRestorerWithDoubleEncoder(
        res=args.image_res, fuse_type=args.fuse_type
    ).to(device)
    e2m_model.load_state_dict(load_file(args.ckpt_e2m))
else:
    model = CoarseRestorer(res=args.image_res).to(device)

idc_model = get_model("vit_l_dp005_mask_005").to(device)
idc_model.load_state_dict(torch.load("checkpoints/idc/transface_ms1mv2_L.pt"))

"""
idc_model = resnet50(num_classes=512).to(device)
idc_model.load_state_dict(load_file("checkpoints/idc/arcface_wo_deg.safetensors"))
"""

ckpt_path = "checkpoints/cr/%s/%s/model.safetensors" % (args.name, args.epoch)
model.load_state_dict(load_file(ckpt_path))


start_test(
    dataloader=test_dataloader, model=model, e2m_model=e2m_model, idc_model=idc_model
)
print("✅ Done!")
