import argparse
from pathlib import Path
import torch 
import imageio.v2 as imageio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")

def weighted_mean(data: torch.Tensor, weights: torch.Tensor):
    return (data * weights).sum() / weights.sum()

def weighted_std(data: torch.Tensor, weights: torch.Tensor):
    mean = weighted_mean(data, weights)
    data = torch.where(weights > 0, data, 0)
    num = (weights * (data - mean[None]) ** 2).sum()
    denom = (len(data) - 1) / len(data) * weights.sum()
    return torch.sqrt(num / denom)

def is_image_suffix(suffix):
    return suffix.lower() in IMAGE_SUFFIXES

def r_sorter(path: Path):
    if path.name.startswith("r_"):
        return int(path.stem[2:])
    else:
        return path
    
def find_unique_file(path: Path, stem: str):
    candidates = list(path.glob(f"{stem}.*"))
    if len(candidates) != 1:
        print(f"Warning: Was looking for unique '{stem}' equivalent in '{path}', but found {candidates}.")
        return None
    return candidates[0]

def psnr(image, gt):
        return -10.0 * torch.log10(torch.mean((image - gt) ** 2))

def rescaled(image):
    return torch.unsqueeze(torch.permute(image, (2, 0, 1)), 0)

def wrap_rescaled(fn):
    def _wrapped(image, gt):
        return fn(rescaled(image), rescaled(gt))
    return _wrapped

def run_evaluation(gt_dir_path: Path, images_dir_path: Path, masks_dir_path: Path, match: str, debug: bool):
    device = ("cuda:0" if torch.cuda.is_available() else "cpu")
    generate_debug_plot = True
    debug_worst_psnr = 1000
    
    metric_fns = {
        "psnr": psnr,

        "ssim": wrap_rescaled(StructuralSimilarityIndexMeasure(
            data_range=(0.0, 1.0),
            reduction='elementwise_mean'
        ).to(device)),

        "lpips": wrap_rescaled(LearnedPerceptualImagePatchSimilarity(
            net_type='vgg',
            reduction='mean',
            normalize=True
        ).to(device))
    }

    results = defaultdict(list)
    weights = []

    image_paths = sorted((p for p in images_dir_path.glob("*") if is_image_suffix(p.suffix)), key=r_sorter)

    # for each image in images dir
    for image_idx, image_path in enumerate(tqdm(image_paths)):

        # search for image in gt dir

        if match == "name":
            gt_path = find_unique_file(gt_dir_path, image_path.stem)
            if gt_path is None:
                continue
        elif match == "lexicographic":
            gt_path = sorted((p for p in gt_dir_path.glob(f"*") if is_image_suffix(p.suffix)), key=r_sorter)[image_idx]

        # read gt image

        gt_image = imageio.imread(str(gt_path))[..., :3]
        assert gt_image.max() > 2
        gt_image = torch.from_numpy(gt_image / 255).to(torch.float32).to(device)

        # read rendered image
        
        rendered_image = imageio.imread(str(image_path))[..., :3]
        assert rendered_image.max() > 2
        rendered_image = torch.from_numpy(rendered_image / 255).to(torch.float32).to(device)

        if masks_dir_path is not None:
            
            # look for mask
            
            mask_path = find_unique_file(masks_dir_path, gt_path.stem)
            mask_image = imageio.imread(str(mask_path)) > 0
            mask_image = torch.from_numpy(mask_image).to(device)
            weight = mask_image.count_nonzero() / rendered_image.shape[0] * rendered_image.shape[1]

            gt_image[~mask_image] = 0
            rendered_image[~mask_image] = 0
        else:
            weight = torch.tensor(1.0, device=device)

        # evaluate metrics on whole image
            
        weights.append(weight)
    
        for metric_name, metric_fn in metric_fns.items():
            value = metric_fn(rendered_image, gt_image).nan_to_num(0)
            results[metric_name].append(value)
        
        if debug and generate_debug_plot and results["psnr"][-1] < debug_worst_psnr:
            debug_worst_psnr = results["psnr"][-1]
            _, axs = plt.subplots(1, 2, figsize=(12, 6))
            axs[0].imshow(gt_image.cpu())
            axs[1].imshow(rendered_image.cpu())
            metrics_str = ', '.join(f"{n}: {v[-1]:.3f}" for n, v in results.items())
            plt.suptitle(metrics_str)
            plt.savefig("eval_debug.png")
            print("new worst PSNR when matching", gt_path, "with", image_path)

    return {k: torch.stack(v).cpu() for k, v in results.items()}, torch.tensor(weights)

@torch.no_grad()
def main():

    parser = argparse.ArgumentParser(
        description="Evaluates given set of images against ground-truth."
    )

    parser.add_argument(
        "--gt",
        type=Path,
        help="Path to the ground-truth images."
    )
    parser.add_argument(
        "--images",
        type=Path,
        help="Path to the renderings."
    )
    parser.add_argument(
        "--masks",
        type=Path,
        default=None,
        help="Path to masks that indicate mirror regions (optional)."
    )
    parser.add_argument(
        "--match",
        choices=["name", "lexicographic"],
        default="name",
        help=(
            "Matching method for names. 'name' will look for exact names in both "
            "directories (recommended), whereas 'lexicographic' will sort files in "
            "both folders lexicographically and match them by list index."
        )

    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode."
    )
    args = parser.parse_args()

    results, weights = run_evaluation(
        gt_dir_path=args.gt,
        images_dir_path=args.images,
        masks_dir_path=args.masks,
        match=args.match,
        debug=args.debug
    )

    # return statistics
            
    for metric_name, values in results.items():
        print(metric_name, "mean", weighted_mean(values, weights).item(), "std", weighted_std(values, weights).item())


if __name__ == "__main__":
    main()

    