from run_utils import Config
from src.model.tramnerf.helper import CylinderMirror, Mirror
from src.model.tramnerf.model import TraMNeRF
from src.model.tramnerf_rough.model import TraMNeRFRough
import src.data.dataset as data
import torch
import numpy as np
import tqdm
import gin
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

from run_utils import render_full_image

torch.set_float32_matmul_precision('high')        
        
def main():
    parser = argparse.ArgumentParser(
        description='Generates rendering of a trained TraM-NeRF model.'
    )

    parser.add_argument(
        '--run',
        type=Path,
        help="Path to the log directory."
    )
    parser.add_argument(
        '--checkpoint',
        default=None,
        help="Path to the checkpoint file to use (optional)."
    )
    parser.add_argument(
        '--skip_checkpoint_check',
        action='store_true',
        help=(
            "If this flag is set, the script will silently choose the newest checkpoint if "
            "none is given, instead of asking for confirmation."
        )
    )
    parser.add_argument(
        "--variance",
        type=float,
        default=None,
        help=(
            "If given, will override the roughness value of the material given in the "
            "config file."
        )
    )
    parser.add_argument(
        "--num_ray_samples",
        default=None,
        type=int,
        help="If given, will override the number of rays given in the config file."
    )
    parser.add_argument(
        "--output_suffix",
        default="",
        help="A suffix to be added to the output path (optional)."
    )
    parser.add_argument(
        "--chunk_size",
        default=2**14,
        type=int,
        help="Number of rays per inference batch. Controls the GPU memory usage."
    )
    parser.add_argument(
        "--image_names",
        default=None,
        help=(
            "A comma-separated list of image names to be rendered (optional). If "
            "not given, all images will be rendered."
        )
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "test"],
        help="Selects which part of the dataset should be rendered."
    )
    args = parser.parse_args()

    run_dir = args.run
    config_name = run_dir.name.rsplit('-', maxsplit=1)[0]
    log_name = run_dir.name

    gin.parse_config_file(args.run / f"{config_name}.gin")

    config = Config()

    torch.manual_seed(config.seed)
    device = ("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.checkpoint is None:
        print("No checkpoint given, looking for newest...")
        best_checkpoint_it = 0
        for checkpoint_candidate_path in run_dir.glob("*.pt"):
            try:
                checkpoint_it = int(checkpoint_candidate_path.stem)
                best_checkpoint_it = max(best_checkpoint_it, checkpoint_it)
            except ValueError:
                pass
        args.checkpoint = f"{best_checkpoint_it}.pt"
        if not args.skip_checkpoint_check and input(f"Found checkpoint '{args.checkpoint}', continue? (y/n) ") != "y":
            print("aborting.")
            exit(0)


    test_dataset = data.SubjectLoader(
        subject_id=config.scene_name,
        root_fp="/data/holland/Masterarbeit/data",
        split=args.split,
        num_rays=None,
        scale=config.scale,
        center=config.center
    )
    test_dataset.images = test_dataset.images.to(device)
    test_dataset.camtoworlds = test_dataset.camtoworlds.to(device)
    test_dataset.K = test_dataset.K.to(device)

    radiance_field = torch.load(f"{args.run}/{args.checkpoint}")

    for mirror in radiance_field.mirrors:
        try:
            mirror.set_dataset(test_dataset)
        except:
            pass

    if args.variance is not None:
        print("Changing variance from", radiance_field.variance, "to", args.variance)
        radiance_field.variance = args.variance

    if args.num_ray_samples is not None:
        print("Changing number of ray samples from", radiance_field.num_ray_samples, "to", args.num_ray_samples)
        radiance_field.num_ray_samples = args.num_ray_samples

    output_dir = Path("results") / config.scene_name / config.model / (log_name + args.output_suffix)
    num_images = len(test_dataset)

    output_dir.mkdir(exist_ok=True, parents=True)
    print("Saving to", output_dir.absolute())

    if args.image_names is not None:
        allowed_image_names = args.image_names.split(",")
        print("Only rendering:", allowed_image_names)
    else:
        allowed_image_names = None

    with torch.no_grad():
        with tqdm.tqdm(total=num_images) as pbar:
            for i in range(num_images):
                image_name = Path(test_dataset.image_paths[i]).stem

                if allowed_image_names is not None and image_name not in allowed_image_names:
                    print("skipping", image_name)
                    continue

                rendered = render_full_image(
                    model=radiance_field,
                    dataset_sample=test_dataset[i],
                    chunk_size=args.chunk_size,
                    return_debug_images=False,
                    loss_data_fn=lambda x: dict(),
                )

                image = (rendered['fine_rgb'].cpu().numpy() * 255).astype(np.uint8)
                try:
                    plt.imsave(str(output_dir / (image_name + ".png")), image)
                except Exception as e:
                    print(e)

                pbar.update(1)


if __name__ == "__main__":
    main()