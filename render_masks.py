from run_utils import Config
from src.model.tramnerf.helper import CylinderMirror, Mirror
import src.data.dataset as data
import torch
import imageio
import tqdm
import gin
import argparse
from pathlib import Path

# import for gin
from src.model.tramnerf.model import TraMNeRF
from src.model.tramnerf_rough.model import TraMNeRFRough

from run_utils import render_full_image

torch.set_float32_matmul_precision('high')


def get_legacy_cylinder(mirrors: list[Mirror]):
    for m in mirrors:
        if isinstance(m, CylinderMirror) and not hasattr(m, "open_cylinder"):
            return m
    return None

def import_legacy_cylinder(mirrors: list[Mirror]):
    # check if there is a legacy mirror
    legacy_mirror = get_legacy_cylinder(mirrors)

    if legacy_mirror is not None and len(mirrors) == 3:
        return [CylinderMirror(
            origin=legacy_mirror.origin,
            end=legacy_mirror.end,
            radius=legacy_mirror.radius
        )]
    else:
        return mirrors
        
        
def main():
    parser = argparse.ArgumentParser(
        description='Renders masks of the mirrors used in the given log checkpoint.'
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
    test_dataset.training = False

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
        if args.skip_checkpoint_check or input(f"Found checkpoint '{args.checkpoint}', continue? (y/n) ") != "y":
            print("aborting.")
            exit(0)

    radiance_field = torch.load(f"{args.run}/{args.checkpoint}")
    radiance_field.mirrors = import_legacy_cylinder(radiance_field.mirrors)
    radiance_field.num_samples = 2
    radiance_field.num_levels = 1

    for mirror in radiance_field.mirrors:
        try:
            mirror.set_dataset(test_dataset)
        except:
            pass

    output_dir = Path("results_masks") / config.scene_name / config.model / log_name
    num_images = len(test_dataset)

    output_dir.mkdir(exist_ok=True, parents=True)
    print("Saving to", output_dir.absolute())

    with torch.no_grad():
        with tqdm.tqdm(total=num_images) as pbar:
            for i in range(num_images):
                image_name = Path(test_dataset.image_paths[i]).stem

                rendered = render_full_image(
                    model=radiance_field,
                    dataset_sample=test_dataset[i],
                    loss_data_fn=lambda x: dict(),
                    render_keys=('coarse_ray_num_bounces',),
                    chunk_size=2**14,
                    return_debug_images=True,
                    return_loss_outputs=False
                )

                mask_image = (((rendered['coarse_ray_num_bounces'] > 0).to(torch.uint8) * 255).cpu().numpy())
                try:
                    imageio.imwrite(str(output_dir / (image_name + ".png")), mask_image)
                except Exception as e:
                    print(e)

                pbar.update(1)


if __name__ == "__main__":
    main()