from functools import partial
from math import sqrt
from time import time
from run_utils import Config, ScheduledOptimizer, TrainProgressBar, render_full_image, setup_writers
from src.data.utils import Rays
import src.data.dataset as data
from src.model.helper import img2mse, mse2psnr
from src.model.tramnerf.losses import LOSS_FUNCTIONS
from src.model.tramnerf.model import TraMNeRF
from src.model.tramnerf_rough.model import TraMNeRFRough
import torch
import gin
import argparse
import shutil
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import torch.nn.utils

# torch.autograd.set_detect_anomaly(True)
torch.set_float32_matmul_precision('high')

import datetime


def main():
    parser = argparse.ArgumentParser(
        description="Train script for TraM-NeRF."
    )

    parser.add_argument(
        '-config',
        required=False,
        type=Path,
        help="Name of the config in the experiments dir."
    ) 
    parser.add_argument(
        '-resume',
        required=False,
        type=Path,
        help="Path to a log directory to resume from."
    )
    parser.add_argument(
        '-checkpoint',
        required=False,
        default=None,
        type=Path,
        help="Name of the checkpoint file to resume from."
    )
    parser.add_argument(
        '-debug',
        action='store_true',
        help=(
            "Logs an eval image to tensorboard every 5 iterations and "
            "overrides a log with the same name if existent."
        )
    )
    args = parser.parse_args()

    # load config

    checkpoint_path = None

    if args.config:
        # if config is given, always use this config
        if not args.config.exists():
            # assume that its the name of a file from experiments dir
            config_path = Path("experiments") / args.config.name
        else:
            config_path = args.config

    elif args.resume:
        # otherwise take config from resume directory
        config_name = args.resume.name.rsplit("-", maxsplit=1)[0]
        config_path = args.resume / f"{config_name}.gin"

    else:
        raise ValueError("-config or -resume is required.")
    
    # set logdir and logname

    if args.resume:
        # set logdir to resume directory
        logdir = args.resume
        logname = args.resume.stem

    else:
        # create a new logdir
        current_time = int(((datetime.datetime.now())-(datetime.datetime(2023, 1, 1, 0, 0))).total_seconds() / 60)
        logname = f"{args.config.stem}-{current_time}"
        logdir = Path(f"logs/tramnerf/{logname}/")

    # load checkpoint

    if args.checkpoint:
        # if checkpoint is given, always load this
        checkpoint_path = args.checkpoint

    elif args.resume:
        # otherwise take newest checkpoint from resume directory

        if not args.resume.exists():
            raise FileNotFoundError(
                f"Resume directory '{args.resume.absolute()}' does not exist."
            )

        checkpoint_paths = [
            p for p in logdir.glob("*.pt")
            if "_opt" not in p.name and p.name != "best.pt"
        ]

        if len(checkpoint_paths) > 0:
            checkpoint_path = max(
                checkpoint_paths,
                key=lambda p: int(p.stem)
            )
        else:
            print(
                f"No checkpoint was found in '{args.resume.absolute()}'. "
                "Starting from scratch..."
            )
            print(logdir, list(logdir.glob("*.pt")))

    # if args.resume is not None:
    #     # look for checkpoint file
    #     config_path = f'{args.resume}'
    #     logdir = Path(args.resume).parent
    #     checkpoint_paths = [p for p in logdir.glob("*.pt") if "_opt" not in p.name]
    #     if args.checkpoint is None:
    #         checkpoint_path = max(checkpoint_paths, key=lambda p: int(p.stem))
    #     else:
    #         checkpoint_path = Path(args.checkpoint)
    #         assert checkpoint_path.exists()
    # else:
    #     checkpoint_path = None
    #     config_path = None

    # if args.config is not None:
    #     if config_path is not None:
    #         print("replacing config from resume dir with", args.config)
    # else:
    #     args.config = Path(args.resume).name
    
    # config_path = f'experiments/{args.config}'
    
    # if args.config is None and args.resume is None:
    #     raise Exception("Provide -config or -resume.")
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file '{config_path.absolute()}' does not exist."
        )

    print(f"reading '{config_path.absolute()}'")
    gin.parse_config_file(config_path)
    
    config = Config()
    torch.manual_seed(config.seed)

    if not args.resume:
        if args.debug and logdir.exists():
            print("removing", logdir, "because debug mode is active...")
            shutil.rmtree(logdir)
        
        print(f"creating log directory '{logdir.absolute()}'")
        logdir.mkdir()
        shutil.copyfile(config_path, logdir / config_path.name)

    writer = SummaryWriter(str(logdir))
    
    device = ("cuda:0" if torch.cuda.is_available() else "cpu")        

    # load dataset

    train_dataset = data.SubjectLoader(
        subject_id=config.scene_name,
        root_fp="/data/holland/Masterarbeit/data",
        split="train",
        num_rays=config.bsz,
        patch_size=config.patch_size,
        scale=config.scale,
        center=config.center
    )
    train_dataset.images = train_dataset.images.to(device)
    train_dataset.camtoworlds = train_dataset.camtoworlds.to(device)
    train_dataset.K = train_dataset.K.to(device)

    test_dataset = data.SubjectLoader(
        subject_id=config.scene_name,
        root_fp="/data/holland/Masterarbeit/data",
        split="test",
        num_rays=None,
        scale=config.scale,
        center=config.center
    )
    test_dataset.images = test_dataset.images.to(device)
    test_dataset.camtoworlds = test_dataset.camtoworlds.to(device)
    test_dataset.K = test_dataset.K.to(device)

    # init model

    if config.use_mc_model:
        radiance_field = TraMNeRFRough().to(device)
    else:
        radiance_field = TraMNeRF().to(device)

    # load checkpoint

    if checkpoint_path is not None:
        print("Loading model state from", checkpoint_path)
        checkpoint_file: TraMNeRF = torch.load(checkpoint_path)
        radiance_field.load_state_dict(checkpoint_file.state_dict())

    # setup loss registry

    LOSS_FUNCTIONS.set_render_fn(partial(
        radiance_field,
        randomized=True,
        white_bkgd=config.white_background,
        near=config.nearplane,
        far=config.farplane,
        return_debug_images=False
    ))

    # init optimizers

    if config.reduce_on_plateau_factor is not None:
        assert config.reduce_on_plateau_min_lr is not None
        assert config.reduce_on_plateau_cooldown is not None
        assert config.reduce_on_plateau_min_lr is not None

        network_optimizer_lr_scheduler_factory = partial(
            torch.optim.lr_scheduler.ReduceLROnPlateau,
            factor=config.reduce_on_plateau_factor,
            patience=config.reduce_on_plateau_patience,
            cooldown=config.reduce_on_plateau_cooldown,
            min_lr=config.reduce_on_plateau_min_lr
        )
    else:
        network_optimizer_lr_scheduler_factory = lambda x: None

    optimizers = {
        "network_optimizer": ScheduledOptimizer(
            parameters=radiance_field.get_network_parameters(),
            lr=config.lr,
            first_iteration=0,
            check_grads=config.check_for_nan_grads,
            clip_grads=config.use_gradient_clipping,
            scheduler_factory=network_optimizer_lr_scheduler_factory
        )
    }

    # load optimizer checkpoints
    
    if checkpoint_path is not None:
        checkpoint_opt_path = checkpoint_path.with_stem(checkpoint_path.stem + "_opt")
        optimizers["network_optimizer"].load(checkpoint_opt_path)
        print("Loaded network optimizer state from", checkpoint_opt_path)

    # define run variables

    if checkpoint_path is not None:
        step = int(checkpoint_path.stem) + 1
        print("starting at iteration", step)
    else:
        step = 1
    first_eval_step = True
    best_psnr = 0
    cur_psnr = 0

    # define static loss data fn

    def get_loss_data(dataset_sample: dict):
        loss_data = dict()

        # add ground truth color

        loss_data["ground_truth"] = dataset_sample["pixels"]

        # add static dataset info

        loss_data["train_cam_to_worlds"] = train_dataset.camtoworlds
        loss_data["train_intrinsics"] = train_dataset.K
        loss_data["train_images"] = train_dataset.images

        # add iteration info

        loss_data["step"] = torch.tensor([step])

        return loss_data

    # define writer functions using the config and model

    OUTPUT_TO_IMAGE_DICT = setup_writers(config, num_mirrors=len(radiance_field.mirrors), num_bounces=radiance_field.num_bounces)
    
    def write_debug_outputs_to_tensorboard(outputs: dict[str, torch.Tensor], prefix: str = ""):
        for key, tensor in outputs.items():
            writer_info = OUTPUT_TO_IMAGE_DICT[key]
            if first_eval_step or not writer_info.only_once:
                mapped = writer_info.to_image_fn(tensor).cpu().numpy()
                if len(mapped.shape) == 2:
                    writer.add_image(prefix + key, mapped, global_step=step, dataformats='HW')
                elif len(mapped.shape) == 3:
                    writer.add_image(prefix + key, mapped, global_step=step, dataformats='HWC')
                else:
                    raise RuntimeError(f"Can't write mapped output of key '{key}' with shape {mapped.shape}.")

    # run training

    pbar = TrainProgressBar(
        num_train_it=config.iterations,
        batch_size=config.bsz,
        num_eval_it=config.iterations // config.eval_every + 1,
        eval_size=test_dataset.width * test_dataset.height,
        first_it=step
    )

    while step <= config.iterations:
        for i in range(len(train_dataset)):

            if step > config.iterations:
                return

            # get data

            dataset_sample = train_dataset[i]
            rays = dataset_sample["rays"]
            normalized_viewdirs = rays.viewdirs / torch.linalg.norm(rays.viewdirs, dim=-1, keepdim=True)

            rays = Rays(
                origins=rays.origins,
                viewdirs=normalized_viewdirs,
                radii=rays.radii,
                cam_idxs=rays.cam_idxs,
                xy=rays.xy
            )

            # get network output

            rendered_results = radiance_field(
                rays=rays,
                randomized=True,
                white_bkgd=config.white_background,
                near=config.nearplane,
                far=config.farplane,
                return_debug_images=False
            )

            # add loss data

            rendered_results = {**rendered_results, **get_loss_data(dataset_sample)}

            # compute losses

            loss_params = dict(rendered_results=rendered_results, rays=rays)
            loss_dict = {name: LOSS_FUNCTIONS[name](**loss_params) for name in config.loss_weights}
            loss = sum(loss_dict[name][0] * config.loss_weights[name] for name in config.loss_weights)
            
            for _, loss_debug_tensors in loss_dict.values():
                if loss_debug_tensors is not None:
                    rendered_results = {**rendered_results, **loss_debug_tensors}

            # optimizer step

            if not config.check_for_nan_loss or loss.isfinite():
                loss.backward()
            
                for optimizer in optimizers.values():
                    optimizer.step(it=step, loss=loss)
                    optimizer.zero_grad(it=step)
                                    
            else:
                print(f"Warning: Skipping optimizer step {step} because loss = {loss.item()}")

            # save model

            if step % 5000 == 0 or step in config.custom_checkpoint_at:
                torch.save(radiance_field, f"logs/{config.model}/{logname}/{step}.pt")
                optimizers["network_optimizer"].save(Path(f"logs/{config.model}/{logname}/{step}_opt.pt"))

            # log metrics

            if step % 10 == 0:
                # log individual losses and their sum
                for loss_name, loss_value in loss_dict.items():
                    writer.add_scalar(f"train/{loss_name}", loss_value[0].item(), step)
                writer.add_scalar("train/loss", loss.item(), step)

                # log additional derived values
                cur_psnr = mse2psnr(loss_dict["fine_l2_rendering_loss"][0]).item()
                writer.add_scalar('train/psnr', cur_psnr, step)
                writer.add_scalar('train/mean_acc', rendered_results['fine_acc'].mean().item(), step)

                if optimizers["network_optimizer"].scheduler is not None:
                    writer.add_scalar('train/network_lr', optimizers["network_optimizer"].optimizer.param_groups[0]['lr'], step)

                if "fine_scheduled_p" in rendered_results:
                    writer.add_scalar('train/scheduled_p', rendered_results["fine_scheduled_p"].item(), step)

            # run evaluation on single test image

            if step % (5 if args.debug else config.eval_every) == 0 or step == 1:
                with torch.no_grad():
                    test_sample = test_dataset[config.test_image_id]

                    pbar.set_state("eval-test")

                    rendered_test = render_full_image(
                        model=radiance_field,
                        dataset_sample=test_sample,
                        loss_data_fn=get_loss_data,
                        render_keys=None,
                        chunk_size=config.bsz,
                        return_debug_images=True,
                        return_loss_outputs=True,
                        pbar=pbar,
                    )

                    mse = img2mse(rendered_test["fine_rgb"], rendered_test["ground_truth"])
                    psnr = mse2psnr(mse)

                    writer.add_scalar('test/mse', mse.item(), step)
                    writer.add_scalar('test/psnr', psnr.item(), step)

                    write_debug_outputs_to_tensorboard(outputs=rendered_test)

                    # additionally render a train view

                    if config.render_train_view_for_eval:
                        train_dataset.training = False
                        train_sample = train_dataset[config.train_eval_image_id]
                        train_dataset.training = True

                        pbar.set_state("eval-train")

                        rendered_train = render_full_image(
                            model=radiance_field,
                            dataset_sample=train_sample,
                            loss_data_fn=get_loss_data,
                            render_keys=None,
                            chunk_size=config.bsz,
                            return_debug_images=False,
                            return_loss_outputs=True,
                            apply_reflections=step > config.num_iterations_without_reflections,
                            pbar=pbar,
                            use_dense_sampling=config.use_dense_sampling
                        )

                        write_debug_outputs_to_tensorboard(outputs=rendered_train, prefix='train_')
                        
                    # save config in tensorboard

                    if first_eval_step:
                        writer.add_text("config", gin.operative_config_str(), 0)
                        first_eval_step = False


            step = step + 1
            pbar.set_state("train")
            pbar.step(psnr=cur_psnr)

if __name__ == "__main__":
    main()