from copy import deepcopy
from pathlib import Path

from .utils import calculate_fid
import torch
from torch.optim import Adam
from torch.nn import Module, ModuleList
from torchvision.utils import save_image
import math

from rectified_flow_pytorch.rectified_flow import RectifiedFlow

from ema_pytorch import EMA
from accelerate import Accelerator

from tqdm import tqdm

# helpers


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def divisible_by(num, den):
    return (num % den) == 0


# reflow wrapper


class Reflow(Module):
    def __init__(
        self,
        rectified_flow: RectifiedFlow,
        frozen_model: RectifiedFlow | None = None,
        *,
        batch_size=16,
        num_train_steps=20000,
        save_results_every: int = 100,
        calculate_fid_every: int = 100,  # New parameter
        checkpoint_every: int = 5000,
        is_online=False,
        learning_rate=3e-4,
        adam_kwargs: dict = dict(),
        accelerate_kwargs: dict = dict(),
        sample_steps=16,
        checkpoints_folder: str = "./checkpoints",
        results_folder: str = "./results",
        num_samples: int = 16,
        use_ema=False,
        beta=0.999,
        verbose=True,
        fid_n_samples=3,
    ):
        super().__init__()
        model, data_shape, device = (
            rectified_flow.model,
            rectified_flow.data_shape,
            rectified_flow.device,
        )
        assert exists(data_shape), "`data_shape` must be defined in RectifiedFlow"

        self.batch_size = batch_size
        self.data_shape = data_shape
        self.device = device

        self.verbose = verbose
        self.losses = []
        self.fid_n_samples = fid_n_samples

        self.sample_steps = sample_steps

        self.num_train_steps = num_train_steps

        self.model = rectified_flow

        self.is_online = is_online

        self.optimizer = Adam(
            rectified_flow.parameters(), lr=learning_rate, **adam_kwargs
        )
        self.accelerator = Accelerator(**accelerate_kwargs)

        self.use_ema = use_ema
        self.ema_model = None

        self.sample_buffer = []
        self.noise_buffer = []

        if use_ema:
            self.ema_model = EMA(
                self.model,
                forward_method_names=("sample",),
                beta=beta,
            )

            self.ema_model.to(self.accelerator.device)

        if not is_online:
            if not exists(frozen_model):
                frozen_model = deepcopy(rectified_flow)

                for p in frozen_model.parameters():
                    p.detach_()

            self.frozen_model = frozen_model
        else:
            self.frozen_model = rectified_flow

        self.checkpoints_folder = Path(checkpoints_folder)
        self.results_folder = Path(results_folder)

        self.checkpoints_folder.mkdir(exist_ok=True, parents=True)
        self.results_folder.mkdir(exist_ok=True, parents=True)

        self.checkpoint_every = checkpoint_every
        self.save_results_every = save_results_every
        self.calculate_fid_every = calculate_fid_every  # New parameter

        self.num_sample_rows = int(math.sqrt(num_samples))
        assert (
            self.num_sample_rows**2
        ) == num_samples, f"{num_samples} must be a square"
        self.num_samples = num_samples

        assert self.checkpoints_folder.is_dir()
        assert self.results_folder.is_dir()

    def device(self):
        return self.device

    def parameters(self):
        return self.model.parameters()

    def sample(self, *args, **kwargs):
        return self.model.sample(steps=self.sample_steps)

    def sample_train(self, fname, calculate_fid_flag=True):
        # Calculate FID over multiple samples if requested
        fid = 0

        if calculate_fid_flag:
            iterator = range(self.fid_n_samples)
            if self.verbose:
                iterator = tqdm(iterator, desc="Calculating FID")

            for i in iterator:
                with torch.no_grad():
                    # Get samples from current model
                    sampled = self.model.sample(batch_size=1, steps=self.sample_steps)
                    sampled.clamp_(0.0, 1.0)

                    # Get samples from frozen/EMA model
                    noise = torch.randn((1, *self.data_shape), device=self.device)
                    if self.use_ema:
                        eval_model = default(self.ema_model, self.frozen_model)
                        frozen_sampled = eval_model.sample(
                            noise=noise, steps=self.sample_steps
                        )
                    else:
                        frozen_sampled = self.frozen_model.sample(
                            steps=self.sample_steps, noise=noise
                        )

                    # Calculate FID for each channel
                    for c in range(3):
                        frozen_feature = frozen_sampled[0, c, :, :].cpu().numpy()
                        sample_feature = sampled[0, c, :64, :64].cpu().numpy()
                        fid_channel = calculate_fid(frozen_feature, sample_feature)
                        fid += fid_channel / (3 * self.fid_n_samples)

                    if self.verbose:
                        iterator.set_postfix(fid=f"{fid:.4f}")

            if self.verbose:
                print(f"Final FID: {fid:.4f}")
        else:
            # Just generate a sample without calculating FID
            with torch.no_grad():
                sampled = self.model.sample(batch_size=1, steps=self.sample_steps)
                sampled.clamp_(0.0, 1.0)

        # Save the generated sample
        save_image(sampled, fname)
        return sampled, fid

    def forward(self):
        noise = torch.randn((self.batch_size, *self.data_shape), device=self.device)
        if self.use_ema:
            eval_model = default(self.ema_model, self.frozen_model)
            sampled_output = eval_model.sample(noise=noise, steps=self.sample_steps)
        else:
            sampled_output = self.frozen_model.sample(
                steps=self.sample_steps, noise=noise
            )

        loss = self.model(sampled_output, noise=noise)
        return loss

    def train(self):
        fid_ar = []

        iterator = range(self.num_train_steps)
        if self.verbose:
            iterator = tqdm(iterator, desc="Training Reflow")

        for ind in iterator:
            step = ind + 1

            loss = self.forward()
            self.losses.append(loss.item())

            if self.verbose:
                iterator.set_postfix(loss=f"{loss.item():.3f}")

            self.accelerator.backward(loss)

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.use_ema:
                self.ema_model.ema_model.data_shape = self.model.data_shape
                self.ema_model.update()

            self.accelerator.wait_for_everyone()

            # Save results based on save_results_every - no FID calculation here
            if divisible_by(step, self.save_results_every):
                prefix = "online" if self.is_online else "reflow"
                with torch.no_grad():
                    sampled = self.model.sample(batch_size=1, steps=self.sample_steps)
                    sampled.clamp_(0.0, 1.0)
                    save_image(
                        sampled,
                        str(
                            self.results_folder
                            / f"{prefix}_{self.sample_steps}_results.{step}.png"
                        ),
                    )

            # Calculate FID separately based on calculate_fid_every
            if divisible_by(step, self.calculate_fid_every):
                prefix = "online" if self.is_online else "reflow"
                _, fid = self.sample_train(
                    fname=str(
                        self.results_folder
                        / f"{prefix}_{self.sample_steps}_fid.{step}.png"
                    ),
                    calculate_fid_flag=True,
                )
                fid_ar.append(fid)

            if divisible_by(step, self.checkpoint_every):
                self.save(f"checkpoint.{step}.pt")

            self.accelerator.wait_for_everyone()

        print("training complete")

        return fid_ar, self.losses

    def retrive_model(self):
        return self.model

    def save(self, path):
        save_package = dict(
            model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(),
        )

        if self.use_ema:
            save_package["ema_model"] = self.ema_model.state_dict()

        torch.save(save_package, str(self.checkpoints_folder / path))
