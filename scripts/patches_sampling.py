import torch
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
parent_dir = str(SCRIPT_DIR.parent)
print(parent_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils import get_model, load_model_from_config
from ldm.util import instantiate_from_config
import numpy as np
from torch.utils.data import default_collate
from ldm.models.diffusion.plms import PLMSSampler
from torchvision.utils import make_grid
from einops import rearrange
from PIL import Image
import pickle
from fire import Fire
import os


class ImageGenerator:
    def __init__(
        self,
        model_path: str = "./logs/brca_hipt_20x/",
        checkpoint: str = "last.ckpt",
        device: str = "cuda:0",
        output_dir: str = "generated_images",
        dataset_path: str = "./dataset_samples/brca_hipt_patches.pickle"
    ):
        """
        Initialize the image generator.
        
        Args:
            model_path: Path to the model directory
            checkpoint: Name of the checkpoint file
            device: Device to run the model on
            output_dir: Directory to save generated images
            dataset_path: Path to the dataset pickle file
        """
        self.model_path = Path(model_path)
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.dataset_path = dataset_path
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self.model, self.config = get_model(self.model_path, self.device, checkpoint)
        self.sampler = PLMSSampler(self.model)

    def prepare_dataset(
        self,
        batch_size: int = 8,
        save_samples: bool = True
    ):
        """
        Prepare the dataset for generation.
        
        Args:
            batch_size: Number of images to generate in each batch
            save_samples: Whether to save the prepared samples
        """
        data = instantiate_from_config(self.config.data)
        data.prepare_data()
        data.setup()
        data.batch_size = batch_size

        data.datasets['train'].p_uncond = 0
        data.datasets['train'].aug = 0
        ds = data.datasets['train']

        idx = np.random.randint(0, len(ds), 100)
        items = [ds[i] for i in idx]

        for item in items:
            item['image'] = (127.5*(item['image'] + 1)).astype(np.uint8)
            del item['feat_5x']
            del item['human_label']

        if save_samples:
            with open(self.dataset_path, "wb") as f:
                pickle.dump(items, f)

        return items

    def generate_images(
        self,
        batch_size: int = 8,
        scale: float = 2.0,
        ddim_steps: int = 50,
        shape: list = [3, 64, 64]
    ):
        """
        Generate images using the loaded model.
        
        Args:
            batch_size: Number of images to generate
            scale: Unconditional guidance scale
            ddim_steps: Number of DDIM sampling steps
            shape: Shape of the generated images
        """
        # Load prepared samples
        with open(self.dataset_path, "rb") as f:
            items = pickle.load(f)

        batch = default_collate(items[:batch_size])

        with torch.no_grad(), self.model.ema_scope():
            batch["feat_20x"] = batch["feat_20x"].to(self.device)
            
            batch_uncond = {**batch}
            batch_uncond["feat_20x"] = torch.zeros_like(batch["feat_20x"])

            cc = self.model.get_learned_conditioning(batch)
            uc = self.model.get_learned_conditioning(batch_uncond)

            samples_ddim, _ = self.sampler.sample(
                S=ddim_steps,
                conditioning=cc,
                batch_size=batch_size,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=uc,
            )

            x_samples_ddim = self.model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            x_samples_ddim = (x_samples_ddim * 255).to(torch.uint8)
            x_samples_ddim = x_samples_ddim.cpu()
            samples_real = batch["image"].permute(0,3,1,2)

        # Create grids
        grid_real = rearrange(make_grid(samples_real, nrow=8), 'c h w -> h w c').cpu().numpy()
        grid_syn = rearrange(make_grid(x_samples_ddim, nrow=8), 'c h w -> h w c').cpu().numpy()

        # Save images
        Image.fromarray(grid_real).save(self.output_dir / "real_images.png")
        Image.fromarray(grid_syn).save(self.output_dir / "synthetic_images.png")

        return "Images generated and saved successfully!"

def main(
    model_path: str = "./logs/brca_hipt_20x/",
    checkpoint: str = "last.ckpt",
    device: str = "cuda:0",
    output_dir: str = "generated_images",
    dataset_path: str = "./dataset_samples/brca_hipt_patches.pickle",
    batch_size: int = 8,
    scale: float = 2.0,
    ddim_steps: int = 50,
    shape: list = [3, 64, 64]
):
    """
    Main function to generate images.
    
    Args:
        model_path: Path to the model directory
        checkpoint: Name of the checkpoint file
        device: Device to run the model on
        output_dir: Directory to save generated images
        dataset_path: Path to the dataset pickle file
        batch_size: Number of images to generate
        scale: Unconditional guidance scale
        ddim_steps: Number of DDIM sampling steps
        shape: Shape of the generated images
    """
    generator = ImageGenerator(
        model_path=model_path,
        checkpoint=checkpoint,
        device=device,
        output_dir=output_dir,
        dataset_path=dataset_path
    )
    
    # Prepare dataset
    generator.prepare_dataset(batch_size=batch_size)
    
    # Generate images
    result = generator.generate_images(
        batch_size=batch_size,
        scale=scale,
        ddim_steps=ddim_steps,
        shape=shape
    )
    
    print(result)

if __name__ == "__main__":
    Fire(main)