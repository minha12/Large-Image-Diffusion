# Learned representation-guided diffusion models for large-image generation


![teaser figure](./teaser.png)
## Requirements
To install python dependencies, 

```
conda env create -f environment.yaml
conda activate ldm
```

## Downloading + Organizing Data

Due to storage limitations, we cannot upload the image patches and embeddings used for training. However, training data can be curated by following these steps - 

### Download the WSIs


We train diffusion models on TCGA-BRCA, TCGA-CRC and Chesapeake Land cover datasets. For BRCA and CRC, we used the [DSMIL](https://github.com/binli123/dsmil-wsi) repository to extract 256 x 256 patches @ 20x magnification, and conditioned the diffusion models from [HIPT](https://github.com/mahmoodlab/HIPT) and [iBOT](https://github.com/owkin/HistoSSLscaling). We also train a model on 5x BRCA patches, conditioned on [CTransPath](https://github.com/Xiyue-Wang/TransPath) embeddings See Section 4.1 from our paper for more details.
 
### Prepare the patches

Once you clone the DSMIL repository, you can use the following command to extract patches from the WSIs. 

```
$ python deepzoom_tiler.py -m 0 -b 20
```

### SSL embeddings

Follow instructions in [HIPT](https://github.com/mahmoodlab/HIPT) / [iBOT](https://github.com/owkin/HistoSSLscaling) repository to extract embeddings for each patch.

## Pretrained models

We provide the following trained models


|  Dataset | # Training  images |  FID | Conditioning | Download link |
|:--------:|:------------------:|:----:|--------------|:-------------:|
| BRCA 20x |       15 Mil       | 6.98 | HIPT         |   [link](https://drive.google.com/drive/folders/1kZ69wVEHV3k3Zr1hgS3kftz9cfNb9BxA?usp=sharing)            |
|  CRC 20x |        8 Mil       | 6.78 | iBOT         |   [link](https://drive.google.com/drive/folders/1r1Kgcgy34rP3O-X4AqhQ09Sf1OZdHvm2?usp=sharing)            |
|   NAIP   |        667 k       | 11.5 | VIT-B/16     |  [link](https://drive.google.com/drive/folders/1mWy5wi-Tcpcb8-0n6eczzyjBoMAdK9fA?usp=sharing)             |
|  BRCA 5x |        976 k       | 9.74 | CTransPath   |  [link](https://drive.google.com/drive/folders/1NL0mpepFzYfrb4tH4NVzAYAWkirMSBuB?usp=sharing)             |




## Training

* **Customization:** Create a config file similar to [./configs/latent-diffusion/crc/only_patch_20x.yaml](./configs/latent-diffusion/crc/only_patch_20x.yaml) to train your own diffusion model.
* **Sample Dataset:** We provide a sample dataset [here](./notebooks/dataset_samples/brca_hipt_patches.pickle).

  Dataset Structure:
  - `brca_hipt_patches.pickle`:
    - List of 100 items, each containing:
      - `feat_20x`: Feature vector (384,) float32
      - `image`: RGB image patches (256,256,3) uint8
  
  - `brca_hipt_large_images.pickle`:
    - List of 15 items, each containing:
      - `feat_20x`: Feature matrix (16,384) float32
      - `img_20x_large`: RGB images (1024,1024,3) uint8

  Key Details:
  - Small patches: 256x256 pixels with 384-dim features
  - Large images: 1024x1024 pixels with 16 sets of 384-dim features
  - All images use RGB format (3 channels) with uint8 values
  - Features stored as float32 arrays

* **Loading Data:** See [./ldm/data/hybrid_cond/crc_only_patch.py](./ldm/data/hybrid_cond/crc_only_patch.py) for an example of how to load data.

  **Data Pipeline Overview:**

  1. Expected Directory Structure:
  ```
  root/
  ├── patches_{mag}_all.npy     # NumPy array of image paths
  ├── features.h5               # HDF5 file with SSL features
  └── {image_files}            # Actual image files referenced in patches_{mag}_all.npy
  ```

  2. Data Flow:
  ```mermaid
  graph TD
      A[Load patches list] --> B[Load features.h5]
      B --> C[Initialize Dataset]
      C --> D[Per-item Processing]
      D --> E[Batch Formation]
      
      subgraph "Per-item Processing"
          F[Load Image] --> G[Convert to RGB]
          G --> H[Normalize image]
          H --> I[Apply Random Flips]
          I --> L[Processed Image]
          J[Load SSL Features] --> K[Optional Feature Zeroing]
          K --> M[Processed Features]
      end
      
      D --> F
      D --> J
      L --> E
      M --> E
  ```

  3. Configuration Requirements:
     - `root`: Base directory path
     - `magnification`: Magnification level (e.g., "20x")
     - `p_uncond`: Probability for feature zeroing (default=0)

  4. Processing Steps:
  ```mermaid
  graph TD
      A[Get image path] --> B[Load image with PIL]
      B --> C[Convert to RGB array]
      C --> D[Verify 256x256 size]
      D --> E[Normalize image values]
      E --> F[Random flips]
      G[Load SSL features] --> H[Apply feature dropout]
      F --> I[Return batch dict]
      H --> I
  ```

  5. Output Format:
  ```python
  # Example of actual values and shapes
  batch = {
      "image": np.array(...),          # Shape: (256,256,3), Range: [-1,1]
      "feat_patch": np.array(...),     # Shape: (384,), SSL features
      "human_label": ""                # Empty string placeholder
  }
  
  # Actual processing code
  def process_image(image):
      # Convert to float32 and normalize to [-1,1] range
      return (image.astype(np.float32) / 127.5) - 1.0
      
  def process_features(features, p_uncond=0.1):
      # Apply random feature dropout
      if np.random.rand() < p_uncond:
          return np.zeros_like(features)
      return features
  ```

  6. DataLoader Configuration Example:
  ```yaml
  data:
    params:
      batch_size: 100
      num_workers: 16
      wrap: false
  ```

* **Embedding Guidance:** We feed the SSL embedding via cross-attention (See Line 52 of [./ldm/modules/encoders/modules.py](./ldm/modules/encoders/modules.py)).


Example training command:

```
python main.py -t --gpus 0,1 --base configs/latent-diffusion/crc/only_patch_20x.yaml
```

## Sampling

Refer to these notebooks for generating images using the provided models:

* **Image Patches:** [./notebooks/brca_patch_synthesis.ipynb](./notebooks/brca_patch_synthesis.ipynb)
* **Large Images:** [./notebooks/large_image_generation.ipynb](./notebooks/large_image_generation.ipynb)

