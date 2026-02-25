# texstat

TexStat is a perceptually-motivated audio feature extraction library for texture sounds, implemented in PyTorch. It can be used both as a **loss function** for training generative models and as an **evaluation metric** for assessing audio quality and similarity.

## Features

- **Perceptual Audio Features**: Statistical texture representation (S_1, S_2, S_3) based on cochlear and modulation filterbanks
- **Dual Usage**:
  - **Loss Function**: Differentiable loss for training neural networks (GANs, VAEs, diffusion models)
  - **Evaluation Metrics**: FAD (Fréchet Audio Distance) and KAD (Kernel Audio Distance) for model evaluation
- **GPU Acceleration**: Full CUDA support for fast computation
- **Embedding Caching**: Automatic caching of computed embeddings for efficient re-evaluation
- **Flexible Segmentation**: Configurable segment size and hop size for data augmentation

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### As a Loss Function

```python
from texstat.functions import sub_statistics_mcds_feature_vector
import torch

# Compute perceptual features for two audio signals
features_real = sub_statistics_mcds_feature_vector(audio_real, coch_fb, mod_fb, downsampler, N_moments, alpha)
features_generated = sub_statistics_mcds_feature_vector(audio_generated, coch_fb, mod_fb, downsampler, N_moments, alpha)

# Use as loss (e.g., L1 or L2 distance)
loss = torch.nn.functional.l1_loss(features_generated, features_real)
loss.backward()
```

### As an Evaluation Metric


#### KAD (Kernel Audio Distance) ⭐ **Recommended**


```python
from texstat.kad import KAD_wrapper

# Initialize KAD with parameters
kad = KAD_wrapper(
    frame_size=88200,      # 2 seconds at 44.1kHz
    hop_size=44100,        # 50% overlap
    sampling_rate=44100,
    bandwidth=None,        # Auto-compute using median heuristic
    kernel='gaussian',     # 'gaussian', 'iq', or 'imq'
    device='cuda'          # or 'cpu'
)

# Compute KAD between two folders of audio files
kad_score = kad.score('path/to/real_audio/', 'path/to/generated_audio/')
print(f"KAD Score: {kad_score:.4f}")
```

#### FAD (Fréchet Audio Distance)

```python
from texstat.fad import FAD_wrapper

# Initialize FAD with parameters
fad = FAD_wrapper(
    frame_size=88200,      # 2 seconds at 44.1kHz
    hop_size=44100,        # 50% overlap
    sampling_rate=44100,
    device='cuda'          # or 'cpu'
)

# Compute FAD between two folders of audio files
fad_score = fad.score('path/to/real_audio/', 'path/to/generated_audio/')
print(f"FAD Score: {fad_score:.4f}")
```

## Evaluation Metrics

### KAD (Kernel Audio Distance) ⭐ **Recommended**
- **Non-parametric approach**: Uses Maximum Mean Discrepancy (MMD) with kernel methods
- **No distribution assumptions**: More robust to complex, multimodal distributions
- **Kernels available**: Gaussian RBF, Inverse Quadratic, Inverse Multiquadric
- **Use case**: More accurate for complex texture sounds, captures non-linear relationships
- **Lower is better**: Smaller distance indicates more similar distributions
- **Why preferred**: Does not assume Gaussian distributions, making it more suitable for the complex, multimodal nature of audio texture statistics

### FAD (Fréchet Audio Distance)
- **Parametric approach**: Assumes Gaussian distributions
- **Formula**: `d² = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2√(Σ₁Σ₂))`
- **Use case**: Fast computation, works well when distributions are approximately Gaussian
- **Lower is better**: Smaller distance indicates more similar distributions
- **Limitation**: May not fully capture the complexity of audio texture distributions

## Examples

See the `examples/` folder for detailed usage:
- `example_loss_function_usage.ipynb`: Using TexStat as a training loss
- `example_evaluation_metric_usage.ipynb`: Using FAD and KAD for evaluation
- `example_kad_usage.ipynb`: Detailed KAD usage with different kernels and bandwidths
- `example_multi_classes.ipynb`: Computing confusion matrices for multi-class evaluation

## Advanced Features

### Embedding Caching
Embeddings are automatically cached to disk for faster re-evaluation:
```python
# First run: computes and saves embeddings
score1 = kad.score('audio_folder_1/', 'audio_folder_2/')

# Second run: loads cached embeddings (much faster!)
score2 = kad.score('audio_folder_1/', 'audio_folder_2/')

# Force recomputation
score3 = kad.score('audio_folder_1/', 'audio_folder_2/', force_recompute=True)
```

### Overlapping Segments
Use `hop_size` parameter to create overlapping segments for more robust statistics:
```python
# No overlap
wrapper = KAD_wrapper(frame_size=88200, hop_size=88200, ...)

# 50% overlap (2x more data)
wrapper = KAD_wrapper(frame_size=88200, hop_size=44100, ...)

# 75% overlap (4x more data)
wrapper = KAD_wrapper(frame_size=88200, hop_size=22050, ...)
```

## Citation

If you use TexStat in your research, please cite:

```bibtex
@inproceedings{gutierrez2025statistics,
 title     = {A Statistics-Driven Differentiable Approach for Sound Texture Synthesis and Analysis},
 author    = {Esteban Gutiérrez and Frederic Font and Xavier Serra and Lonce Wyse},
 booktitle = {Proceedings of the 28th International Conference on Digital Audio Effects (DAFx25)},
 year      = {2025},
 address   = {Ancona, Italy},
 month     = {September},
 note      = {2--5 September 2025}
}
```

## License

See LICENSE file for details. 
