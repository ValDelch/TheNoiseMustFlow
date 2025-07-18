<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>

<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]

![Test Status](https://github.com/ValDelch/TheNoiseMustFlow/actions/workflows/tests.yml/badge.svg)

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/ValDelch/TheNoiseMustFlow">
    <img src="https://github.com/user-attachments/assets/d3bab683-33b7-45b7-b063-1a412f21d0b6" alt="Logo" width="320" height="320">
  </a>

  <p align="center">
    A creative lab for building, training, and understanding diffusion models.<br>
    Learn the flow of noise. Generate beauty from randomness.
    <br />
    <br />
    <!--
    <a href="https://github.com/othneildrew/Best-README-Template">View Demo</a>
    &middot;
    <a href="https://github.com/othneildrew/Best-README-Template/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/othneildrew/Best-README-Template/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
    -->
  </p>
</div>

# About This Project

![TheNoiseMustFlow-illustration](https://github.com/user-attachments/assets/74873629-8264-4664-b3d9-5fcda3f9bf19)

Welcome to **The Noise Must Flow**, your all-in-one lab for understanding, building, and experimenting with **diffusion models** — the noisy heartbeats behind today's most powerful generative AIs.

This project aims to make generative diffusion approachable and powerful, offering:

- Clean and modular implementations
- Visualization tools
- Code to help **you** build your own diffusion workflows

---

## 🔧 Built With

[![Python][Python]][Python-url]
[![Pytorch][PyTorch]][PyTorch-url]

---

## 🗺️ Roadmap

### 📦 Core Features
- [x] Project setup and repo structure
- [x] Implement basic DDPM architecture
  - [x] Forward process (noise scheduling)
  - [x] DDPM sampling
  - [x] DDIM sampling
  - [x] VAE and U-Net architectures
  - [x] Loss functions
- [ ] Implement some utility functions to validate the configuration files

### 🧪 Training & Evaluation
- [x] VAE Training loop
- [x] Diffusion model training loop
- [ ] Metrics: loss, FID, PSNR
- [x] Checkpointing & resume
- [ ] Support for distributed training
- [x] Support for mixed precision training
- [ ] Add pretrained model zoo

### 🎨 Visualization
- [ ] Animate denoising steps
- [ ] Plot noise schedule and beta curves
- [x] TensorBoard support

### 🧰 Developer Experience
- [ ] Modular code with config files
- [ ] CLI to run training and sampling
- [ ] Documentation with usage examples

### 🌐 Educational Material
- [x] Hands-on: training a diffusion model on MNIST
- [ ] High-level explanation in README
- [ ] Notebooks with explanations of key concepts and tutorials

---

## 🔧 Installation

For an easy install:
```bash
pip install git+https://github.com/ValDelch/TheNoiseMustFlow.git
```

To install the project and start experimenting with the project, follow these steps:

1. **Clone the repository**:
  ```bash
  git clone https://github.com/ValDelch/TheNoiseMustFlow
  cd TheNoiseMustFlow
  ```
2. **Create a virtual environment** (optional but recommended):
  ```bash
  python3 -m venv .venv
  ```
3. **Activate the virtual environment**:
  ```bash
  source .venv/bin/activate
  ```
4. **Install the required packages and the project**:
  ```bash
  pip install -e .
  ```

The different components of the project will be installed along with their dependencies. After that, you can start using the different components of the project, e.g.,
```python
from TheNoiseMustFlow.core.schedulers import NoiseScheduler
scheduler = NoiseScheduler()
```

---

## 📂 Project Structure

Here's a quick overview of the main components in this repository:

| Path                                  | Description                                                  |
|---------------------------------------|--------------------------------------------------------------|
| `assets/`                             | Diagrams, logos, and generated outputs                       |
| `configs/`                            | YAML/JSON configuration files for models and training        |
| ├── `default_VAE.yaml`                | Default configuration for VAE architecture                   |
| └── `default_UNet.yaml`               | Default configuration for U-Net architecture                 |
| `docs/`                               | Project documentation and references                         |
| `notebooks/`                          | Jupyter notebooks for tutorials and mathematical insights    |
| ├── `__a__Noise_Schedulers.ipynb`     | Notebook introducing the Noise Schedulers                    |
| └── `__b__Samplers.ipynb`             | Notebook introducing the Samplers                            |
| `examples/`                           | Example scripts for training and inference                   |
| └── `MNIST/`                          | Example training script for MNIST dataset                    |
|     ├── `MNIST_VAE.yaml`              | Configuration for VAE on MNIST                               |
|     ├── `MNIST_UNet.yaml`             | Configuration for U-Net on MNIST                             |
|     ├── `MNIST.py`                    | Implementation of the MNIST dataset for training             |
|     ├── `sample.ipynb`                | Jupyter notebook to sample from a trained model              |            
|     └── `train_mnist.py`              | Script to train a diffusion model on MNIST                   |
| `src/TheNoiseMustFlow/`               | Root source folder for all Python modules                    |
| ├── `core/`                           | Core logic for diffusion models and sampling                 |
| │   ├── `basic_components/`           | Modular low-level building blocks for models                 |
| │   │   ├── `basic_blocks.py`         | Simple building blocks                                       |
| │   │   ├── `decoder_blocks.py`       | Decoder-specific building blocks                             |
| │   │   ├── `encoder_blocks.py`       | Encoder-specific building blocks                             |
| │   │   ├── `encodings.py`            | Positional or learned encodings                              |
| │   │   └── `functional_blocks.py`    | Functional blocks like attention, normalization, etc.        |
| │   ├── `models.py`                   | U-Net, VAE, or other neural architectures                    |
| │   ├── `samplers.py`                 | Sampling routines from noise                                 |
| │   └── `schedulers.py`               | Noise schedule and beta/variance utilities                   |
| ├── `scripts/`                        | CLI scripts to run training or inference                     |
| │   ├── `sample_ddpm.py`              | Command-line entry point to sample from trained model        |
| │   └── `train_ddpm.py`               | Command-line entry point to train DDPM                       |
| ├── `tests/`                          | Unit tests for validating modules                            |
| ├── `trainer/`                        | Training workflows and evaluation tools                      |
| │   ├── `custom_lr_scheduler.py`      | Custom learning rate schedulers                              |
| │   ├── `losses.py`                   | Loss functions used during training                          |
| │   ├── `metrics.py`                  | Evaluation metrics like FID, PSNR                            |
| │   └── `train.py`                    | Training loop orchestration                                  |
| └── `utils/`                          | Utility functions                                            |
|     ├── `config_utils.py`             | Configuration file utilities (check validity, etc.)          |
|     ├── `logger.py`                   | Logging, experiment tracking                                 |
|     └── `visualize.py`                | Visual tools for training or generation                      |
| `LICENSE`                             | Project license (MIT)                                        |
| `README.md`                           | Project overview and usage instructions                      |
| `pyproject.toml`                      | Build and packaging configuration with project metadata      |

---

## 🤝 Contributing

Feel free to contribute to the project. Any contributions you make are **greatly appreciated**.

### 🔧 How to Contribute

1. Fork the repository
2. Create a new branch:  
   `git checkout -b feature/my-cool-feature`
3. Make your changes and **use conventional commits** when committing:  
   `git commit -m "feat(core): add new sampler"`
4. Push to your fork:  
   `git push origin feature/my-cool-feature`
5. Open a pull request to the `main` branch

---

### 📝 Commit Message Format

We follow the **[Conventional Commits](https://www.conventionalcommits.org/)** standard to enable automated versioning and changelog generation.

The general format is:

```
<type>[optional scope]: <description>
```

#### Common types (used by Commitizen):

| Type     | Purpose                                 |
|----------|-----------------------------------------|
| `feat!`  | Add a new major feature (breaking change) |
| `feat`   | Add a new feature                       |
| `fix!`   | Fix a bug (breaking change)             |
| `fix`    | Fix a bug                               |
| `chore`  | Maintenance tasks (no production code)  |

#### Examples

```bash
git commit -m "feat(trainer): add early stopping"
git commit -m "fix(core): handle NaNs in beta schedule"
```

For the branch name, we recommend using the format:

```
feature/<description>
```
for adding new features, or

```
fix/<description>
```
for bug fixes, or finally

```
chore/<description>
```
for maintenance tasks.

---

## 👥 Top Contributors

<a href="https://github.com/ValDelch/TheNoiseMustFlow/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ValDelch/TheNoiseMustFlow" alt="contrib.rocks image" />
</a>

## 📬 Contact

Valentin Delchevalerie - 📧 vdelchevalerie@gmail.com - 🔗 [LinkedIn](https://www.linkedin.com/in/valentin-delchevalerie-075ab4194/)

## 🧠 Acknowledgments

This project is based on several resources. Here is a list of some of the major inspirations:

* [Ho et al., 2020] — (https://arxiv.org/abs/2006.11239)
* [Umar Jamil](https://github.com/hkproj/pytorch-stable-diffusion) — and his video "[Coding Stable Diffusion from scratch in PyTorch](https://www.youtube.com/watch?v=ZBKpAp_6TGI&t=13980s)"
* Jeremy Hummel and his [step-by-step guide](https://github.com/LambdaLabsML/diffusion-from-scratch)

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]:  https://img.shields.io/github/contributors/ValDelch/TheNoiseMustFlow.svg?style=for-the-badge
[contributors-url]: https://github.com/ValDelch/TheNoiseMustFlow/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/ValDelch/TheNoiseMustFlow.svg?style=for-the-badge
[forks-url]: https://github.com/ValDelch/TheNoiseMustFlow/network/members
[stars-shield]: https://img.shields.io/github/stars/ValDelch/TheNoiseMustFlow.svg?style=for-the-badge
[stars-url]: https://github.com/ValDelch/TheNoiseMustFlow/stargazers
[issues-shield]: https://img.shields.io/github/issues/ValDelch/TheNoiseMustFlow.svg?style=for-the-badge
[issues-url]: https://github.com/ValDelch/TheNoiseMustFlow/issues

[Python]: https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/
[PyTorch]: https://img.shields.io/badge/PyTorch-2.7.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[PyTorch-url]: https://pytorch.org/
