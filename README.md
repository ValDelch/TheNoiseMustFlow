<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>

<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

![Test Status](https://github.com/ValDelch/TheNoiseMustFlow/actions/workflows/tests.yml/badge.svg)

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/ValDelch/TheNoiseMustFlow">
    <img src="https://github.com/user-attachments/assets/3c8b4fc8-7d13-4788-9ca9-ad8a12856087" alt="Logo" width="256" height="256">
  </a>

  <h3 align="center">The Noise Must Flow</h3>

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
- Visualization tools and dashboards
- Intuitive interfaces to explore denoising
- Code to help **you** build your own diffusion workflows

---

## 🎯 Features (Coming Soon)

⏳ Coming soon

---

## 🔧 Built With

[![Python][Python]][Python-url]
[![Pytorch][PyTorch]][PyTorch-url]
<!--
* [![Vue][Vue.js]][Vue-url]
-->

---

## 🗺️ Roadmap

### 📦 Core Features
- [x] Project setup and repo structure
- [ ] Implement basic DDPM architecture
  - [ ] Forward process (noise scheduling)
  - [ ] Reverse sampling
  - [ ] Loss functions
- [ ] Add beta-schedule variants (linear, cosine, etc.)
- [ ] DDIM sampling
- [ ] Improve support for multi-resolution outputs
- [ ] Add support for conditional diffusion (class labels, images, etc.)

### 🧪 Training & Evaluation
- [ ] Training loop with logging
- [ ] Metrics: loss, FID, PSNR (optionally)
- [ ] Checkpointing & resume
- [ ] Support for distributed training
- [ ] Add pretrained model zoo

### 🎨 Visualization
- [ ] Save sampled images during training
- [ ] Animate denoising steps
- [ ] Plot noise schedule and beta curves
- [ ] TensorBoard support

### 🧰 Developer Experience
- [ ] Modular code with config files
- [ ] CLI to run training and sampling
- [ ] Documentation with usage examples

### 🎛️ User Interfaces
- [ ] Interface for sampling
- [ ] Web-based demo / notebook

### 🌐 Educational Material
- [ ] High-level explanation in README
- [ ] Notebooks: DDPM math & intuition
- [ ] Tutorials: build your own DDPM from scratch

---

## 🔧 Installation

⏳ Coming soon

---

## 📂 Project Structure

Here's a quick overview of the main components in this repository:

| Path                      | Description                                                  |
|---------------------------|--------------------------------------------------------------|
| `configs/`                | YAML/JSON config files for models and training               |
| `core/`                   | Core logic for diffusion models and sampling                 |
| ├── `diffusion.py`        | Forward and reverse diffusion process                        |
| ├── `models.py`           | U-Net, VAE, or other neural architectures                    |
| ├── `building_blocks.py`  | Modular building blocks used inside models                   |
| ├── `sample.py`           | Sampling routines from noise                                 |
| └── `schedule.py`         | Noise schedule and beta/variance utilities                   |
| `trainer/`                | Training workflows and evaluation tools                      |
| ├── `train.py`            | Training loop orchestration                                 |
| ├── `losses.py`           | Loss functions used during training                          |
| └── `metrics.py`          | Evaluation metrics like FID, PSNR                            |
| `ui/`                     | Gradio or web-based interface for interactive demos          |
| └── `interface.py`        | Script to launch the UI                                      |
| `utils/`                  | Utility functions                                             |
| ├── `logger.py`           | Logging, experiment tracking                                 |
| └── `visualize.py`        | Visual tools for training or generation                      |
| `scripts/`                | CLI scripts to run training or inference                     |
| ├── `train_ddpm.py`       | Command-line entry point to train DDPM                       |
| └── `sample_ddpm.py`      | Command-line entry point to sample from trained model        |
| `notebooks/`              | Jupyter notebooks for tutorials and mathematical insights    |
| └── `intro_to_diffusion.ipynb` | Notebook introducing DDPMs                          |
| `tests/`                  | Unit tests for validating modules                            |
| `assets/`                 | Diagrams, logos, generated outputs                           |
| `docs/`                  | Documentation files                                         |
| `requirements.txt`        | List of Python dependencies                                  |
| `pyproject.toml`          | Optional packaging/build config                              |
| `README.md`               | Project description                                          |
| `LICENSE`                 | Project license                                              |

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
| `docs`   | Documentation only changes              |
| `refactor` | Code refactoring without behavior change |
| `test`   | Adding or updating tests                |
| `ci`     | Changes to CI/CD config                 |
| `perf`   | Performance improvements                |

#### Examples

```bash
git commit -m "feat(trainer): add early stopping"
git commit -m "fix(core): handle NaNs in beta schedule"
git commit -m "docs: improve README structure"
```

## 👥 Top Contributors

<a href="https://github.com/ValDelch/TheNoiseMustFlow/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ValDelch/TheNoiseMustFlow" alt="contrib.rocks image" />
</a>

## 📬 Contact

Valentin Delchevalerie - 📧 vdelchevalerie@gmail.com - 🔗 [LinkedIn](https://www.linkedin.com/in/valentin-delchevalerie-075ab4194/)

## 🧠 Acknowledgments

This project is based on many different resources. Here is a list of some of the major inspirations:

* [Ho et al., 2020] — (https://arxiv.org/abs/2006.11239)
* [Umar Jamil](https://github.com/hkproj/pytorch-stable-diffusion) — and his video "[Coding Stable Diffusion from scratch in PyTorch](https://www.youtube.com/watch?v=ZBKpAp_6TGI&t=13980s)"

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
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/valentin-delchevalerie-075ab4194

[Python]: https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/
[PyTorch]: https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[PyTorch-url]: https://pytorch.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
