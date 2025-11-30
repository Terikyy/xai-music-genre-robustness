# CNN Robustness for Music Genre Classification with Grad-CAM

## Project Information

**Course:** Explainable AI  
**University:** DHBW Stuttgart  

**Name:** Erik von Heyden  
**Email:** inf23066@lehre.dhbw-stuttgart.de  
**Student ID:** 8720832  

**GitHub:** [xai-music-genre-robustness](https://github.com/Terikyy/xai-music-genre-robustness)

## Research Question

How robust is a CNN for music genre classification against noise perturbations, and how can Grad-CAM be used to visualize which regions of the spectrogram the model uses for its decisions and how these change due to adversarial perturbations?

## Data Sources

- **Dataset:** [GTZAN Music Genre Classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data)
- **Adversarial Robustness:** [Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox)

## Setup

### Requirements

- **Python Version:** 3.10 - 3.13 (tested on Python 3.12)
- **PyTorch:** CPU or GPU version (CUDA 11.8 for GPU support)
- **Operating System:** Linux, macOS, or Windows (WSL required for automated setup script)

### Option 1: Automated Setup (Recommended)

Run the setup script to automatically install dependencies and prepare the environment:

```bash
chmod +x setup.sh
./setup.sh
```

The script will:
- Create a virtual environment
- Install base dependencies from `requirements.txt`
- Prompt you to choose between CPU or GPU (CUDA) PyTorch installation
- Download the GTZAN dataset from Kaggle (requires Kaggle API token)

### Option 2: Manual Setup

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **Install base dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Install PyTorch:**
   
   Choose one based on your hardware:
   
   - **For GPU (CUDA 11.8):**
     ```bash
     pip install -r requirements-gpu.txt
     ```
   
   - **For CPU only:**
     ```bash
     pip install -r requirements-cpu.txt
     ```

4. **Download the dataset:**
   - Download the [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data)
   - Extract it to `./data/raw/` (create the directory if it doesn't exist yet)
   - The file structure should look like this:
     ```
     data/raw/
     ├── features_3_sec.csv
     ├── features_30_sec.csv
     ├── genres_original/
     └── images_original/
     ```

## Documentation 

More detailed documentation and results analysis can be found in [Documentation](/docs/documentation.md).
