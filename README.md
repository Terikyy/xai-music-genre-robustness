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

### Option 1: Automated Setup (Recommended)

Run the setup script to automatically install dependencies and prepare the environment:

```bash
chmod +x setup.sh
./setup.sh
```

### Option 2: Manual Setup

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Download the dataset:**
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

More detailed documentation and results analysis can be found here:  
- [Documentation](/docs/documentation.md)
- [Methodology](/docs/methodology.md)
- [Results Analyisis](/docs/results_analysis.md)
