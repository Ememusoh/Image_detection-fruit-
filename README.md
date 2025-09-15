# Apple Image Detection

This project is designed to detect apples in images using a deep learning model. It includes data preprocessing, model training, and inference scripts.

## Project Structure

```
.
├── apple_detector_model.h5         # Trained Keras model for apple detection
├── preprocess.py                   # Script for preprocessing images and labels
├── data/
│   └── raw/
│       ├── labels.csv              # CSV file containing image labels
│       └── images/                 # Directory containing raw images
│           ├── 1.jpg
│           ├── 2.jpg
│           └── ...
```

## Getting Started

### Prerequisites
- Python 3.7+
- TensorFlow (with Metal support for Mac)
- NumPy
- Pandas

Install dependencies:
```bash
pip install tensorflow-macos tensorflow-metal numpy pandas
```

### Data
- Place your images in `data/raw/images/`.
- Ensure `labels.csv` is present in `data/raw/` with appropriate labels for each image.

### Preprocessing
Run the preprocessing script to prepare your data:
```bash
python preprocess.py
```

### Model
- The trained model is saved as `apple_detector_model.h5`.
- You can use this model for inference or further training.

## Usage
- Modify and run `preprocess.py` to preprocess your data.
- Use the trained model for apple detection tasks.

## License
This project is for educational purposes.
