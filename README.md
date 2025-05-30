# ML-Model: Analog & Digital Meter Reading Detection

This project provides a complete pipeline for detecting and reading analog and digital meters using deep learning models (YOLO, CNNs) and OCR. It supports detection, cropping, and reading of analog dials, resistance, and temperature panels, with FastAPI for serving predictions.

## Features

- **Analog Meter Reading**: Detects meter region, crops, and computes the reading using YOLO and geometric analysis.
- **Digital Panel Reading**: Detects and crops resistance/temperature panels, then recognizes digits using a two-stage YOLO + CNN pipeline.
- **FastAPI Backend**: REST API for uploading images and getting meter readings.
- **Jupyter Notebooks**: For training, evaluation, and experimentation.
- **EasyOCR Integration**: For fallback or additional OCR tasks.

## Project Structure

- [`app.py`](app.py): FastAPI backend for inference.
- [`analog.py`](analog.py): Analog meter detection and reading logic.
- [`Lenet_res.py`](Lenet_res.py), [`Lenet-temp.py`](Lenet-temp.py): CNN models for digit recognition.
- [`res_temp_N2N.py`](res_temp_N2N.py): End-to-end resistance/temperature reading pipeline.
- [`ultra.ipynb`](ultra.ipynb): Main notebook for detection, cropping, and reading experiments.
- [`train_1_0_1.ipynb`](train_1_0_1.ipynb): Model training and evaluation notebook.
- [`data.yaml`](data.yaml): YOLO dataset configuration.
- `Models/`: Pretrained YOLO and CNN models.
- `dataset/`, `dataset_synthetic/`: Training and validation datasets.
- `test_images/`, `cropped_images/`, `extracted_frames/`: Example/test images and intermediate results.

## Installation

1. **Clone the repository**  
   ```sh
   git clone https://github.com/NitinRwt/ML-Model/
   cd ML-Model
   ```

2. **Install dependencies**  
   It is recommended to use a virtual environment.
   ```sh
   pip install -r requirements.txt
   ```

3. **(Optional) Download/prepare datasets and models**  
   Place your images in the appropriate folders (`dataset/`, `test_images/`, etc.) and models in `Models/`.

## Usage

### FastAPI Server

Start the API server:
```sh
uvicorn app:app --reload
```
Visit `http://localhost:8000/docs` for the interactive API documentation.

### Jupyter Notebooks

- Run [`ultra.ipynb`](ultra.ipynb) for detection and reading experiments.
- Use [`test.ipynb`](test.ipynb) to train or evaluate models.

### Command Line

You can run scripts directly for batch processing or testing:
```sh
python res_temp_N2N.py
python analog.py
```

## Training

See [`train_1_0_1.ipynb`](test.ipynb) for details on training YOLO and CNN models.

## Docker

A `Dockerfile` is provided for containerized deployment:
```sh
docker build -t ml-model .
docker run -p 8000:8000 ml-model
```

## Data & Models

- Place your YOLO and CNN models in the `Models/` directory.
- Organize datasets as specified in `data.yaml` and notebooks.

## Ignore Patterns

The following files/folders are ignored by git (see [.gitignore](.gitignore)):
- Intermediate results: `runs/`, `cropped_images/`, `extracted_frames/`, etc.
- Datasets and test images: `dataset/`, `test_images/`, etc.
- Model outputs and caches: `__pycache__/`, `*.zip`, etc.

