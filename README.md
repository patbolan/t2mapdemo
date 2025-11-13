# T2Map Demo - Interactive Demonstrations for Quantitative MRI with CNNs

This mini web application provides interactive demonstrations to complement the published manuscript [*Improved Quantitative Parameter Estimation for Prostate T2 Relaxometry Using Convolutional Neural Networks*, Bolan et al., MAGMA 2024.](https://link.springer.com/article/10.1007/s10334-024-01186-3)


## Overview

This paper explores the use of convolutional neural networks (CNNs) to replace conventional curve fitting for quantitative MRI applications. The results demonstrate that CNNs provide better quantitative performance, particularly in noisy regions, due to improved representation of the noise distribution and the inherent regularization of the convolutional architecture.

This web application hosts three interactive demonstrations using the data from the paper to help develop intuition for how CNNs perform in quantitative MRI parameter estimation.

## Technology Stack

- **Backend**: Flask (Python web framework)
- **Frontend**: HTML, CSS, JavaScript
- **Math Rendering**: KaTeX
- **Medical Imaging**: NiBabel for NIFTI file handling
- **Visualization**: Matplotlib, NumPy
- **Deployment**: Gunicorn WSGI server, Docker support

## Installation

### Prerequisites
- Python 3.x
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/patbolan/t2mapdemo.git
cd t2mapdemo
```

2. Install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the Application

### Development Mode
```bash
python3 app.py
```

or, to run on local network:

```bash
flask --app . run --host=0.0.0.0
```


### Production Mode (with Gunicorn)
```bash
gunicorn --bind 0.0.0.0:5000 wsgi:app
```

The application will be available at `http://localhost:5000`

## Docker Deployment

Build and run using Docker:

```bash
# Build the image
docker build -t t2mapdemo .

# Run the container
docker run -d -p 5000:5000 --name t2mapdemo-container t2mapdemo

# Stop and clean up
docker stop t2mapdemo-container
docker rm t2mapdemo-container
docker rmi t2mapdemo
```


## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use this work, please cite:

```
Bolan, P.J., et al. (2024). Improved Quantitative Parameter Estimation for Prostate T2 
Relaxometry Using Convolutional Neural Networks. MAGMA. 
https://doi.org/10.1007/s10334-024-01186-3
```
