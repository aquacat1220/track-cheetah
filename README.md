# track-cheetah

A simple python script to locally run YOLO models for traffic detection.

## Dependencies

- `python>=3.12`.
- Pytorch, compatible with your CUDA version. See https://pytorch.org/get-started/locally/.
- `opencv-python<4.13`. The latest version (4.13) seems to have issues with Qt font, best to avoid.
- `ultralytics`. See https://docs.ultralytics.com/quickstart/.
- `huggingface-hub`, if you wish to use finetuned models provided by https://huggingface.co/Perception365/VehicleNet-Y26m.
- `pandas`, for writing results to csv files.


## How to Use

1. Create a python virtual environment.

```bash
python -m venv .venv
```

2. Activate the environment.

On macOS/Linux:
```bash
source .venv/bin/activate
```
On Windows `cmd`:
```cmd
.venv\Scripts\activate
```
On Windows `powershell`:
```ps
.\venv\Scripts\Activate.ps1
```

3. Install dependencies.

3-1. Install pytorch, according to the instructions at https://pytorch.org/get-started/locally/. It should probably look something like:

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cuxx
```

3-2. Install everything else.

```bash
pip install "opencv-python<4.13" ultralytics huggingface-hub pandas jsonpickle
```

4. Run the script.
```bash
python track_cheetah.py
```

## Pretrained Models

This script uses ultralytic's [YOLO26](https://docs.ultralytics.com/models/yolo26/), and optionally allows traffic-finetuned models from [Perception365](https://huggingface.co/Perception365/VehicleNet-Y26m).