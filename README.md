# track-cheetah

A simple python script to locally run YOLO models for traffic detection.

## How to Use

**0. Run the script.**

```bash
python track_cheetah.py
```

**1. Specify configuration file.**

```bash
Specify a config file if you have one.
Enter path to config file, or leave it blank for manual configuration: █
```

The configuration file contains all the necessary information to run inference on a traffic video.
- Path to video file to analyze.
- When, where, and how to count vehicles.
- The model variant to use.
- Etc.

If you have a configuration file from a previous run, you can reuse it to rerun the model with the same configuration.

If this is your first time running the script, or you want to try a different configuration, leave it blank to proceed to manually.

**2. Specify video path.**

```bash
Specify a video file to analyze.
Enter path to video file: █
```

**3. Specify counting conditions.**

```bash
Specify counting conditions.
1. Single line: objects that pass the line will be counted.
2. Two lines: objects that pass line 1 first and then pass line 2 will be counted.
q. Proceed to counting.
Write "1", "2", or "q", and press Enter: █
```

`track-cheetah` comes with two "counting modes".
- **1. Single line**: Draw a line on the captured video, and count all vehicles that pass that line.
- **2. Two lines**: Draw two lines (line 1 and 2) on the captured video. Vehicles that pass line 1 first, and then pass line 2 will be counted.
    - The order between two lines matter! vehicles that pass line 2 first will not be counted.

Object detection/tracking models aren't perfect. Below are recommendations for good results.
- **Use two lines instead of one**. The **Single line** condition may erroneously count the same vehicle multiple times when it stops on top of the line.
- **Ensure vehicles do not get occluded between two lines of the same condition**. The tracking model often relabels vehicles when they disappear (due to occlusion) and reappear. The **Two lines** condition depends on the traking labels remaining constant.

**4. Specify video stride.**

```bash
Specify video stride.
Stride of 2 means the model will only process once every 2 frames.
High video stride -> fast but inaccurate.
Write a number, and press Enter: █
```

In some cases (like real-time traffic analysis), faster processing might be preferrable over more accurate results.
Specifying a higher video stride makes the model skip more frames, allowing for faster processing at the cost of lower accuracy.
In particular, the model may fail to detect/track fast moving objects.

**5. Specify a model.**

```bash
Specify a model.
ultralytics models are trained for general object detection.
1. ultralytics/yolo26n
2. ultralytics/yolo26s
3. ultralytics/yolo26m
4. ultralytics/yolo26l
5. ultralytics/yolo26x
Perception365 models are fine-tuned for traffic detection, but requires access to a gated repo.
6. Perception365/VehicleNet-Y26n
7. Perception365/VehicleNet-Y26s
8. Perception365/VehicleNet-Y26m
9. Perception365/VehicleNet-Y26x
Write "1" - "9", and press Enter: █
```

**YOLO26**: Models 1-5 are from [ultralytics](https://docs.ultralytics.com/models/yolo26/). They were trained on the more general COCO dataset with 80 common object categories.
**VehicleNet-Y26**: Models 6-9 are from [Perception365](https://huggingface.co/Perception365/VehicleNet-Y26m). They are finetuned versions of the **YOLO26** model on the UVH-26-MV dataset for traffic detection.

Note that **VehicleNet-Y26** is kept behind a gated huggingface repo, and the user will be prompted to login to their huggingface account to use it. The script will fail if the account doesn't have access to the gated model.

## Dependencies

- `python>=3.12`.
- Pytorch, compatible with your CUDA version. See https://pytorch.org/get-started/locally/.
- `opencv-python<4.13`. The latest version (4.13) seems to have issues with Qt font, best to avoid.
- `ultralytics`. See https://docs.ultralytics.com/quickstart/.
- `huggingface-hub`, if you wish to use finetuned models provided by https://huggingface.co/Perception365/VehicleNet-Y26m.
- `pandas`, for writing results to csv files.


## Getting Started

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

## Disclaimer

This project uses ultralytic's [YOLO26](https://docs.ultralytics.com/models/yolo26/) (licensed under GNU AGPL v3), and optionally Perception365's [VehicleNet-Y26](https://huggingface.co/Perception365/VehicleNet-Y26m) (licensed under Apache 2.0).
As GNU AGPL v3 is [compatible with Apache 2.0](https://www.apache.org/licenses/GPL-compatibility.html), this script is also distributed under the GNU AGPL v3 license (see LICENSE.md for more).

VehicleNet-Y26 is a gated model.
I don't understand how gated models work with open source licenses, but it should be best not to ship my project with their weights.
This project **does not include the weights for VehicleNet-Y26**, and therefore the user should request access to their gated repo individually.