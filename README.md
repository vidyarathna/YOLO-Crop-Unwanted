# YOLO Object Detection and Cropping Script

This script uses YOLO (You Only Look Once) for object detection in an input image and crops out the detected objects based on the YOLO detections.

## Requirements

- Python 3.x
- OpenCV (`pip install opencv-python`)
- NumPy (`pip install numpy`)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/vidyarathna/your-repository.git
   cd your-repository
   ```

3. Download YOLO weights and configuration files (not included in this repository) and place them in a directory named `yolo-coco` within the repository:

   - `yolov3.weights`
   - `yolov3.cfg`
   - `coco.names`

## Usage

Run the script with the following command:

```bash
python crop_unwanted.py --image /path/to/your/image.jpg --yolo /path/to/your/yolo-coco
```

### Optional Arguments

- `-c`, `--confidence`: Minimum probability to filter weak detections (default: 0.5)
- `-t`, `--threshold`: Threshold when applying non-maxima suppression (default: 0.3)

## Output

The script will display the cropped images of detected objects from the input image.

## Example

```bash
python crop_unwanted.py --image /path/to/your/image.jpg --yolo /path/to/your/yolo-coco
```

### Notes:

- Replace `/path/to/your/image.jpg` with the path to your input image.
- Replace `/path/to/your/yolo-coco` with the path to the directory containing your YOLO configuration and weights files.
- You can add more sections or customize the README according to the specific details of your project and any additional features or functionalities you might have implemented.
