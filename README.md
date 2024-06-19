# Master-Project1

TASKS
```
1. Install YOLOV8 Detection
2. From boundaries of YOLOV8, put a text tag at any one of the corner (Overlay Original Image with Text at the pizel of the any one corner)
3. Make the Text Tag fixed to some feature of the object (Second Stage) - Example: Text always close to Front Right Wheel.
```
Steps
1. Create Conda environment (masterproject)
```
conda create --name masterproject python=3.9
```
2. Activate the specific project environment 
``` 
conda activate masterproject
```
3. Clone yolov8(Usefull for detection and semantic segmentation)
```
# This is command to download yolov8 
git clone https://github.com/ultralytics/ultralytics.git
```
4. Install YOLOV8
```
pip install ultralytics
```
5. Verify installation
```
yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'
```
6. Run Script
```
cd Arun/Scripts
python detect.py
```
To Do
1. Try to use sematic segmentation to get a sailanecy map
2. Come up with the optimization method to minimise overlap on sailency features and minimise distance to the corect object
