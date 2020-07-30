# Computer Pointer Controller

This project utilizes the Intel OpenVINO library to control your mouse pointer using your gaze estimation. Project utilizes four different models; face detection model to locate the face, facial landmark detection model to detect location of eyes, head pose estimation to determine the facing direction of the head and the gaze estimation model, to determine the overall gaze of person in frame. Models depend on inputs from each other to function optimally.

## Project Set Up and Installation
To set up this project, Clone then proceed to download the models using the OpenVINO model downloader below;

##### Face Detection Model
```
python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "face-detection-adas-binary-0001"
```
##### Head Pose Estimation Model
```
python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "head-pose-estimation-adas-0001"
```
##### Facial Landmark Detection Model
```
python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "landmarks-regression-retail-0009"
```
##### Gaze Estimation Model
```
python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "gaze-estimation-adas-0002"
```

Models would be downloaded in an Intel folder at the top level of project directory.

Next you can create a Virtual Environment to prevent requirements from installing globally using the command below;
```
python3 -m venv venv
```
and activating the virtual environment using the command below;
```
source venv/bin/activate
```
Next install the requirements using the command below;
```
pip3 install -r requirements.txt
```

## Demo
After installing the requirements for this project, you can run a demo using the command below;

```
python3 app.py --input "../bin/demo.mp4"
```

## Documentation
While running the project using the command above is sufficient for a demo of project, extra arguments can be parsed also for extra features. To view arguments, Run the command below; 
```
python3 app.py --help
```

## Benchmarks
Results obtained on my local CPU using the FP32 model;

Total model load time - 0.943542 seconds
Total inference time - 1.434543 seconds
Total Frames per second - 10.044346

## Results
Given that four models load together to give the total model load time, The model load time seems fair and since models are high precision. Inference time can be improved by analysing models individually and pruning to remove least efficient layers or neurons to make them more efficient.

### Edge Cases
Application doesn't work very well in poorly lit areas as models can't pick up features properly so please use in a well lit area or environment

### License
MIT license