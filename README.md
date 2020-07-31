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
After installing the requirements for this project, you can run a demo using the commands below;
```
cd src
```

```
python3 app.py --input "../bin/demo.mp4"
```

## Documentation
While running the project using the command above is sufficient for a demo of project, extra arguments can be parsed also for extra features. To view arguments and uses, Run the command below; 
```
python3 app.py --help
```

The different command line arguments include;

```
-i, --input, which is used to specify the path to input file or "cam" to use webcam (Required)

-e, --extension, which is used to specify the path to cpu extension (Optional)

-d, --device, which is used to specify device to run model on, options: FPGA, GPU, MYRIAD, defaults to CPU (Optional)

-v, --visualization, specify to view visualization from the other models. face for face detection, facel for face landmark, head for head pose, gaze for gaze estimation. for visualization of more than one model, seperate with a space") (Optional)

```

The scripts for preprocessing the models and making inference are present in the src folder which is also where app.py file used to run project resides. In this folder, The base_model.py contains the reusable code used to instantiate the other models and also contains the methods used for processing the inputs, loading the model, making inferences. The other model files are used to process the outputs of models. input_feeder.py is used to specify type of input that inference is to be made on and the mouse_controller.py file is used for mouse movements using the gaze estimation. app.py contains the main file where all model loading takes place, inference and eventual mouse movement using the results obtained. The bin folder contains the video used for quick demo. When models are downloaded using the commands above, they are stored in a new top level intel folder.

* Project Structure

```
|--bin
    |--demo.mp4
|--intel
|--src
    |--app.py
    |--base_model.py
    |--face_detection.py
    |--facial_landmarks_detection.py
    |--gaze_estimation.py
    |--head_pose_estimation.py
    |--input_feeder.py
    |--mouse_controller.py
|--README.md
|--requirements.txt
```

## Benchmarks
Results obtained on my local CPU using the FP32 model;

* Total model load time - 0.943542 seconds
* Total inference time - 1.434543 seconds
* Total Frames per second - 10.044346

Results obtained on my local CPU using the FP16 model;

* Total model load time - 0.902414 seconds
* Total inference time - 1.391234 seconds
* Total Frames per second - 9.854323

## Results
The FP16 model gives inference faster than the FP32 model, this is due to the lower precision of the FP16 model. FP32 model is more accurate than the FP16 model however. Inference time can be improved however by analysing models individually and pruning to remove least efficient layers or neurons to make them more efficient.

### Edge Cases
Application doesn't work very well in poorly lit areas as models can't pick up features properly so please use in a well lit area or environment

### License
MIT license