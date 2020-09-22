# Project Write-Up

This write-up is based on the template given in the course. It provides explanation of the model selection, model convertion and further questions.

## Explaining Custom Layers

The OpenVINO toolkit support various layer types of the compatible modelling frameworks (TensorFlow, Caffe, MXNet, Kaldi and ONYX). However, which layers are supported differs from framework to framework. A list of supported layers for each framework can be found [here](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html).  

Each layer that is not in this layer is classified as custom layer for the OpenVINO toolkit.
Potential reasons for the use of custom layers are that they enable individual operations and manipulation of tensors if the operation is not natively supported in the framework or not supported in the OpenVINO toolkit. 

OpenVINO supports handling of custom layers in the Model Optmizer and brings along tools for the conversion of custom layers. Therefore, following steps need to be implemented:
- Generate the Extension Template Files Using the Model Extension Generator
- Edit the Extractor and the Operation Extension Template File
- Using Model Optimizer to Generate IR Files Containing the Custom Layer
- Edit and Compile the Device Extension Template Files (CPU/GPU)
- Execute the Model with the Custom Layer

An example for this process is presented in [this repository](https://github.com/david-drew/OpenVINO-Custom-Layers/tree/master/2019.r2.0).  
Besides this extension registering option, there is a second option to use custom layers in Caffee and Tensorflow. Both frameworks can be used installed locally to offload computations during inference.

## Comparing Model Performance

During the project I downloaded and tested a couple of different modelsfor the application. I challenged myself to test different source frameworks, so I searched for models from the Tensorflow, Caffee and ONYX/Pytorch framework. 

To compare models before and after conversion to Intermediate Representations (IR) it is in general possible to compare model size, model accuracy and inference time pre- and post-conversion. However, I could not find a way to determine accuracy and inference time of the pre-conversion models, since I only used the Udacity Jupyter Lab workspace to work on the project, besides some local testing on a weaker 6th generation laptop CPU.

So I was only able to compare the model size before and after the conversion per model. Between converted IR models I could determine accuracy, inference time and model size.

This table show the result to compare the different models. For the IR model size only the binary file is listed here, since the XML file is comparatively small. For the inference time the unit is milliseconds (ms), benchmarked in the workspace environment. The accuracy is determined for how many pedestrians have been detected from the test video with **6** pedestrians, with a confindence of 60%. Lost frames are only quantified visually. 
 Name | Type | Size (original) | Size (IR, binary) |Inference time [ms] | Accuracy | Lost frames
 --- | --- | --- | --- | --- | --- | ---
 Model1 | SSD MobileNetv2 | 67Mb | 65Mb | 70ms | 14oo6 | a lot of clearly identifiable
 Model2 | SSDlite MobileNetv2 | 19Mb | 8.6Mb | 30ms | 17oo6 | a lot of clearly identifiable
 Model3 | SSD300 VGG | 101Mb | 101Mb | 900ms | 7oo6 | some clearly identifiable
 Model4 | F-RCNN Inception | 55Mb | 51Mb | - | 0oo6 | -
 Model5 | person-detection-retail-0013 | - | 2.8Mb | 45ms | 6oo6 | negligible

 It can be clearly seen, that:
 - SSDlite fullfilled the expectation that it is lighter than SSD
 - SDD and SSDlite had unsatisfying performance regarding accuracy, f.e. the second person in the test video ```resources/Pedestrian_Detect_2_1_1.mp4``` was only detected in the first and last seconds of its appearance in the frame
 - SSD300 performed better accuracy wise, but was unsatisfyable slow
 - F-RCNN models could not be implemented correctly
 - The intel model worked best and performed accurate and fast on the test video!



## Assess Model Use Cases

There are some potential used cases for the finished project, the people counter app:
- During a pandemic, as the current COVID19 pandemic, the app could ensure that people follow the distance and hygiene regulations. Imagine an elevator, transport vehicle or similar small closed room is only allowed to be used if a maximum number of people is observed. For example, the control system then could forbid that doors are closing if the maximum number of peaple is exceeded.
- Similar the app could be used to determine if pedestrians stop at certain situations in traffic, e.g. at a traffic sign. This could help to additionally warn the pedestrians or vehicles ariving to improve secure road passing.
- The app could also be used to do something like AB testing with crowds. Imagine a shop wants to test if some advertisement or presentation in their shop window showcase is more attracting to pedestrians compared to others. Therefore the app could count stopping pedestrians per showcase and an AB comparison could help the store to improve their services.   


## Assess Effects on End User Needs

The effects on end user needs of the following properties are discussed:
- **lighting**: the lighthing of the camera scene that is captured is a important property to take care of. If the lighting is to weak then a camera can have problems with adapting in such way that the contrast is to weak to record shapes and colors correctly. And because this is what is needed to detect obejcts with a network, this would decrease the accuracy of the app. However, if lighting conditions can not be changed, then maybe training and using an infrared camera could help to tackle the problem.
- **model accuracy**: Accuracy always come with a the trade-off, accuracy vs. ressources. So if you plan to have a high accuracy in detection than you have to use a powerful hardware. However, if you do not have enough budget for a powerful hardware than you have to deal with lower accuracy. But, it should be possible to use a lower accuracy with a strong tracking algorithm to get precise results in counting with basic hardware. This could be done with an identification of the objects and a memory which objects has been seen where in the frame.
- **image size**: 


Discuss lighting, model accuracy, and camera focal length/image size, and the effects these may have on an end user requirement.

## Model Research

Standard procedure for model research:
1. Download the model archive with wget:
    ```
    wget [link to model archive]
    ```
2. Extract the archive with tar:
    ```
    tar -xvf [path to model archive]
    ```
3. Use the model optimizer to convert to the Intermediate Representation (IR):  
    Use an environment variable for the model optimizer:
    ```
    export MO=/opt/intel/openvino/deployment_tools/model_optimizer/mo.py
    ```
    - if source framework was Tensorflow:
        ```
        $MO -input_model [model path].pb --tensorflow_object_detection_api_pipeline_config [pipeline path].config --tensorflow_use_custom_operations_config [support file path].json --reverse_input_channels 
        ```
    - if source framework was Caffee:
        ```
        $MO --input_model [model path].caffemodel --input_proto [model deploy file path].prototxt
        ```
    - if source framework was ONYX:
        ```
        $MO --input_model [model path].onnx
        ```
- Or use the directly by Intel supported models. Therefore use the model downloader. Available models can be explored with:
    ```
    /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --print-all
    ```
    Then use the downloader to download the model in all available or specific precision, f.e. name=person-detection-retail-0013 and precision=FP32.
    ```
    /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name [model name] --precision [specific precision]
    ```

In investigating potential people counter models, I tried each of the following three models:

- Model 1: SSD MobileNet v2 COCO (Tensorflow)
  - Model Source: [Download archive](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)
  - I converted the model using the Tensorflow procedure, as presented above.
  - The model was insufficient for the app because the confidence on the persons in the video was to low, so in a lot of frames the persons could not be detected. 
  - I tried different input sizes and different confidence levels.
  
- Model 2: SSD Lite MobileNet v2 COCO (Tensorflow)
  - Model Source: [Download archive](http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz)
  - I converted the model using the Tensorflow procedure, as presented above.
  - Similar result as Model 1, but with less performance used and shorter inference time.
  - Again, I tried different input sizes and different confidence levels to fix the accuracy issue.

- Model 3: SSD 300 based on VGG (Caffee)
  - Model Source: [Model site](https://docs.openvinotoolkit.org/latest/omz_models_public_ssd300_ssd300.html)
  - I converted the model using the Caffee procedure, as presented above.
  - The model was insufficient for the app because it was to heavy for the workspace and local environment. The inference time was so slow. But the accuracy was okay.
  - Again, I tried different input sizes and different confidence levels to fix the inference performance issue.

- Model 4: F-RCNN Inception v2 COCO (Tensorflow)
  - Model Source: [Download archive](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz)
  - I converted the model using the Tensorflow procedure, as presented above.
  - The model was insufficient for the app because it produces a segmentation map and not an bounding box.
  - I tried to implent a box drawer from the segmentation map, but it was really slow and inefficient.

- Model 5: person-detection-retail-0013 based on MobileNetV2-like (Intel)
  - Model Source: [Model site](https://docs.openvinotoolkit.org/2019_R1/_person_detection_retail_0013_description_person_detection_retail_0013.html)
  - I downloaded the model using the model downloader procedure, as presented above.
  - The model had the best performance, no accuracy issues!