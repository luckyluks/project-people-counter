# Model commands: tools, model references and run commands

## additional references:
- **convert main page**: https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model.html
- **detailed tf convert page**: https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_Object_Detection_API_Models.html 
**tensorflow models from intel page**: https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html#supported_topologies
- **caffee model names**: https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Caffe.html 
- **model downloader hint**: https://community.intel.com/t5/Intel-Distribution-of-OpenVINO/Build-and-convert-Caffe-SSD300-VGG16-model-to-IR/td-p/1165772 


## script tools used:

- wget for downloads
    ```
    wget
    ```
- tar for unpacking of *.tar.gz files
    ```
    tar -xvf
    ```

### models used:
- ### model 01:
    link: http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz

    ```
    /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model=ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb --output=detection_classes,detection_scores,detection_boxes,num_detections --tensorflow_object_detection_api_pipeline_config=ssd_mobilenet_v2_coco_2018_03_29/pipeline.config --tensorflow_use_custom_operations_config=/opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --input=image_tensor --input_shape=[1,300,300,3] --reverse_input_channels --data_type FP16
    ```

    ```
    python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
    ```

    ```
    result: **SUCCESS** [ SUCCESS ] Total execution time: 71.86 seconds. 
    ```


- ### model 02:
    link;
    http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz

    ```
    /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model=ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb --output=detection_classes,detection_scores,detection_boxes,num_detections --tensorflow_object_detection_api_pipeline_config=ssdlite_mobilenet_v2_coco_2018_05_09/pipeline.config -tensorflow_use_custom_operations_config=/opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --input=image_tensor --input_shape=[1,300,300,3] --reverse_input_channels --data_type FP16
    ```

    ```
    [ SUCCESS ] Generated IR model.
    [ SUCCESS ] XML file: /home/workspace/./frozen_inference_graph.xml
    [ SUCCESS ] BIN file: /home/workspace/./frozen_inference_graph.bin
    [ SUCCESS ] Total execution time: 52.28 seconds. 
    ```

- ### model 03:
    link: found over model downloader

    go to downloader:  
    ```cd /opt/intel/openvino_2019.3.376/deployment_tools/tools/model_downloader/```  
    list all available models:  
    ```python downloader.py --print_all```  
    download specific:  
    ```sudo python downloader.py --name ssd300 -o /home/workspace/```

    ```
    python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model SSD_300x300_ft/VGG_VOC0712Plus_SSD_300x300_ft_iter_160000.caffemodel --input_proto SSD_300x300_ft/deploy.prototxt
    ```

- ### model 04:
    link: http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz

    ```
    python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config faster_rcnn_inception_v2_coco_2018_01_28/pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json --input_shape=[1,300,300,3]
    ```

    ```
    [ SUCCESS ] Generated IR model.
    [ SUCCESS ] XML file: /home/workspace/./frozen_inference_graph.xml
    [ SUCCESS ] BIN file: /home/workspace/./frozen_inference_graph.bin
    [ SUCCESS ] Total execution time: 148.33 seconds.
    ```

## RUN!
- ### FRNN
    ```
    python main.py -m models/frozen_inference_graph_FRNN_IC_V2.xml -i resources/Pedestrian_Detect_2_1_1.mp4 -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
    

- ### SSDLite mobilenet
    ```
    python main.py -m models/frozen_inference_graph_SSDlite_MN_V2.xml -i resources/Pedestrian_Detect_2_1_1.mp4 -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
    ```

- ### VGG 
    ```
    python main.py -m models/VGG_VOC0712Plus_SSD_300x300_ft_iter_160000.xml -i resources/Pedestrian_Detect_2_1_1.mp4 -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -pt 0.8 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
    ```

- ### SSD mobilenet

    python main.py -m models/frozen_inference_graph_SSD_MN_V2.xml -i resources/Pedestrian_Detect_2_1_1.mp4 -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
    ```

- ### INTEL model
    ```
    python main.py -m models/person-detection-retail-0013-FP32.xml -i resources/Pedestrian_Detect_2_1_1.mp4 -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
    ```