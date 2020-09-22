"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt
import numpy as np

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

# Number of frames that the detection holds if detection flickers
DETECTION_HOLD_FRAMES = 5
FONT = cv2.FONT_HERSHEY_COMPLEX
COLOR = ()


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.6,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-md", "--maximum_detections", type=int, default=3,
                        help="Maximum count of detections in the frame "
                        " before a warning appears")
    parser.add_argument("-mt", "--maximum_time", type=int, default=10,
                        help="Maximum time a detected person is in the frame"
                        " before a warning appears (in seconds)")
    
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    
    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the network class
    infer_network = Network()

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(model_xml=args.model, device=args.device, cpu_extension=args.cpu_extension)

    ### TODO: Handle the input stream ###
    single_image_mode = False
    input_argument = args.input.lower()
    if input_argument == 'CAM':
        input_stream = 0
    elif input_argument.endswith((".jpg", ".bmp", ".png")):
        single_image_mode = True
        input_stream = args.input
    elif args.input.endswith('.mp4') or args.input.lower().endswith('.avi'):
        input_stream = args.input
    else:
        log.warn("File type not supported: {}".format(args.input))
        sys.exit("ERROR: unknown input file type!")
        
    # Get input shape of the network
    net_input_shape = infer_network.get_input_shape()
    net_output_shape = infer_network.get_output_shape()
    log.info("Model input shape is: {}".format(net_input_shape))
    log.info("Model output shape is (desired): {}".format(net_output_shape))

    # Open the input capture
    cap = cv2.VideoCapture(input_stream)
    cap.open(input_stream)
    
    # Read and log playback stats
    input_width = int(cap.get(3))
    input_height = int(cap.get(4))
    input_fps = int(cap.get(cv2.CAP_PROP_FPS))
    log.info("Playback of \"{}\" is open! ({}x{}) - {}fps".format(input_stream, input_width, input_height, input_fps))

    # Initialize playback variables
    # request_id=0 # can be set for multiple streams
    person_counter= 0
    person_counter_prev = 0
    person_counter_total = 0
    duration_frames = 0
    duration_frames_prev = 0    

    ### TODO: Loop until stream is over ###
    while cap.isOpened():

        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break

        ### TODO: Pre-process the image as needed ###
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1)) # HWC => CHW
        p_frame = p_frame.reshape(1, *p_frame.shape) #(N,C,H,W)

        ### TODO: Start asynchronous inference for specified request ###
        inference_start = time.time()
        infer_network.exec_net(p_frame)

        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:
            
            ### TODO: Get the results of the inference request ###
            inference_stop = time.time() - inference_start
            net_output = infer_network.get_output()            
            
            ## TODO: Extract any desired stats from the results ###
            frame, persons_current_count = draw_boxes(frame,
                                            net_output,
                                            args.prob_threshold)
            
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            
            # Determine counted detections and duration
            last_duration_seconds = None
            
            # If a rising/falling flank occurs
            if persons_current_count != person_counter:
                # Update/Rotate counters
                person_counter_prev = person_counter
                person_counter = persons_current_count
                # If duration was minimum hold then "start new duration counting", else "continue previous duration"
                if duration_frames >= DETECTION_HOLD_FRAMES:
                    duration_frames_prev = duration_frames
                    duration_frames = 0
                else:
                    duration_frames = duration_frames_prev + duration_frames
                    duration_frames_prev = 0
            else:
                # Increase duration counter
                duration_frames += 1
                # "Delay" used to prevent flicker
                if duration_frames >= DETECTION_HOLD_FRAMES:
                    # if more persons are in the frame than before, update total count
                    if duration_frames == DETECTION_HOLD_FRAMES and person_counter > person_counter_prev:
                        person_counter_total += person_counter - person_counter_prev
                    # if less persons are in the frame than before, update duration
                    elif duration_frames == DETECTION_HOLD_FRAMES and person_counter < person_counter_prev:
                        last_duration_seconds = int((duration_frames_prev / input_fps) * 1000)
  
            # Decorate the frame with additional meta, if playback is an image, correct the total sum
            if single_image_mode:
                person_counter_total = persons_current_count
            # update the seconds counter    
            duration_seconds = (duration_frames / input_fps)
            frame = decorate_meta_to_frame(frame, 
                                           inference_stop, 
                                           persons_current_count, 
                                           person_counter_total, 
                                           duration_seconds)
            
            # Send a status every frame (is required for the web-ui to see rising/falling flank in the ui
            # but requires a frequent messaging, could be optimized)
            client.publish('person',
                           payload=json.dumps({
                               'count': person_counter, 'total': person_counter_total})
                          )
            # Send the durations
            if last_duration_seconds is not None:
                client.publish('person/duration',
                               payload=json.dumps({'duration': last_duration_seconds}),
                               )

            # Detection counter alert
            if person_counter > args.maximum_detections:
                txt2 = "Alert! Maximum count of {} detections reached!".format(args.maximum_detections)
                labelSize = cv2.getTextSize(txt2, FONT, 0.5, thickness=1)
                box_coords = ((10, frame.shape[0] - 30 + 2), 
                              (10 + labelSize[0][0], frame.shape[0] - 30 - labelSize[0][1] - 2))
                cv2.rectangle(frame, box_coords[0], box_coords[1], (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, txt2, (10, frame.shape[0] - 30), FONT, 0.5, (0, 0, 0), 1)
                
           # Detection time alert (Hint: of course this only detects if persons are longer that a certain time 
           # in the frame, there is NO detection of how long each person is in the frame)
            if duration_seconds > args.maximum_time:
                txt2 = "Alert! Maximum time of {}s attendance was reached!".format(args.maximum_time)
                labelSize = cv2.getTextSize(txt2, FONT, 0.5, thickness=1)
                box_coords = ((10, frame.shape[0] - 10 + 2), 
                              (10 + labelSize[0][0], frame.shape[0] - 10 - labelSize[0][1] - 2))
                cv2.rectangle(frame, box_coords[0], box_coords[1], (0, 215, 255), cv2.FILLED)
                cv2.putText(frame, txt2, (10, frame.shape[0] - 10), FONT, 0.5, (0, 0, 0), 1)

           
        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

        ### TODO: Write an output image if `single_image_mode` ###
        if single_image_mode:
            cv2.imwrite('out_image.jpg', frame)

        # Break loop if key pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
            
def draw_boxes(frame, results, prob_threshold):
    """
    Draw a box on every object with a probabilty over the given threshold and add a label with the probability.
    Only works for (1, 1, N, 7)-ish network outputs!
    :param frame: the playback frame
    :param results: the detection results from the network in shape (1, 1, N, 7)
    :param prob_threshold: the minimal confidence for a valid detection
    :return frame: the decorated frame
    :return current_box_count: the current count of detected boxes in the frame, as integer
    """
    # Check the model output blob shape
    output_dims = results.shape
    if (len(output_dims) != 4)  or (output_dims[3] != 7):
        log.error("Incorrect output dimensions: {}".format(output_dims))
        sys.exit(1)  
        
    # Look up frame shapes    
    width = frame.shape[1]
    height = frame.shape[0]
    
    # Draw boxes with enough confidence
    current_box_count = 0
    probs = results[0, 0, :, 2]
    for i, p in enumerate(probs):
        if p > prob_threshold:
            # Increase counter
            current_box_count += 1
            
            # Draw the box
            box = results[0, 0, i, 3:]
            xmin = int(box[0] * width)
            ymin = int(box[1] * height)
            xmax = int(box[2] * width)
            ymax = int(box[3] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
            
            # Add confidence label
            conf_message = "conf:{:.1f}".format(p)
            labelSize = cv2.getTextSize(conf_message, FONT, 0.5, 2)
            _x1 = xmax
            _y1 = ymin + int(labelSize[0][1])
            _x2 = xmax + labelSize[0][0]
            _y2 = ymin
            cv2.rectangle(frame, (_x1, _y1), (_x2, _y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, conf_message, (xmax, ymin + 10), FONT, 0.5, (0, 0, 0), 1)

    return frame, int(current_box_count)


def decorate_meta_to_frame(frame, inference_duration, person_count_current, persons_count_total, duration_current):
    """
    Decorates the given frame with the given meta and returns the decorated frame
    :param frame: the original frame
    :param inference_duration: the inference duration
    :param person_count_current: the current count in frame
    :param persons_count_total: the total count counted until the frame
    :param duration_current: current duration of a person in the frame
    :return frame: the decorated frame
    """
    # Tag the inference time 
    message_time = "Inference time: {:.1f}ms".format(inference_duration * 1000)
    labelSize = cv2.getTextSize(message_time, FONT, 0.5, 2)
    cv2.rectangle(frame, (15, 20), (15 + labelSize[0][0], 15 - labelSize[0][1]), (0,255,0), cv2.FILLED)
    cv2.putText(frame, message_time, (15, 15), FONT, 0.5, (0, 0, 0), 1)

    # Tag the current count in frame
    message_current = "Count(current): {}".format(person_count_current)
    labelSize = cv2.getTextSize(message_current, FONT, 0.5, 2)
    cv2.rectangle(frame, (15, 35), (15 + labelSize[0][0], 30 - labelSize[0][1]), (0,255,0), cv2.FILLED)
    cv2.putText(frame, message_current, (15, 30), FONT, 0.5, (0, 0, 0), 1)

    # Tag the total count
    message_total = "Count(total) {}".format(persons_count_total)
    labelSize = cv2.getTextSize(message_total, FONT, 0.5, 2)
    cv2.rectangle(frame, (15, 50), (15 + labelSize[0][0], 45 - labelSize[0][1]), (0,255,0), cv2.FILLED)
    cv2.putText(frame, message_total, (15, 47), FONT, 0.5, (0, 0, 0), 1)
    
    # Tag the total time detected, fix running duration for "zero persons"
    if person_count_current == 0:
        duration_current = 0
    message_duration = "Duration(in frame) {}s".format(duration_current)
    labelSize = cv2.getTextSize(message_duration, FONT, 0.5, 2)
    cv2.rectangle(frame, (15, 65), (15 + labelSize[0][0], 60 - labelSize[0][1]), (0,255,0), cv2.FILLED)
    cv2.putText(frame, message_duration, (15, 63), FONT, 0.5, (0, 0, 0), 1)
    
    return frame


def main():
    """
    Load the network and parse the output.
    :return: None
    """
    # Set log level to log some info in console
    log.basicConfig(level=log.INFO)
    
    # Grab command line args
    args = build_argparser().parse_args()
    
    # Connect to the MQTT server
    client = connect_mqtt()
    
    # Perform inference on the input stream
    infer_on_stream(args, client)
    log.info("Playback finished!")
    
    # Disconnect the MQTT
    client.disconnect()
    
    # Log end
    log.info("[ SUCCESS ] Application finished!")
    

if __name__ == '__main__':
    main()
    