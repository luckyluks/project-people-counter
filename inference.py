#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        """
        Initialize Network object variables with None
        """
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.num_requests = None

    def load_model(self, model_xml, device="CPU", cpu_extension=None, num_requests=4):
        """
        Load the model given IR files on the device and add extensions
        :param model_xml: the path to the model XML file
        :param device: the desired device for the inference, CPU is default
        :param cpu_extension: the cpu extension file, if CPU is selected as device
        :param num_requests: the number of requests, that can be handled asynchronously 
        """
        ### TODO: Load the model ###
        ### TODO: Check for supported layers ###
        ### TODO: Add any necessary extensions ###
        ### TODO: Return the loaded inference plugin ###
        ### Note: You may need to update the function parameters. ###
        
        # Create model binary file path and check existence
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        if not os.path.exists(model_bin):
            sys.exit("ERROR: path \"{}\" is not a valid file path for the model binaries!".format(model_bin))

        # Initialize the plugin
        self.plugin = IECore()
        self.num_requests = num_requests

        # Add a CPU extension, if applicable
        if cpu_extension and "CPU" in device:
            self.plugin.add_extension(cpu_extension, device)

        # Read the IR as a IENetwork
        self.network = IENetwork(model=model_xml, weights=model_bin)
        
        # Get the supported layers of the network
        supported_layers = self.plugin.query_network(network=self.network, device_name=device)

        # Check for any unsupported layers. Exit the program, if so.
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            log.warn("Unsupported layers found: {}".format(unsupported_layers))
            log.warn("Check whether extensions are available to add to IECore.")
            sys.exit("ERROR: unsupported layers found!")

        # Load the IENetwork into the plugin
        self.exec_network = self.plugin.load_network(self.network, device, num_requests=num_requests)

        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        
        log.info("IR model \"{}\" succesfully loaded!".format(os.path.basename(model_xml)))
        
        return

    
    def get_input_shape(self):
        """
        Returns the input shape of the network
        :return: shape of the input
        """
        ### TODO: Return the shape of the input layer ###
        return self.network.inputs[self.input_blob].shape
    
    def get_output_shape(self):
        """
        Returns the output shape of the network
        :return: shape of the output
        """
        ### TODO: Return the shape of the input layer ###
        return self.network.outputs[self.output_blob].shape

    def exec_net(self, image, request_id):
        """
        Starts an asynchronous inference request with the given request id
        :param image: the input image
        :param request_id: the request id to use for the request
        :return request_id: the used request id
        :return next_request_id: a follow up request id
        """
        ### TODO: Start an asynchronous request ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        self.exec_network.start_async(request_id=request_id, inputs={self.input_blob: image})
        
        # Generate next request id (must be in range of num_requests, otherwise will be an "incorrect id")
        next_request_id = (request_id + 1) % self.num_requests
        return request_id, next_request_id
    

    def wait(self, request_id):
        """
        Waits until the inference request for the given request_id is finished
        :param request_id: the request id that is given in the request
        :return: the status of the processed request
        """
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        status = self.exec_network.requests[request_id].wait(-1)
        return status
        
        
    def get_output(self, request_id):
        """
        Returns the extracted output for the given request_id
        :param request_id: the request id that is given in the request
        :return: the extracted output for the given request_id
        """
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        return self.exec_network.requests[request_id].outputs[self.output_blob]