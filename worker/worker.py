# worker/worker.py

import grpc
from concurrent import futures
import time
import torch
import torch.nn as nn
import argparse
import threading

from distributed_training_pb2 import (
    RegisterRequest,
    ForwardRequest,
    ForwardResponse,
    BackwardRequest,
    BackwardResponse
)
import distributed_training_pb2_grpc as distributed_training_pb2_grpc
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.utils import (
    serialize_tensor,
    deserialize_tensor,
    setup_logger,
    get_device
)

class TrainingServiceServicer(distributed_training_pb2_grpc.TrainingServiceServicer):
    def __init__(self, layer_index, logger):
        self.layer_index = layer_index
        self.layer = self.initialize_layer()
        self.logger = logger
        self.device = get_device()
        self.layer.to(self.device)
        self.optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.01)
        self.lock = threading.Lock()
        self.last_input = None  # To store input from ForwardPass
        self.logger.info(f"Worker assigned to layer {self.layer_index} initialized.")

    def initialize_layer(self):
        """
        Initialize the layer based on the assigned layer_index.
        """
        if self.layer_index == 0:
            return nn.Linear(784, 256)
        elif self.layer_index == 1:
            return nn.Linear(256, 10)
        else:
            raise ValueError("Unsupported layer index.")

    def ForwardPass(self, request, context):
        """
        Handle forward pass request from the server.
        """
        try:
            # Deserialize the input tensor
            input_tensor = deserialize_tensor(request.input, device=self.device)
            self.logger.info(f"Received ForwardPass with input shape {input_tensor.shape}")

            with torch.no_grad():
                # Forward pass through the layer
                output_tensor = self.layer(input_tensor)
                if isinstance(self.layer, nn.ReLU):
                    output_tensor = torch.relu(output_tensor)

            self.logger.info(f"ForwardPass: Processed input through layer {self.layer_index}, output shape {output_tensor.shape}")

            # Store input for backward pass
            self.last_input = input_tensor.clone()
            self.logger.info(f"Stored input for backward pass: {self.last_input.shape}")

            # Serialize the output tensor
            serialized_output = serialize_tensor(output_tensor)

            return ForwardResponse(output=serialized_output)
        except Exception as e:
            self.logger.error(f"Error in ForwardPass: {e}")
            # Return an empty output to indicate failure
            return ForwardResponse(output=b'')

    def BackwardPass(self, request, context):
        """
        Handle backward pass request from the server.
        """
        try:
            # Deserialize the incoming grad_output tensor
            grad_output = deserialize_tensor(request.grad_output, device=self.device)
            self.logger.info(f"Received BackwardPass with grad_output shape {grad_output.shape}")

            with self.lock:
                if self.last_input is None:
                    self.logger.error("No input stored from ForwardPass to compute gradients.")
                    return BackwardResponse(grad_input=b'')

                # Compute gradients
                # For nn.Linear, dL/dW = grad_output^T * input
                # dL/db = sum over batch of grad_output
                dL_dW = torch.matmul(grad_output.t(), self.last_input)  # Shape: [10, 256]
                dL_db = torch.sum(grad_output, dim=0)  # Shape: [10]

                # Assign gradients to layer
                self.layer.weight.grad = dL_dW
                self.layer.bias.grad = dL_db

                self.logger.info(f"Assigned dL/dW with shape {dL_dW.shape} and dL/db with shape {dL_db.shape} to layer {self.layer_index}.")

                # Update parameters
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.logger.info(f"Updated parameters for layer {self.layer_index}.")

                # Compute grad_input to send back to the server
                # grad_input = grad_output @ weight
                grad_input = torch.matmul(grad_output, self.layer.weight.data)  # Shape: [64, 256]
                self.logger.info(f"Computed grad_input for layer {self.layer_index}: {grad_input.shape}")

                # Serialize the grad_input tensor
                serialized_grad_input = serialize_tensor(grad_input)
                self.logger.info(f"Serialized grad_input for layer {self.layer_index}, size: {len(serialized_grad_input)} bytes.")

                return BackwardResponse(grad_input=serialized_grad_input)
        except Exception as e:
            self.logger.error(f"Error in BackwardPass: {e}")
            # Return an empty grad_input to indicate failure
            return BackwardResponse(grad_input=b'')

def serve(worker_address, server_address, logger):
    # Register the worker with the server
    with grpc.insecure_channel(server_address) as channel:
        stub = distributed_training_pb2_grpc.TrainingServiceStub(channel)
        register_request = RegisterRequest(
            worker_name=worker_address,
            worker_address=worker_address
        )
        response = stub.RegisterWorker(register_request)
        if response.success:
            worker_id = response.worker_id
            layer_index = worker_id - 1
            logger.info(f"Successfully registered with Worker ID {worker_id}, assigned to layer {layer_index}")
        else:
            logger.error(f"Registration failed: {response.message}")
            return

    # Start gRPC server to handle ForwardPass and BackwardPass
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    servicer = TrainingServiceServicer(layer_index=layer_index, logger=logger)
    distributed_training_pb2_grpc.add_TrainingServiceServicer_to_server(servicer, server)
    server.add_insecure_port(worker_address)
    server.start()
    logger.info(f"Worker gRPC server started at {worker_address}")

    try:
        while True:
            time.sleep(86400)  # Keep worker alive
    except KeyboardInterrupt:
        server.stop(0)
        logger.info("Worker server stopped.")

def main():
    parser = argparse.ArgumentParser(description='Distributed Training Worker')
    parser.add_argument('--address', type=str, required=True, help='Worker address (e.g., localhost:50052)')
    parser.add_argument('--server', type=str, default='localhost:50051', help='Server address (e.g., localhost:50051)')
    parser.add_argument('--log', type=str, default='worker.log', help='Log file path')
    args = parser.parse_args()

    # Setup logger
    logger = setup_logger('worker_logger', args.log, to_console=True)

    # Serve
    serve(worker_address=args.address, server_address=args.server, logger=logger)

if __name__ == '__main__':
    main()
