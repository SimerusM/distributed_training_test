import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from concurrent import futures
import grpc
import threading
import time
from flask import Flask, jsonify, request
from distributed_training_pb2 import ForwardRequest, BackwardRequest, RegisterResponse
from distributed_training_pb2_grpc import TrainingServiceStub

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.utils import serialize_tensor, deserialize_tensor, setup_logger, get_device

import distributed_training_pb2_grpc
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server_logger")

training_in_progress = False
servicer = None

# Load MNIST dataset
def load_mnist_data(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader

class TrainingServiceServicer(distributed_training_pb2_grpc.TrainingServiceServicer):
    def __init__(self):
        self.workers = []
        self.lock = threading.Lock()

    def RegisterWorker(self, request, context):
        with self.lock:
            worker_id = len(self.workers) + 1
            self.workers.append({"worker_id": worker_id, "address": request.worker_address})
            logger.info(f"Registered Worker {worker_id}: {request.worker_address}, assigned to layer {worker_id - 1}")
        return RegisterResponse(
            success=True,
            message=f"Worker {worker_id} registered successfully.",
            worker_id=worker_id,
        )

    def run_training(self):
        """
        Main training loop to process MNIST data.
        """
        global training_in_progress
        with self.lock:
            if training_in_progress:
                logger.warning("Training is already in progress.")
                return
            training_in_progress = True

        logger.info("Training started.")
        num_epochs = 100
        batch_size = 256

        # Load MNIST data
        train_loader = load_mnist_data(batch_size)

        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs} started.")

            for batch_idx, (batch_inputs, batch_labels) in enumerate(train_loader):
                # Flatten the images for fully connected layers
                batch_inputs = batch_inputs.view(batch_inputs.size(0), -1)  # Shape: [batch_size, 784]

                # Forward Pass
                current_input = batch_inputs
                for worker in self.workers:
                    worker_id = worker['worker_id']
                    logger.info(f"Sending ForwardPass to Worker {worker_id}")
                    serialized_input = serialize_tensor(current_input)
                    forward_request = ForwardRequest(
                        worker_id=worker_id,
                        input=serialized_input
                    )
                    try:
                        with grpc.insecure_channel(worker['address']) as channel:
                            stub = TrainingServiceStub(channel)
                            response = stub.ForwardPass(forward_request)
                            if not response.output:
                                logger.error(f"Received empty ForwardPass response from Worker {worker_id}.")
                                training_in_progress = False
                                return
                            current_input = deserialize_tensor(response.output, device='cpu')
                            logger.info(f"Received ForwardPass response from Worker {worker_id}, new input shape {current_input.shape}")
                    except Exception as e:
                        logger.error(f"Error during ForwardPass with Worker {worker_id}: {e}")
                        training_in_progress = False
                        return

                # Compute loss at the last layer
                loss_fn = nn.CrossEntropyLoss()
                logits = current_input  # Assuming last layer outputs logits
                loss = loss_fn(logits, batch_labels)
                logger.info(f"Batch {batch_idx+1}: Computed loss: {loss.item()}")

                # Manual Gradient Computation (dL/dlogits)
                with torch.no_grad():
                    softmax = torch.softmax(logits, dim=1)
                    one_hot = torch.zeros_like(logits)
                    one_hot.scatter_(1, batch_labels.view(-1, 1), 1)
                    grad_output = (softmax - one_hot) / batch_size
                    logger.info(f"Computed grad_output shape: {grad_output.shape}")

                # Backward Pass
                for worker in reversed(self.workers):
                    worker_id = worker['worker_id']
                    logger.info(f"Sending BackwardPass to Worker {worker_id}")
                    serialized_grad_output = serialize_tensor(grad_output)
                    backward_request = BackwardRequest(
                        worker_id=worker_id,
                        grad_output=serialized_grad_output
                    )
                    try:
                        with grpc.insecure_channel(worker['address']) as channel:
                            stub = TrainingServiceStub(channel)
                            response = stub.BackwardPass(backward_request)
                            if not response.grad_input:
                                logger.error(f"Received empty grad_input from Worker {worker_id}.")
                                training_in_progress = False
                                return
                            grad_input = deserialize_tensor(response.grad_input, device='cpu')
                            logger.info(f"Received BackwardPass response from Worker {worker_id}, new grad_input shape {grad_input.shape}")
                            grad_output = grad_input  # Pass to the previous layer
                    except Exception as e:
                        logger.error(f"Error during BackwardPass with Worker {worker_id}: {e}")
                        training_in_progress = False
                        return

            logger.info(f"Epoch {epoch+1}/{num_epochs} completed.")

        logger.info("Training finished.")
        training_in_progress = False


@app.route('/run_train', methods=['POST'])
def run_train():
    """
    API endpoint to start training.
    """
    global servicer

    if not servicer:
        return jsonify({"status": "Error: TrainingServiceServicer not initialized."}), 400

    threading.Thread(target=servicer.run_training).start()
    return jsonify({"status": "Training started."}), 200


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    global servicer
    servicer = TrainingServiceServicer()
    distributed_training_pb2_grpc.add_TrainingServiceServicer_to_server(servicer, server)
    server.add_insecure_port('[::]:50051')
    server.start()
    logger.info("Central Server started on port 50051.")

    # Start Flask app in a separate thread
    flask_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=8000))
    flask_thread.start()
    logger.info("Flask app started on port 8000.")

    try:
        while True:
            time.sleep(86400)  # Keep server alive
    except KeyboardInterrupt:
        server.stop(0)
        logger.info("Central Server stopped.")


if __name__ == '__main__':
    serve()
