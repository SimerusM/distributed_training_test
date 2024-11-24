# server/server.py

import grpc
from concurrent import futures
import time
import threading
import torch
import torch.nn as nn
import torch.optim as optim

from distributed_training_pb2 import RegisterResponse, ForwardResponse, BackwardResponse, ForwardRequest, BackwardRequest
import distributed_training_pb2_grpc as distributed_training_pb2_grpc

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.utils import serialize_tensor, deserialize_tensor, setup_logger, get_device

from flask import Flask, jsonify, request as flask_request

# Initialize logger
logger = setup_logger('server_logger', 'server.log', to_console=True)

app = Flask(__name__)
training_in_progress = False

# Declare servicer as a global variable
servicer = None

class TrainingServiceServicer(distributed_training_pb2_grpc.TrainingServiceServicer):
    def __init__(self):
        self.workers = []  # List of workers in queue
        self.lock = threading.Lock()
        self.layer_assignments = {}  # worker_id: layer_index
        self.model_layers = []  # List of layers
        self.optimizer = None
        self.initialize_model()
        logger.info("TrainingServiceServicer initialized.")

    def initialize_model(self):
        """
        Initialize a simple model with a predefined number of layers.
        For simplicity, we use a model with two layers.
        """
        self.model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        self.model_layers = [
            self.model[0],  # Layer 0: Linear(784, 256)
            self.model[2]   # Layer 1: Linear(256, 10)
        ]
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        logger.info("Model initialized with two layers.")

    def RegisterWorker(self, request, context):
        with self.lock:
            worker_id = len(self.workers) + 1
            worker_info = {
                'worker_id': worker_id,
                'name': request.worker_name,
                'address': request.worker_address
            }
            self.workers.append(worker_info)
            layer_index = worker_id - 1  # Assign layers sequentially
            if layer_index < len(self.model_layers):
                self.layer_assignments[worker_id] = layer_index
                logger.info(f"Registered Worker {worker_id}: {request.worker_name} at {request.worker_address}, assigned to layer {layer_index}")
                return RegisterResponse(
                    success=True,
                    message=f"Worker {worker_id} registered and assigned to layer {layer_index}.",
                    worker_id=worker_id
                )
            else:
                # No more layers to assign
                self.workers.pop()
                logger.warning(f"Registration failed for {request.worker_name}: No available layers to assign.")
                return RegisterResponse(
                    success=False,
                    message="No available layers to assign.",
                    worker_id=0
                )

    def ForwardPass(self, request, context):
        """
        Handle forward pass from a worker.
        """
        worker_id = request.worker_id
        input_tensor = deserialize_tensor(request.input, device='cpu')
        logger.info(f"Received ForwardPass from Worker {worker_id} with input shape {input_tensor.shape}")

        with self.lock:
            if worker_id not in self.layer_assignments:
                logger.error(f"Worker {worker_id} not assigned to any layer.")
                return ForwardResponse(output=b'')

            layer_index = self.layer_assignments[worker_id]
            layer = self.model_layers[layer_index]

            # Forward pass through the assigned layer
            with torch.no_grad():
                output_tensor = layer(input_tensor)
                if isinstance(layer, nn.ReLU):
                    output_tensor = torch.relu(output_tensor)

            logger.info(f"ForwardPass: Applied layer {layer_index} for Worker {worker_id}, output shape {output_tensor.shape}")

            serialized_output = serialize_tensor(output_tensor)

            return ForwardResponse(output=serialized_output)

    def BackwardPass(self, request, context):
        """
        Handle backward pass from a worker.
        """
        worker_id = request.worker_id
        grad_output = deserialize_tensor(request.grad_output, device='cpu')
        logger.info(f"Received BackwardPass from Worker {worker_id} with grad_output shape {grad_output.shape}")

        with self.lock:
            if worker_id not in self.layer_assignments:
                logger.error(f"Worker {worker_id} not assigned to any layer.")
                return BackwardResponse(grad_input=b'')

            layer_index = self.layer_assignments[worker_id]
            layer = self.model_layers[layer_index]

            # Perform backward pass manually
            layer.weight.grad = grad_output
            layer.bias.grad = grad_output.mean(dim=0)  # Simplified gradient for bias

            # Update model parameters
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Compute grad_input to send back to the previous layer
            if layer_index == 0:
                # No previous layer
                grad_input = torch.matmul(grad_output, layer.weight.data)
            else:
                prev_layer = self.model_layers[layer_index - 1]
                grad_input = torch.matmul(grad_output, layer.weight.data)

            logger.info(f"BackwardPass: Updated layer {layer_index} for Worker {worker_id}, grad_input shape {grad_input.shape}")

            serialized_grad_input = serialize_tensor(grad_input)

            return BackwardResponse(grad_input=serialized_grad_input)

    # server/server.py
    def run_training(self):
        global training_in_progress
        with self.lock:
            if training_in_progress:
                logger.warning("Training is already in progress.")
                return
            training_in_progress = True

        logger.info("Training started.")
        num_epochs = 1  # For simplicity, use 1 epoch
        batch_size = 64

        # Generate dummy data for simplicity
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1} started.")
            # Create dummy input and labels
            inputs = torch.randn(batch_size, 784)  # Batch of 64 samples, 784 features
            labels = torch.randint(0, 10, (batch_size,))  # Batch of 64 labels

            # Forward Pass
            current_input = inputs
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
                        stub = distributed_training_pb2_grpc.TrainingServiceStub(channel)
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
            loss = loss_fn(logits, labels)
            logger.info(f"Computed loss: {loss.item()}")

            # Manual Gradient Computation (dL/dlogits)
            with torch.no_grad():
                softmax = torch.softmax(logits, dim=1)
                one_hot = torch.zeros_like(logits)
                one_hot.scatter_(1, labels.view(-1, 1), 1)
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
                        stub = distributed_training_pb2_grpc.TrainingServiceStub(channel)
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

            logger.info(f"Epoch {epoch+1} completed.")

        logger.info("Training finished.")
        training_in_progress = False



@app.route('/run_train', methods=['POST'])
def run_train():
    """
    API endpoint to start training.
    """
    global servicer

    print(servicer)
    if not servicer:
        return jsonify({"status": "Error."}), 400

    threading.Thread(target=servicer.run_training).start()
    return jsonify({"status": "Training started."}), 200

def serve():
    global servicer  # Declare that we're modifying the global servicer
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = TrainingServiceServicer()
    distributed_training_pb2_grpc.add_TrainingServiceServicer_to_server(
        servicer, server)
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


# # server/server.py

# import grpc
# from concurrent import futures
# import time
# import threading
# import torch
# import torch.nn as nn
# import torch.optim as optim

# from common.distributed_training_pb2 import RegisterResponse, ForwardResponse, BackwardResponse, ForwardRequest, BackwardRequest
# import common.distributed_training_pb2_grpc as distributed_training_pb2_grpc
# from common.utils import serialize_tensor, deserialize_tensor, setup_logger, get_device

# from flask import Flask, jsonify, request as flask_request

# # Initialize logger
# logger = setup_logger('server_logger', 'server.log', to_console=True)

# app = Flask(__name__)
# training_in_progress = False

# # Declare servicer as a global variable
# servicer = None

# class TrainingServiceServicer(distributed_training_pb2_grpc.TrainingServiceServicer):
#     def __init__(self):
#         self.workers = []  # List of workers in queue
#         self.lock = threading.Lock()
#         self.layer_assignments = {}  # worker_id: layer_index
#         self.model_layers = []  # List of layers
#         self.optimizer = None
#         self.initialize_model()
#         logger.info("TrainingServiceServicer initialized.")

#     def initialize_model(self):
#         """
#         Initialize a simple model with a predefined number of layers.
#         For simplicity, we use a model with two layers.
#         """
#         self.model = nn.Sequential(
#             nn.Linear(784, 256),
#             nn.ReLU(),
#             nn.Linear(256, 10)
#         )
#         self.model_layers = [
#             self.model[0],  # Layer 0: Linear(784, 256)
#             self.model[2]   # Layer 1: Linear(256, 10)
#         ]
#         self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
#         logger.info("Model initialized with two layers.")

#     def RegisterWorker(self, request, context):
#         with self.lock:
#             worker_id = len(self.workers) + 1
#             worker_info = {
#                 'worker_id': worker_id,
#                 'name': request.worker_name,
#                 'address': request.worker_address
#             }
#             self.workers.append(worker_info)
#             layer_index = worker_id - 1  # Assign layers sequentially
#             if layer_index < len(self.model_layers):
#                 self.layer_assignments[worker_id] = layer_index
#                 logger.info(f"Registered Worker {worker_id}: {request.worker_name} at {request.worker_address}, assigned to layer {layer_index}")
#                 return RegisterResponse(
#                     success=True,
#                     message=f"Worker {worker_id} registered and assigned to layer {layer_index}.",
#                     worker_id=worker_id
#                 )
#             else:
#                 # No more layers to assign
#                 self.workers.pop()
#                 logger.warning(f"Registration failed for {request.worker_name}: No available layers to assign.")
#                 return RegisterResponse(
#                     success=False,
#                     message="No available layers to assign.",
#                     worker_id=0
#                 )

#     def ForwardPass(self, request, context):
#         """
#         Handle forward pass from a worker.
#         """
#         worker_id = request.worker_id
#         input_tensor = deserialize_tensor(request.input, device='cpu')
#         logger.info(f"Received ForwardPass from Worker {worker_id} with input shape {input_tensor.shape}")

#         with self.lock:
#             if worker_id not in self.layer_assignments:
#                 logger.error(f"Worker {worker_id} not assigned to any layer.")
#                 return ForwardResponse(output=b'')

#             layer_index = self.layer_assignments[worker_id]
#             layer = self.model_layers[layer_index]

#             # Forward pass through the assigned layer
#             with torch.no_grad():
#                 output_tensor = layer(input_tensor)
#                 if isinstance(layer, nn.ReLU):
#                     output_tensor = torch.relu(output_tensor)

#             logger.info(f"ForwardPass: Applied layer {layer_index} for Worker {worker_id}, output shape {output_tensor.shape}")

#             serialized_output = serialize_tensor(output_tensor)

#             return ForwardResponse(output=serialized_output)

#     def BackwardPass(self, request, context):
#         """
#         Handle backward pass from a worker.
#         """
#         worker_id = request.worker_id
#         grad_output = deserialize_tensor(request.grad_output, device='cpu')
#         logger.info(f"Received BackwardPass from Worker {worker_id} with grad_output shape {grad_output.shape}")

#         with self.lock:
#             if worker_id not in self.layer_assignments:
#                 logger.error(f"Worker {worker_id} not assigned to any layer.")
#                 return BackwardResponse(grad_input=b'')

#             layer_index = self.layer_assignments[worker_id]
#             layer = self.model_layers[layer_index]

#             # Perform backward pass manually
#             layer.weight.grad = grad_output
#             layer.bias.grad = grad_output.mean(dim=0)  # Simplified gradient for bias

#             # Update model parameters
#             self.optimizer.step()
#             self.optimizer.zero_grad()

#             # Compute grad_input to send back to the previous layer
#             if layer_index == 0:
#                 # No previous layer
#                 grad_input = torch.matmul(grad_output, layer.weight.data)
#             else:
#                 prev_layer = self.model_layers[layer_index - 1]
#                 grad_input = torch.matmul(grad_output, layer.weight.data)

#             logger.info(f"BackwardPass: Updated layer {layer_index} for Worker {worker_id}, grad_input shape {grad_input.shape}")

#             serialized_grad_input = serialize_tensor(grad_input)

#             return BackwardResponse(grad_input=serialized_grad_input)

#     def run_training(self):
#         global training_in_progress
#         with self.lock:
#             if training_in_progress:
#                 logger.warning("Training is already in progress.")
#                 return
#             training_in_progress = True

#         logger.info("Training started.")
#         num_epochs = 1  # For simplicity, use 1 epoch
#         batch_size = 64

#         # Generate dummy data for simplicity
#         for epoch in range(num_epochs):
#             logger.info(f"Epoch {epoch+1} started.")
#             # Create dummy input and labels
#             inputs = torch.randn(batch_size, 784)  # Batch of 64 samples, 784 features
#             labels = torch.randint(0, 10, (batch_size,))  # Batch of 64 labels

#             # Forward Pass
#             current_input = inputs
#             for worker in self.workers:
#                 worker_id = worker['worker_id']
#                 logger.info(f"Sending ForwardPass to Worker {worker_id}")
#                 serialized_input = serialize_tensor(current_input)
#                 forward_request = ForwardRequest(
#                     worker_id=worker_id,
#                     input=serialized_input
#                 )
#                 try:
#                     with grpc.insecure_channel(worker['address']) as channel:
#                         stub = distributed_training_pb2_grpc.TrainingServiceStub(channel)
#                         response = stub.ForwardPass(forward_request)
#                         current_input = deserialize_tensor(response.output, device='cpu')
#                         logger.info(f"Received ForwardPass response from Worker {worker_id}, new input shape {current_input.shape}")
#                 except Exception as e:
#                     logger.error(f"Error during ForwardPass with Worker {worker_id}: {e}")
#                     training_in_progress = False
#                     return

#             # Compute loss at the last layer
#             loss_fn = nn.CrossEntropyLoss()
#             logits = current_input  # Assuming last layer outputs logits
#             loss = loss_fn(logits, labels)
#             logger.info(f"Computed loss: {loss.item()}")

#             # Backward Pass
#             grad_output = torch.ones_like(logits) * loss.grad_fn(loss).next_functions[0][0].variable.data
#             for worker in reversed(self.workers):
#                 worker_id = worker['worker_id']
#                 logger.info(f"Sending BackwardPass to Worker {worker_id}")
#                 serialized_grad_output = serialize_tensor(grad_output)
#                 backward_request = BackwardRequest(
#                     worker_id=worker_id,
#                     grad_output=serialized_grad_output
#                 )
#                 try:
#                     with grpc.insecure_channel(worker['address']) as channel:
#                         stub = distributed_training_pb2_grpc.TrainingServiceStub(channel)
#                         response = stub.BackwardPass(backward_request)
#                         grad_input = deserialize_tensor(response.grad_input, device='cpu')
#                         grad_output = grad_input  # Pass to the previous layer
#                         logger.info(f"Received BackwardPass response from Worker {worker_id}, new grad_input shape {grad_input.shape}")
#                 except Exception as e:
#                     logger.error(f"Error during BackwardPass with Worker {worker_id}: {e}")
#                     training_in_progress = False
#                     return

#             logger.info(f"Epoch {epoch+1} completed.")

#         logger.info("Training finished.")
#         training_in_progress = False

# @app.route('/run_train', methods=['POST'])
# def run_train():
#     """
#     API endpoint to start training.
#     """
#     global servicer

#     print(servicer)
#     if not servicer:
#         return jsonify({"status": "Error."}), 400

#     threading.Thread(target=servicer.run_training).start()
#     return jsonify({"status": "Training started."}), 200

# def serve():
#     global servicer  # Declare that we're modifying the global servicer
#     server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
#     servicer = TrainingServiceServicer()
#     distributed_training_pb2_grpc.add_TrainingServiceServicer_to_server(
#         servicer, server)
#     server.add_insecure_port('[::]:50051')
#     server.start()
#     logger.info("Central Server started on port 50051.")

#     # Start Flask app in a separate thread
#     flask_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=8000))
#     flask_thread.start()
#     logger.info("Flask app started on port 8000.")

#     try:
#         while True:
#             time.sleep(86400)  # Keep server alive
#     except KeyboardInterrupt:
#         server.stop(0)
#         logger.info("Central Server stopped.")

# if __name__ == '__main__':
#     serve()


# # # server/server.py

# # import grpc
# # from concurrent import futures
# # import time
# # import threading
# # import torch
# # import torch.nn as nn
# # import torch.optim as optim

# # import common.distributed_training_pb2 as distributed_training_pb2
# # from common.distributed_training_pb2 import RegisterResponse, ForwardResponse, BackwardResponse
# # import common.distributed_training_pb2_grpc as distributed_training_pb2_grpc
# # from common.utils import serialize_tensor, deserialize_tensor, setup_logger, get_device

# # from flask import Flask, jsonify
# # from flask import request as flask_request

# # # Initialize logger
# # logger = setup_logger('server_logger', 'server.log', to_console=True)
# # servicer = None

# # app = Flask(__name__)
# # training_in_progress = False

# # class TrainingServiceServicer(distributed_training_pb2_grpc.TrainingServiceServicer):
# #     def __init__(self):
# #         self.workers = []  # List of workers in queue
# #         self.lock = threading.Lock()
# #         self.layer_assignments = {}  # worker_id: layer_index
# #         self.model_layers = []  # List of layers
# #         self.optimizer = None
# #         self.initialize_model()
# #         logger.info("TrainingServiceServicer initialized.")

# #     def initialize_model(self):
# #         """
# #         Initialize a simple model with a predefined number of layers.
# #         For simplicity, we use a model with two layers.
# #         """
# #         self.model = nn.Sequential(
# #             nn.Linear(784, 256),
# #             nn.ReLU(),
# #             nn.Linear(256, 10)
# #         )
# #         self.model_layers = [
# #             self.model[0],  # Layer 0: Linear(784, 256)
# #             self.model[2]   # Layer 1: Linear(256, 10)
# #         ]
# #         self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
# #         logger.info("Model initialized with two layers.")

# #     def RegisterWorker(self, request, context):
# #         with self.lock:
# #             worker_id = len(self.workers) + 1
# #             worker_info = {
# #                 'worker_id': worker_id,
# #                 'name': request.worker_name,
# #                 'address': request.worker_address
# #             }
# #             self.workers.append(worker_info)
# #             layer_index = worker_id - 1  # Assign layers sequentially
# #             if layer_index < len(self.model_layers):
# #                 self.layer_assignments[worker_id] = layer_index
# #                 logger.info(f"Registered Worker {worker_id}: {request.worker_name} at {request.worker_address}, assigned to layer {layer_index}")
# #                 return RegisterResponse(
# #                     success=True,
# #                     message=f"Worker {worker_id} registered and assigned to layer {layer_index}.",
# #                     worker_id=worker_id
# #                 )
# #             else:
# #                 # No more layers to assign
# #                 self.workers.pop()
# #                 logger.warning(f"Registration failed for {request.worker_name}: No available layers to assign.")
# #                 return RegisterResponse(
# #                     success=False,
# #                     message="No available layers to assign.",
# #                     worker_id=0
# #                 )

# #     def ForwardPass(self, request, context):
# #         """
# #         Handle forward pass from a worker.
# #         """
# #         worker_id = request.worker_id
# #         input_tensor = deserialize_tensor(request.input, device='cpu')
# #         logger.info(f"Received ForwardPass from Worker {worker_id} with input shape {input_tensor.shape}")

# #         with self.lock:
# #             if worker_id not in self.layer_assignments:
# #                 logger.error(f"Worker {worker_id} not assigned to any layer.")
# #                 return ForwardResponse(output=b'')

# #             layer_index = self.layer_assignments[worker_id]
# #             layer = self.model_layers[layer_index]

# #             # Forward pass through the assigned layer
# #             with torch.no_grad():
# #                 output_tensor = layer(input_tensor)
# #                 if isinstance(layer, nn.ReLU):
# #                     output_tensor = torch.relu(output_tensor)

# #             logger.info(f"ForwardPass: Applied layer {layer_index} for Worker {worker_id}, output shape {output_tensor.shape}")

# #             serialized_output = serialize_tensor(output_tensor)

# #             return ForwardResponse(output=serialized_output)

# #     def BackwardPass(self, request, context):
# #         """
# #         Handle backward pass from a worker.
# #         """
# #         worker_id = request.worker_id
# #         grad_output = deserialize_tensor(request.grad_output, device='cpu')
# #         logger.info(f"Received BackwardPass from Worker {worker_id} with grad_output shape {grad_output.shape}")

# #         with self.lock:
# #             if worker_id not in self.layer_assignments:
# #                 logger.error(f"Worker {worker_id} not assigned to any layer.")
# #                 return BackwardResponse(grad_input=b'')

# #             layer_index = self.layer_assignments[worker_id]
# #             layer = self.model_layers[layer_index]

# #             # Perform backward pass manually
# #             layer.weight.grad = grad_output
# #             layer.bias.grad = grad_output.mean(dim=0)  # Simplified gradient for bias

# #             # Update model parameters
# #             self.optimizer.step()
# #             self.optimizer.zero_grad()

# #             # Compute grad_input to send back to the previous layer
# #             if layer_index == 0:
# #                 # No previous layer
# #                 grad_input = torch.matmul(grad_output, layer.weight.data)
# #             else:
# #                 prev_layer = self.model_layers[layer_index - 1]
# #                 grad_input = torch.matmul(grad_output, layer.weight.data)

# #             logger.info(f"BackwardPass: Updated layer {layer_index} for Worker {worker_id}, grad_input shape {grad_input.shape}")

# #             serialized_grad_input = serialize_tensor(grad_input)

# #             return BackwardResponse(grad_input=serialized_grad_input)

# #     def run_training(self):
# #         global training_in_progress
# #         with self.lock:
# #             if training_in_progress:
# #                 logger.warning("Training is already in progress.")
# #                 return
# #             training_in_progress = True

# #         logger.info("Training started.")
# #         num_epochs = 1  # For simplicity, use 1 epoch
# #         batch_size = 64

# #         # Generate dummy data for simplicity
# #         for epoch in range(num_epochs):
# #             logger.info(f"Epoch {epoch+1} started.")
# #             # Create dummy input and labels
# #             inputs = torch.randn(batch_size, 784)  # Batch of 64 samples, 784 features
# #             labels = torch.randint(0, 10, (batch_size,))  # Batch of 64 labels

# #             # Forward Pass
# #             current_input = inputs
# #             for worker in self.workers:
# #                 worker_id = worker['worker_id']
# #                 logger.info(f"Sending ForwardPass to Worker {worker_id}")
# #                 serialized_input = serialize_tensor(current_input)
# #                 forward_request = distributed_training_pb2.ForwardRequest(
# #                     worker_id=worker_id,
# #                     input=serialized_input
# #                 )
# #                 try:
# #                     with grpc.insecure_channel(worker['address']) as channel:
# #                         stub = distributed_training_pb2_grpc.TrainingServiceStub(channel)
# #                         response = stub.ForwardPass(forward_request)
# #                         current_input = deserialize_tensor(response.output, device='cpu')
# #                         logger.info(f"Received ForwardPass response from Worker {worker_id}, new input shape {current_input.shape}")
# #                 except Exception as e:
# #                     logger.error(f"Error during ForwardPass with Worker {worker_id}: {e}")
# #                     training_in_progress = False
# #                     return

# #             # Compute loss at the last layer
# #             loss_fn = nn.CrossEntropyLoss()
# #             logits = current_input  # Assuming last layer outputs logits
# #             loss = loss_fn(logits, labels)
# #             logger.info(f"Computed loss: {loss.item()}")

# #             # Backward Pass
# #             grad_output = torch.ones_like(logits) * loss.grad_fn(loss).next_functions[0][0].variable.data
# #             for worker in reversed(self.workers):
# #                 worker_id = worker['worker_id']
# #                 logger.info(f"Sending BackwardPass to Worker {worker_id}")
# #                 serialized_grad_output = serialize_tensor(grad_output)
# #                 backward_request = distributed_training_pb2.BackwardRequest(
# #                     worker_id=worker_id,
# #                     grad_output=serialized_grad_output
# #                 )
# #                 try:
# #                     with grpc.insecure_channel(worker['address']) as channel:
# #                         stub = distributed_training_pb2_grpc.TrainingServiceStub(channel)
# #                         response = stub.BackwardPass(backward_request)
# #                         grad_input = deserialize_tensor(response.grad_input, device='cpu')
# #                         grad_output = grad_input  # Pass to the previous layer
# #                         logger.info(f"Received BackwardPass response from Worker {worker_id}, new grad_input shape {grad_input.shape}")
# #                 except Exception as e:
# #                     logger.error(f"Error during BackwardPass with Worker {worker_id}: {e}")
# #                     training_in_progress = False
# #                     return

# #             logger.info(f"Epoch {epoch+1} completed.")

# #         logger.info("Training finished.")
# #         training_in_progress = False



# # @app.route('/run_train', methods=['POST'])
# # def run_train():
# #     """
# #     API endpoint to start training.
# #     """
# #     global servicer

# #     print(servicer)
# #     if not servicer:
# #         return jsonify({"status": "Error."}), 400
    
# #     threading.Thread(target=servicer.run_training).start()
# #     return jsonify({"status": "Training started."}), 200

# # def serve():
# #     global servicer
# #     server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
# #     servicer = TrainingServiceServicer()
# #     distributed_training_pb2_grpc.add_TrainingServiceServicer_to_server(
# #         servicer, server)
# #     server.add_insecure_port('[::]:50051')
# #     server.start()
# #     logger.info("Central Server started on port 50051.")

# #     # Start Flask app in a separate thread
# #     flask_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=8000))
# #     flask_thread.start()
# #     logger.info("Flask app started on port 8000.")

# #     try:
# #         while True:
# #             time.sleep(86400)  # Keep server alive
# #     except KeyboardInterrupt:
# #         server.stop(0)
# #         logger.info("Central Server stopped.")

# # if __name__ == '__main__':
# #     serve()

