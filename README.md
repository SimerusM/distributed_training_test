# Distributed MNIST Training System

This project demonstrates a distributed machine learning framework for training a neural network on the MNIST dataset. It uses **gRPC** for communication between a central server and worker nodes, and **PyTorch** for training the model. The central server orchestrates the training process, while the workers perform computations for different layers of the model.

## How It Works
- **Central Server**: Coordinates training by sending data to workers for forward and backward passes and computes the loss.
- **Worker Nodes**: Each worker handles computations for a specific layer of the model.
- **Communication**: Uses gRPC to send and receive serialized tensors and gradients between the server and workers.

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- `pip` (Python package manager)
- A virtual environment (recommended)

### Installation
1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>

2. Virtual Env
```python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate```

3. Dependencies
`pip install -r requirements.txt`


### Running the System

1. Start server
python server.py



### Flow Diagram
Assuming we have 1-n workers/layers.

[Server]
    |
    | ForwardPass (inputs [64, 784])
    v
[Worker 1 (Layer 0)]
    | ForwardPass (output [64, 256])
    v
[Server]
    |
    | ForwardPass (current_input [64, 256])
    v
[Worker n (Layer n)]
    | ForwardPass (output [64, 10])
    v
[Server]
    |
    | Compute Loss (logits [64, 10])
    | Compute Grad Output (grad_output [64, 10])
    |
    | BackwardPass (grad_output [64, 10])
    v
[Worker n (Layer n)]
    | BackwardPass (grad_input [64, 256])
    v
[Server]
    |
    | BackwardPass (grad_output [64, 256])
    v
[Worker 1 (Layer 0)]
    | BackwardPass (grad_input [64, 784])
    v
[Server]
    |
    | Epoch Completed
