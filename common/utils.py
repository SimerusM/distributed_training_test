# common/utils.py

import logging
import torch
import io

def setup_logger(name, log_file, level=logging.INFO, to_console=False):
    """
    Sets up a logger with the specified name and log file.

    Args:
        name (str): Name of the logger.
        log_file (str): Path to the log file.
        level (int): Logging level.
        to_console (bool): Whether to also log to console.

    Returns:
        logging.Logger: Configured logger.
    """
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s:%(name)s: %(message)s')

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # File handler
    file_handler = logging.FileHandler(log_file)        
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    if to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

def serialize_tensor(tensor):
    """
    Serializes a PyTorch tensor to bytes using torch.save.

    Args:
        tensor (torch.Tensor): Tensor to serialize.

    Returns:
        bytes: Serialized tensor.
    """
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return buffer.getvalue()

def deserialize_tensor(tensor_bytes, device='cpu'):
    """
    Deserializes bytes to a PyTorch tensor using torch.load.

    Args:
        tensor_bytes (bytes): Serialized tensor.
        device (str): Device to place the tensor on.

    Returns:
        torch.Tensor: Deserialized tensor.
    """
    buffer = io.BytesIO(tensor_bytes)
    tensor = torch.load(buffer, map_location=device)
    return tensor

def get_device():
    """
    Determines the available device.

    Returns:
        torch.device: Available device (GPU if available, else CPU).
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# # common/utils.py

# import logging
# import torch
# import io

# def setup_logger(name, log_file, level=logging.INFO, to_console=False):
#     """
#     Sets up a logger with the specified name and log file.

#     Args:
#         name (str): Name of the logger.
#         log_file (str): Path to the log file.
#         level (int): Logging level.
#         to_console (bool): Whether to also log to console.

#     Returns:
#         logging.Logger: Configured logger.
#     """
#     formatter = logging.Formatter('[%(asctime)s] %(levelname)s:%(name)s: %(message)s')

#     logger = logging.getLogger(name)
#     logger.setLevel(level)

#     # File handler
#     file_handler = logging.FileHandler(log_file)        
#     file_handler.setFormatter(formatter)
#     logger.addHandler(file_handler)

#     # Console handler
#     if to_console:
#         console_handler = logging.StreamHandler()
#         console_handler.setFormatter(formatter)
#         logger.addHandler(console_handler)

#     return logger

# def serialize_tensor(tensor):
#     """
#     Serializes a PyTorch tensor to bytes using torch.save.

#     Args:
#         tensor (torch.Tensor): Tensor to serialize.

#     Returns:
#         bytes: Serialized tensor.
#     """
#     buffer = io.BytesIO()
#     torch.save(tensor, buffer)
#     return buffer.getvalue()

# def deserialize_tensor(tensor_bytes, device='cpu'):
#     """
#     Deserializes bytes to a PyTorch tensor using torch.load.

#     Args:
#         tensor_bytes (bytes): Serialized tensor.
#         device (str): Device to place the tensor on.

#     Returns:
#         torch.Tensor: Deserialized tensor.
#     """
#     buffer = io.BytesIO(tensor_bytes)
#     tensor = torch.load(buffer, map_location=device)
#     return tensor

# def get_device():
#     """
#     Determines the available device.

#     Returns:
#         torch.device: Available device (GPU if available, else CPU).
#     """
#     return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
