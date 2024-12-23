# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import warnings

import distributed_training_pb2 as distributed__training__pb2

GRPC_GENERATED_VERSION = '1.68.0'
GRPC_VERSION = grpc.__version__
_version_not_supported = False

try:
    from grpc._utilities import first_version_is_lower
    _version_not_supported = first_version_is_lower(GRPC_VERSION, GRPC_GENERATED_VERSION)
except ImportError:
    _version_not_supported = True

if _version_not_supported:
    raise RuntimeError(
        f'The grpc package installed is at version {GRPC_VERSION},'
        + f' but the generated code in distributed_training_pb2_grpc.py depends on'
        + f' grpcio>={GRPC_GENERATED_VERSION}.'
        + f' Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}'
        + f' or downgrade your generated code using grpcio-tools<={GRPC_VERSION}.'
    )


class TrainingServiceStub(object):
    """Service definition for the central server
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.RegisterWorker = channel.unary_unary(
                '/distributed_training.TrainingService/RegisterWorker',
                request_serializer=distributed__training__pb2.RegisterRequest.SerializeToString,
                response_deserializer=distributed__training__pb2.RegisterResponse.FromString,
                _registered_method=True)
        self.ForwardPass = channel.unary_unary(
                '/distributed_training.TrainingService/ForwardPass',
                request_serializer=distributed__training__pb2.ForwardRequest.SerializeToString,
                response_deserializer=distributed__training__pb2.ForwardResponse.FromString,
                _registered_method=True)
        self.BackwardPass = channel.unary_unary(
                '/distributed_training.TrainingService/BackwardPass',
                request_serializer=distributed__training__pb2.BackwardRequest.SerializeToString,
                response_deserializer=distributed__training__pb2.BackwardResponse.FromString,
                _registered_method=True)
        self.Heartbeat = channel.unary_unary(
                '/distributed_training.TrainingService/Heartbeat',
                request_serializer=distributed__training__pb2.HeartbeatRequest.SerializeToString,
                response_deserializer=distributed__training__pb2.HeartbeatResponse.FromString,
                _registered_method=True)


class TrainingServiceServicer(object):
    """Service definition for the central server
    """

    def RegisterWorker(self, request, context):
        """Register a worker node
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ForwardPass(self, request, context):
        """Send activation/output from one layer to the next
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def BackwardPass(self, request, context):
        """Send gradients from one layer to the previous
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Heartbeat(self, request, context):
        """Optional: Heartbeat for health checks
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_TrainingServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'RegisterWorker': grpc.unary_unary_rpc_method_handler(
                    servicer.RegisterWorker,
                    request_deserializer=distributed__training__pb2.RegisterRequest.FromString,
                    response_serializer=distributed__training__pb2.RegisterResponse.SerializeToString,
            ),
            'ForwardPass': grpc.unary_unary_rpc_method_handler(
                    servicer.ForwardPass,
                    request_deserializer=distributed__training__pb2.ForwardRequest.FromString,
                    response_serializer=distributed__training__pb2.ForwardResponse.SerializeToString,
            ),
            'BackwardPass': grpc.unary_unary_rpc_method_handler(
                    servicer.BackwardPass,
                    request_deserializer=distributed__training__pb2.BackwardRequest.FromString,
                    response_serializer=distributed__training__pb2.BackwardResponse.SerializeToString,
            ),
            'Heartbeat': grpc.unary_unary_rpc_method_handler(
                    servicer.Heartbeat,
                    request_deserializer=distributed__training__pb2.HeartbeatRequest.FromString,
                    response_serializer=distributed__training__pb2.HeartbeatResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'distributed_training.TrainingService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('distributed_training.TrainingService', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class TrainingService(object):
    """Service definition for the central server
    """

    @staticmethod
    def RegisterWorker(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/distributed_training.TrainingService/RegisterWorker',
            distributed__training__pb2.RegisterRequest.SerializeToString,
            distributed__training__pb2.RegisterResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def ForwardPass(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/distributed_training.TrainingService/ForwardPass',
            distributed__training__pb2.ForwardRequest.SerializeToString,
            distributed__training__pb2.ForwardResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def BackwardPass(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/distributed_training.TrainingService/BackwardPass',
            distributed__training__pb2.BackwardRequest.SerializeToString,
            distributed__training__pb2.BackwardResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def Heartbeat(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/distributed_training.TrainingService/Heartbeat',
            distributed__training__pb2.HeartbeatRequest.SerializeToString,
            distributed__training__pb2.HeartbeatResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)
