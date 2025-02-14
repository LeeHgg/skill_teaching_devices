import sys
sys.path.append('interfaces/impl')

from .impl.device_pb2_grpc import *
from .impl.config_pb2_grpc import *
from .impl.control_pb2_grpc import *
from .impl.rtde_pb2_grpc import *


from .config_socket_client import ConfigSocketClient as ConfigClient
from .control_socket_client import ControlSocketClient as ControlClient
from .rtde_socket_client import RTDESocketClient as RTDEClient
