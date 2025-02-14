## Includes: common, interface
## Requirements: numpy scipy netifaces openvr
## Requirements: protobuf==3.19.4 grpcio==1.34.1 grpcio-tools==1.34.1
## Need to remove: condy_servicer.py
 
import sys
import os
import netifaces
import matplotlib.pyplot as plt
from threading import Thread
import time
from datetime import datetime
import numpy as np
import grpc
from concurrent import futures
from scipy.signal import butter
from scipy.stats import pearsonr
import dtw
import json
import math
import glob
 
from interfaces.control_socket_client import ControlSocketClient
from interfaces.rtde_socket_client import RTDESocketClient
import teleop_dev_pb2 as teleop_data
import teleop_dev_pb2_grpc as teleop_grpc
 
from get_device_data import *
 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'impl'))
 
TRIGGE_NAME = "menu_button"
CONTROL_PERIOD = 0.02
VEL_SCALE = 0.8
ACC_SCALE = 10.0
DEVICE_PORT = 20500
ERROR_TIME = 5.0
 
start_time = None

class SE3:
    def __init__(self):
        self.pos = np.zeros(3, dtype=np.float32) # [0,0,0]
        self.rot = np.identity(3, dtype=np.float32) # [1,0,0
                                                    #  0,1,0
                                                    #  0,0,1] <- 회전이 없는 상태

class UtilityFunc:
    @staticmethod
    def single_axis_rotMat(axis, angle):
        assert axis in ['x', 'y', 'z'], "Unavailable axis"
        
        c = np.cos(angle)
        s = np.sin(angle)

        if axis == 'x':
            return np.array([[1, 0, 0],
                            [0, c, -s],
                            [0, s, c]])
        elif axis == 'y':
            return np.array([[c, 0, s],
                            [0, 1, 0],
                            [-s, 0, c]])
        else:
            return np.array([[c, -s, 0],
                            [s, c, 0],
                            [0, 0, 1]])

    @staticmethod
    def euler_to_rotMat(euler_x, euler_y, euler_z):
        R_x = UtilityFunc.single_axis_rotMat('x', euler_x)
        R_y = UtilityFunc.single_axis_rotMat('y', euler_y)
        R_z = UtilityFunc.single_axis_rotMat('z', euler_z)
        return R_z @ R_y @ R_x
    
    @staticmethod
    def is_rotMat(rotMat):
        Rt = np.transpose(rotMat)
        shouldBeIdentity = np.dot(Rt, rotMat)
        I = np.identity(3, dtype = rotMat.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-4
    
    @staticmethod
    def rotMat_to_euler(rotMat):
        assert(UtilityFunc.is_rotMat(rotMat))
        sy = math.sqrt(rotMat[0,0] * rotMat[0,0] + rotMat[1,0] * rotMat[1,0])
        singular = sy < 1e-6
        if  not singular :
            x = math.atan2(rotMat[2,1] , rotMat[2,2])
            y = math.atan2(- rotMat[2,0], sy)
            z = math.atan2(rotMat[1,0], rotMat[0,0])
        else :
            x = math.atan2(- rotMat[1,2], rotMat[1,1])
            y = math.atan2(- rotMat[2,0], sy)
            z = 0
        return np.array([x, y, z])

class ControlTransformation:
    def __init__(self, fixed_robot_to_fixed_vive_euler):
        # 주어진 tele.json 파일의 오일러 각도를 이용해 회전 행렬 계산 -> 로봇과 컨트롤러 좌표계 간의 회전 행렬
        self.R_RV = UtilityFunc.euler_to_rotMat(
            fixed_robot_to_fixed_vive_euler[0], fixed_robot_to_fixed_vive_euler[1], fixed_robot_to_fixed_vive_euler[2]) 
        self.init_controller = SE3()  # P_Vv, R_Vv
        self.current_controller = SE3()  # P_Vv_, R_Vv_
        self.init_transformated = SE3()  # P_Rr, R_Rr
        self.current_transformated = SE3()  # P_Rr_, R_Rr_

    # 컨트롤러의 움직임을 로봇 좌표계로 변환
    def apply(self):
        P_vv__in_R = self.R_RV @ (self.current_controller.pos - self.init_controller.pos) # 컨트롤러의 위치 변화량과 회전 행렬 곱셈을 진행 -> 로봇 좌표계 기준에서 컨트롤러의 상대적 움직임을 나타내는 위치 벡터

        R_Rv = self.R_RV @ self.init_controller.rot # 컨트롤러의 로봇 좌표계 기준 초기 회전 행렬
        R_Rv_ = self.R_RV @ self.current_controller.rot # 컨트롤러의 로봇 좌표계 기준 현재 회전 행렬
        R_Rv_Rv = R_Rv_ @ R_Rv.T  # 현재 회전행렬과 초기 회전행렬의 전치행렬을 곱셈 -> 현재 상태와 초기 상태의 상대 회전 행렬

        self.current_transformated.pos = self.init_transformated.pos + P_vv__in_R # 현재 위치 = 처음 위치 + 상대적 움직임을 구하는 위치
        self.current_transformated.rot = R_Rv_Rv @ self.init_transformated.rot # 현재 회전 = 처음 회전 * 상대 회전 행렬

        current_transformated_euler = UtilityFunc.rotMat_to_euler(self.current_transformated.rot)  # 회전 행렬을 오일러 각도로 변환
        current_transformated_euler = current_transformated_euler * 180 / np.pi # 라디안을 도(degree)로 변환
        return self.current_transformated.pos.tolist() + current_transformated_euler.tolist() # 최종 변환값 [x,y,z,u,v,w]
 
def realtime_iir_filter(x, b, a, channel, x_prev, y_prev, filter_initialized):
    if not filter_initialized[channel]:
        x_prev[channel] = [x, x]
        y_prev[channel] = [x, x]
        filter_initialized[channel] = True
        return x
    x_hist = x_prev[channel]
    y_hist = y_prev[channel]
    y = b[0] * x + b[1] * x_hist[0] + b[2] * x_hist[1] - a[1] * y_hist[0] - a[2] * y_hist[1]
    x_prev[channel] = [x, x_hist[0]]
    y_prev[channel] = [y, y_hist[0]]
    return y
 
def low_pass_filter(value, b, a, x_prev, y_prev, filter_initialized):
    value = np.array(value)
    filtered_data = np.zeros_like(value)
    for channel in range(len(value)):
        filtered_data[channel] = realtime_iir_filter(value[channel], b, a, channel, x_prev, y_prev, filter_initialized)
    return filtered_data
 
def manage_Jump(value, prev_value, offset_for_jump):
    value = np.array(value)
    if prev_value is None:
        return value, offset_for_jump, value
    position_diff = value[:3] - prev_value[:3]
    angle_diff = (value[3:] - prev_value[3:] + 180) % 360 - 180
    if np.any(np.abs(position_diff) > 100):
        offset_for_jump[:3] += position_diff
    if np.any(np.abs(angle_diff) > 50):
        offset_for_jump[3:] += angle_diff
    adjusted_value = value - offset_for_jump
    return adjusted_value, offset_for_jump, value
 
def save_data(value_raw, value_adjusted, pos_robot, folder_path, save_count):
    global start_time
    elapsed_time_ms = int((time.time() - start_time) * 1000)
    info_with_time_raw = [elapsed_time_ms] + list(value_raw)
    info_with_time_adjusted = [elapsed_time_ms] + list(value_adjusted)
    info_with_time_robot = [elapsed_time_ms] + list(pos_robot)

    raw_data_path = os.path.join(folder_path, "raw_data")
    adjusted_data_path = os.path.join(folder_path, "adjusted_data")
    robot_data_path = os.path.join(folder_path, "robot_data")

    os.makedirs(raw_data_path, exist_ok=True)
    os.makedirs(adjusted_data_path, exist_ok=True)
    os.makedirs(robot_data_path, exist_ok=True)

    timestamp = datetime.fromtimestamp(start_time).strftime('%Y_%m_%d_%H_%M_%S')
    filename_raw = os.path.join(raw_data_path, f"{timestamp}_part{save_count}.txt")
    filename_adjusted = os.path.join(adjusted_data_path, f"{timestamp}_part{save_count}.txt")
    filename_robot = os.path.join(robot_data_path, f"{timestamp}_part{save_count}.txt")

    try:
        with open(filename_raw, "a") as file:
            file.write(",".join(map(str, info_with_time_raw)) + "\n")
        with open(filename_adjusted, "a") as file:
            file.write(",".join(map(str, info_with_time_adjusted)) + "\n")
        with open(filename_robot, "a") as file:
            file.write(",".join(map(str, info_with_time_robot)) + "\n")
    except Exception as e:
        print(f"Error saving data to files: {e}")
    
def load_data(folder_path):
    timestamp = datetime.fromtimestamp(start_time).strftime('%Y_%m_%d_%H_%M_%S')
    print(f"Loading data from {timestamp}")
 
    file_paths1 = sorted(glob.glob(os.path.join(folder_path, "adjusted_data", f"{timestamp}_part*.txt")))
    file_paths2 = sorted(glob.glob(os.path.join(folder_path, "robot_data", f"{timestamp}_part*.txt")))

    if not file_paths1 or not file_paths2:
        print("No data files found.")
        return None, None, None, None, None, None

    try:
        data1_list = [np.loadtxt(file, delimiter=",") for file in file_paths1]
        data2_list = [np.loadtxt(file, delimiter=",") for file in file_paths2]

    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None, None, None

    time_list = []
    controller_list = []
    robot_list = []

    for data1, data2 in zip(data1_list, data2_list):
        time1 = data1[:, 0] / 1000
        time2 = data2[:, 0] / 1000
        
        controller_data = data1[:, 1:7]
        controller_data[:,4] *= -1
        robot_data = data2[:, 1:7]

        time1_offset = time1[0]
        time2_offset = time2[0]
        controller_offset = controller_data[0, :]
        robot_offset = robot_data[0, :]

        time1 -= time1_offset
        time2 -= time2_offset
        controller_data -= controller_offset
        robot_data -= robot_offset

        for i in range(3, len(controller_data[0])):
            controller_data[:, i] = np.where(controller_data[:, i] > 180, controller_data[:, i] - 360, controller_data[:, i])
            controller_data[:, i] = np.where(controller_data[:, i] < -180, controller_data[:, i] + 360, controller_data[:, i])

        for i in range(3, len(robot_data[0])):
            robot_data[:, i] = np.where(robot_data[:, i] > 180, robot_data[:, i] - 360, robot_data[:, i])
            robot_data[:, i] = np.where(robot_data[:, i] < -180, robot_data[:, i] + 360, robot_data[:, i])

        time_list.append((time1, time2))
        controller_list.append(controller_data)
        robot_list.append(robot_data)

    return time_list, controller_list, robot_list

def plot_data(time_list, controller_list, robot_list):
    timestamp = datetime.fromtimestamp(start_time).strftime('%Y_%m_%d_%H_%M_%S')
    axes_labels = ['X', 'Y', 'Z', 'U', 'V', 'W']

    for idx, ((time1, time2), controller_data, robot_data) in enumerate(zip(time_list, controller_list, robot_list)):
        part_title = f"{timestamp} Part {idx + 1}"

        # 3D Trajectory Plot
        fig1 = plt.figure(figsize=(8, 6))
        ax_3d = fig1.add_subplot(111, projection='3d')
        ax_3d.plot(controller_data[:, 0], controller_data[:, 1], controller_data[:, 2], label='Controller value', color='orange')
        ax_3d.plot(robot_data[:, 0], robot_data[:, 1], robot_data[:, 2], label='Robot value', color='blue')
        ax_3d.set_title(f'3D Trajectory Plotting - {part_title}')
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')
        ax_3d.set_xlim(-150, 150)
        ax_3d.set_ylim(-150, 150)
        ax_3d.set_zlim(-150, 150)
        ax_3d.legend()
        ax_3d.grid()

        # XYZUVW 각 축별 포지션 플롯
        fig2, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
        fig2.suptitle(f"Position Comparison - {part_title}")

        for i, ax in enumerate(axes):
            ax.plot(time1, controller_data[:, i], label=f'Controller {axes_labels[i]}', linestyle='-', color='orange')
            ax.plot(time2, robot_data[:, i], label=f'Robot {axes_labels[i]}', linestyle='-', color='blue')
            ax.set_title(f"{axes_labels[i]}-Axis Comparison")

            if i < 3:
                ax.set_ylim(-250, 250)
                ax.set_ylabel(f"{axes_labels[i]} (mm)")
            else:
                ax.set_ylabel(f"{axes_labels[i]} (°)")
                ax.set_ylim(-90, 90)

            ax.legend()
            ax.grid()

        axes[-1].set_xlabel('Time (sec)')
        fig2.tight_layout()

        plt.show()

def calculate_perform(time_list, controller_list, robot_list):
    axes = ['X', 'Y', 'Z', 'U', 'V', 'W']

    for idx, ((time1, time2), controller_data, robot_data) in enumerate(zip(time_list, controller_list, robot_list)):
        print(f"\n=== Performance Metrics for Part {idx + 1} ===")

        # 1. 위치 벡터의 속도 및 가속도 계산
        print(f"\n<XYZ Position Vector Analysis>")
        controller_times = np.diff(time1)
        robot_times = np.diff(time2)

        controller_distances = np.sqrt(np.sum(np.diff(controller_data[:, :3], axis=0) ** 2, axis=1))
        robot_distances = np.sqrt(np.sum(np.diff(robot_data[:, :3], axis=0) ** 2, axis=1))

        controller_speeds = controller_distances / controller_times
        robot_speeds = robot_distances / robot_times
        controller_accel = np.abs(np.diff(controller_speeds) / controller_times[:-1])
        robot_accel = np.abs(np.diff(robot_speeds) / robot_times[:-1])

        print(f"    Controller Max Speed: {np.max(controller_speeds):.4f}")
        print(f"    Robot Max Speed: {np.max(robot_speeds):.4f}")
        print(f"    Controller Max Acceleration: {np.max(controller_accel):.4f}")
        print(f"    Robot Max Acceleration: {np.max(robot_accel):.4f}")

        # 2. 회전 벡터의 각도 차이 평균 및 RMSE
        print(f"\n<Rotation Error Calculation>")
        euler_rad_1 = np.radians(controller_data[:, 3:])
        euler_rad_2 = np.radians(robot_data[:, 3:])

        rotation1 = Rotation.from_euler('xyz', euler_rad_1)
        rotation2 = Rotation.from_euler('xyz', euler_rad_2)

        vector1 = rotation1.apply(np.array([[1, 0, 0]] * len(controller_data)))
        vector2 = rotation2.apply(np.array([[1, 0, 0]] * len(robot_data)))

        vec1_norm = vector1 / np.linalg.norm(vector1, axis=1, keepdims=True)
        vec2_norm = vector2 / np.linalg.norm(vector2, axis=1, keepdims=True)

        dot_product = np.sum(vec1_norm * vec2_norm, axis=1)
        dot_product = np.clip(dot_product, -1.0, 1.0)

        angle_deg = np.degrees(np.arccos(dot_product))

        print(f"    평균 각도 차이: {np.average(angle_deg):.4f} degrees")

        # 3. 각 X, Y, Z, U, V, W의 값 차이 RMSE
        print(f"\n<Value Difference RMSE>")
        for i in range(6):
            rmse = np.sqrt(np.mean((controller_data[:, i] - robot_data[:, i]) ** 2))
            print(f"    {axes[i].upper()}: RMSE = {rmse:.4f}")

        # 4. Dynamic Time Warping (DTW)
        print(f"\n<Normalized DTW>")
        for i in range(6):
            alignment = dtw.dtw(controller_data[:, i], robot_data[:, i], keep_internals=False)
            normalized_dtw = alignment.distance / len(alignment.index1)
            print(f"    {axes[i].upper()}: DTW = {normalized_dtw:.4f}")

        # 5. Pearson Correlation Coefficient (PCC) and R²
        print(f"\n<Pearson Correlation and R²>")
        for i in range(6):
            pearson_corr, p_value = pearsonr(controller_data[:, i], robot_data[:, i])
            print(f"    {axes[i].upper()}: R² = {pearson_corr**2:.4f}")

class DeviceManager:
    def __init__(self, folder_path, handler, calibration_file):
        self.prev_value = None
        self.offset_for_jump = np.zeros(6)
        self.folder_path = folder_path
        self.handler = handler
        self.handler.init_vive() if hasattr(self.handler, 'init_vive') else self.handler.init_udp()
        self.uvw = self.load_calibration(calibration_file)

    def load_calibration(self, file_name):
        try:
            with open(file_name, "r") as f:
                calibrated_data = json.load(f)
                print(f"Calibration data loaded from {file_name}: {calibrated_data['uvw']}")
                return calibrated_data["uvw"]
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading calibration data from {file_name}: {e}")
            return np.zeros(3).tolist()

    def get_data(self):
        if hasattr(self.handler, 'get_vive_pose'):
            return self.handler.get_vive_pose(), True, self.handler.get_vive_input()
        return self.handler.pos_rot, self.handler.val_detect, self.handler.val_enable

    def adjust_value(self, value):
        adjusted_value, self.offset_for_jump, self.prev_value = manage_Jump(value, self.prev_value, self.offset_for_jump)
        return adjusted_value

    def save_data(self, value_raw, value_adjusted, pos_robot, save_count):
        save_data(value_raw, value_adjusted, pos_robot, self.folder_path, save_count)

    def plot_data(self):
        try:
            time_list, controller_list, robot_list = load_data(self.folder_path)
            plot_data(time_list, controller_list, robot_list)
        except Exception as e:
            print(f"Error while plotting data: {e}")

    def calculate_perform(self):
        try:
            time_list, controller_list, robot_list = load_data(self.folder_path)
            calculate_perform(time_list, controller_list, robot_list)
        except Exception as e:
            print(f"Error while calculating performance: {e}")

class PhoneManager(DeviceManager):
    def __init__(self):
        self.fs = 50.0
        self.fc = 3
        self.wn = self.fc / (self.fs / 2)
        self.b, self.a = butter(2, self.wn, btype='low', analog=False)
        self.x_prev = [[0.0, 0.0] for _ in range(6)]
        self.y_prev = [[0.0, 0.0] for _ in range(6)]
        self.filter_initialized = [False] * 6
        super().__init__("./dataList", UDPHandler(), "./jsons/tele_phone.json")
        print(f"cutoff = {self.fc}")

    def adjust_value(self, value):
        adjusted_value = super().adjust_value(value)
        # return adjusted_value
        return low_pass_filter(adjusted_value, self.b, self.a, self.x_prev, self.y_prev, self.filter_initialized)

class ViveManager(DeviceManager):
    def __init__(self):
        super().__init__("./dataList_vive", ViveHandler(), "./jsons/tele_vive.json")

class ViveTrackerManager(DeviceManager):
    def __init__(self):
        super().__init__("./dataList_tracker", ViveTrackerHandler(), "./jsons/tele_tracker.json")

class TeleOpDeviceServicer(teleop_grpc.TeleOpDeviceServicer):
    ip_indy: str
    port_indy: str
    control: ControlSocketClient
    rtde: RTDESocketClient
    _thread: Thread
    _stop_stream: bool
   
    def __init__(self, device_type):
        if device_type == "phone": 
            self.device = PhoneManager()
        elif device_type == "vive": 
            self.device = ViveManager()
        else: 
            self.device = ViveTrackerManager()
        self.ip_indy = None
        self.port_indy = None
        self.control = None
        self.rtde = None
        self._stop_stream = False
        self._stream_running = False
        self._thread = None
        self.value_raw = None
        self.value_adjusted = None
        self.pos_robot = None
        self.save_enabled = False
        self._error_lasttime = False
        self.save_count = 0

        self.control_transform = ControlTransformation(self.device.uvw)
        self.prev_enable = False
           
    def StartTeleOpStream(self, request: teleop_data.TeleOpStreamReq, context) -> teleop_data.Response:
        if self._stream_running and self._thread is not None:
            if self.ip_indy == request.ip_indy:
                print(f"StartTeleOpStream re-requested from {request.ip_indy}:{request.port}")
                return teleop_data.Response()
            self._stop_stream = True
            self._thread.join()
        print(f"StartTeleOpStream to {request.ip_indy}:{request.port}")
        self.ip_indy = request.ip_indy
        self.port_indy = request.port
        self.control = ControlSocketClient(self.ip_indy, port=self.port_indy)
        self.rtde = RTDESocketClient(self.ip_indy)
        self._stop_stream = False
        self._thread = Thread(target=self._stream_fun, daemon=True)
        self._thread.start()
        return teleop_data.Response()
           
    def StopTeleOpStream(self, request: teleop_data.Empty, context) -> teleop_data.Response:
        print(f"StopTeleOpStream to {self.ip_indy}")
        self._stop_stream = True
        if self._thread and self._thread.is_alive():
            self._thread.join()
        return teleop_data.Response()
   
    def _stream_fun(self):
        self._stream_running = True
        self._error_count = 0
        time_last = time.time()
        while not self._stop_stream:
            try:
                step_time = time.time() - time_last # 스마트폰
                if step_time > CONTROL_PERIOD:
                    # 컨트롤러 값 가져오기, 보정
                    self.value_raw, detecv, enable = self.device.get_data() 
                    self.value_adjusted = self.device.adjust_value(self.value_raw)

                    # 기술 교시(컨트롤러 값 사용)
                    res = self.control.EnableTeleKey(enable)
                    if res is not None and detecv is not False and enable is not None:
                        res = self.control.MoveTeleLRec(self.value_adjusted, VEL_SCALE, ACC_SCALE)
                    if res is None:
                        raise(RuntimeError("Communication Failure"))
                    
                    # 로봇 좌표계로 변환된 컨트롤러 값, 로봇 값 저장
                    end_pose = np.array(self.value_adjusted)
                    end_pose[3:] *= np.pi / 180  # degree -> rad
                    self.control_transform.current_controller.pos = end_pose[:3]
                    self.control_transform.current_controller.rot = UtilityFunc.euler_to_rotMat(end_pose[3:][0], end_pose[3:][1], end_pose[3:][2])   
                    
                    self.pos_robot = self.rtde.GetControlData()['p']
                    if self.rtde.GetControlData()['op_state']==17 and enable is True: # 기술교시 중이고 enable이 시작할 때
                        # 좌표계 변환 함수 초기값 업데이트
                        if not self.prev_enable: 
                            self.save_count += 1
                            self.control_transform.init_controller.pos = self.control_transform.current_controller.pos
                            self.control_transform.init_controller.rot = self.control_transform.current_controller.rot
                            # pos_controller_robot_frame = self.control_transform.apply()
                            # end_pose = np.array(pos_controller_robot_frame)
                            end_pose = np.array(self.pos_robot)
                            end_pose[3:] *= np.pi / 180 
                            self.control_transform.init_transformated.pos = end_pose[:3] 
                            self.control_transform.init_transformated.rot = UtilityFunc.euler_to_rotMat(end_pose[3:][0], end_pose[3:][1], end_pose[3:][2])
                        # 좌표계 변환 값 
                        pos_controller_robot_frame = self.control_transform.apply()

                        if self.save_enabled: # 저장 옵션을 선택한 경우만 값 저장
                            self.device.save_data(self.value_adjusted, pos_controller_robot_frame, self.pos_robot, self.save_count)
                else :
                    time.sleep(CONTROL_PERIOD - step_time)
                if enable is not None:
                    self.prev_enable = enable
                self._error_lasttime = False
                self._error_count = 0

            except Exception as e:
                if not self._error_lasttime:
                    self._error_lasttime = True
                    print(f'Error in stream {e}')
                self._error_count += 1
                if self._error_count > 10:
                    print(f'Stop Stream By Error')
                    self._stop_stream = True
        self._stream_running = False
 
class TeleOp:
    def __init__(self, port=20500):
        self.device_type = None
        self.save_enabled = False
        self.port = port
        self.server_ip_address = None
        self.server = None
        self.servicer = None
        self.exit_flag = False
 
    def get_server_ip(self):
        try:
            for iface_name in netifaces.interfaces():
                iface_details = netifaces.ifaddresses(iface_name)
                if netifaces.AF_INET in iface_details:
                    for addr in iface_details[netifaces.AF_INET]:
                        ip_address = addr.get('addr')
                        if ip_address and ip_address.startswith(("192.168", "10.", "172.")):
                            print(f"Local IP address: {ip_address}")
                            return ip_address
            print("No local IP address found.")
            return None
        except Exception as e:
            print(f"Error retrieving server IP: {e}")
            return None
        
    def select_device(self):
        device_option = input("Select device (2: tracker, 1: phone, 0: vive): ").strip()
        if device_option not in ["0", "1", "2"]:
            print("Invalid input. Please enter 2 (tracker) or 1 (phone) or 0 (vive).")
            sys.exit(1)
        if device_option == "2" : self.device_type = "tracker"
        elif device_option == "1" : self.device_type = "phone"
        else : self.device_type = "vive"
        
    def enableSave(self):
        save_option = input("Enable data saving? (1: Yes, 0: No): ").strip()
        if save_option not in ["0", "1"]:
            print("Invalid input. Please enter 1 (Yes) or 0 (No).")
            sys.exit(1)
        self.save_enabled = save_option == "1"
 
    def initialize_server(self):
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        self.servicer = TeleOpDeviceServicer(device_type=self.device_type)
        teleop_grpc.add_TeleOpDeviceServicer_to_server(servicer=self.servicer, server=self.server)
        self.server.add_insecure_port(f'[::]:{DEVICE_PORT}')
        self.servicer.save_enabled = self.save_enabled
 
    def open_server(self):
        def server_thread():
            self.server.start()
            print(f"\nServer started for {self.device_type}")
            while not self.exit_flag:
                time.sleep(1)
            self.server.stop(0)
        Thread(target=server_thread, daemon=True).start()
 
    def wait_for_exit(self):
        print("\nPress 'q' to stop the server...")
        while not self.exit_flag:
            user_input = input().strip()
            if user_input.lower() == 'q':
                if self.device_type == "tracker" and self.servicer.device.handler.v:
                    self.servicer.device.handler.v.shutdown()
                self.exit_flag = True
                break
        
    def plot_data(self):
        if self.save_enabled:
            self.servicer.device.calculate_perform()
            self.servicer.device.plot_data()
        else:
            return
 
if __name__ == "__main__":
    start_time = time.time()
    # start_time = datetime.strptime("2025_01_23_10_53_26", '%Y_%m_%d_%H_%M_%S')
    # start_time = int(start_time.timestamp())
    teleop = TeleOp()
    teleop.get_server_ip()
    teleop.select_device()
    teleop.enableSave()
    teleop.initialize_server()
    teleop.open_server()
    time.sleep(0.5)
    teleop.wait_for_exit()
    time.sleep(0.5)
    teleop.plot_data()
