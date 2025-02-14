import time
import sys
import openvr
import math
import json
from websocket import create_connection
import threading

from openvr import VREvent_t, VREvent_ButtonPress, VREvent_TrackedDeviceActivated, VREvent_TrackedDeviceDeactivated, VREvent_ButtonUnpress

from functools import lru_cache

class WebSocketThread(threading.Thread):
    def __init__(self, url, triad_openvr_instance, max_retries=5, retry_delay=2, skip_existing_logs=True):
        super().__init__()
        self.url = url
        self.ws = None
        self.triad_openvr_instance = triad_openvr_instance
        self.running = True
        self.max_retries = max_retries  # 최대 재시도 횟수
        self.retry_delay = retry_delay  # 재시도 대기 시간(초)
        self.skip_existing_logs = skip_existing_logs  # 기존 로그 무시 여부
        self.connected_time = None  # 연결된 시간 기록

    def run(self):
        retries = 0
        while self.running:
            try:
                # WebSocket 연결 시도
                self.ws = create_connection(self.url)
                print("WebSocket 연결 성공!")
                self.ws.send("console_open")
                retries = 0  # 연결 성공 시 재시도 횟수 초기화
                self.connected_time = time.time()  # 연결 시간 기록

                while self.running:
                    try:
                        # 메시지 수신 및 처리
                        response = self.ws.recv()
                        data = json.loads(response)

                        # 기존 로그 무시 로직
                        if self.skip_existing_logs:
                            current_time = time.time()
                            if current_time - self.connected_time < 1:  # 연결 후 1초 동안 수신된 로그 무시
                                continue

                        if data.get("sLogName") == "vrserver":
                            message = data.get("sMessage")
                            if "power button" in message:
                                power_button_state = int(
                                    message.split("power button: ")[1].split(",")[0].strip()
                                )
                                if power_button_state == 1:
                                    self.triad_openvr_instance.toggle_enable()
                    except json.JSONDecodeError:
                        print("수신된 메시지를 JSON으로 파싱할 수 없습니다:", response)
                    except Exception as e:
                        print("WebSocket 오류 발생:", e)
                        break
            except Exception as e:
                print(f"WebSocket 연결 실패: {e}")

                # 재연결 로직
                retries += 1
                if retries > self.max_retries:
                    print("최대 재시도 횟수를 초과하여 WebSocket 쓰레드를 종료합니다.")
                    break
                print(f"{self.retry_delay}초 후 WebSocket 재연결 시도... (시도 {retries}/{self.max_retries})")
                time.sleep(self.retry_delay)
            finally:
                if self.ws:
                    self.ws.close()
                    print("WebSocket 연결 종료")

    def stop(self):
        self.running = False
        if self.ws:
            self.ws.close()

# Function to print out text but instead of starting a new line it will overwrite the existing line
def update_text(txt):
    sys.stdout.write('\r'+txt)
    sys.stdout.flush()

#Convert the standard 3x4 position/rotation matrix to a x,y,z location and the appropriate Euler angles (in degrees)
def convert_to_euler(pose_mat):
    yaw = 180 / math.pi * math.atan2(pose_mat[1][0], pose_mat[0][0])
    pitch = 180 / math.pi * math.atan2(pose_mat[2][0], pose_mat[0][0])
    roll = 180 / math.pi * math.atan2(pose_mat[2][1], pose_mat[2][2])
    x = pose_mat[0][3]
    y = pose_mat[1][3]
    z = pose_mat[2][3]
    return [x,y,z,yaw,pitch,roll]

#Convert the standard 3x4 position/rotation matrix to a x,y,z location and the appropriate Quaternion
def convert_to_quaternion(pose_mat):
    # Per issue #2, adding a abs() so that sqrt only results in real numbers
    r_w = math.sqrt(abs(1+pose_mat[0][0]+pose_mat[1][1]+pose_mat[2][2]))/2
    r_x = (pose_mat[2][1]-pose_mat[1][2])/(4*r_w)
    r_y = (pose_mat[0][2]-pose_mat[2][0])/(4*r_w)
    r_z = (pose_mat[1][0]-pose_mat[0][1])/(4*r_w)

    x = pose_mat[0][3]
    y = pose_mat[1][3]
    z = pose_mat[2][3]
    return [x,y,z,r_w,r_x,r_y,r_z]

#Define a class to make it easy to append pose matricies and convert to both Euler and Quaternion for plotting
class pose_sample_buffer():
    def __init__(self):
        self.i = 0
        self.index = []
        self.time = []
        self.x = []
        self.y = []
        self.z = []
        self.yaw = []
        self.pitch = []
        self.roll = []
        self.r_w = []
        self.r_x = []
        self.r_y = []
        self.r_z = []

    def append(self,pose_mat,t):
        self.time.append(t)
        self.x.append(pose_mat[0][3])
        self.y.append(pose_mat[1][3])
        self.z.append(pose_mat[2][3])
        self.yaw.append(180 / math.pi * math.atan(pose_mat[1][0] /pose_mat[0][0]))
        self.pitch.append(180 / math.pi * math.atan(-1 * pose_mat[2][0] / math.sqrt(pow(pose_mat[2][1], 2) + math.pow(pose_mat[2][2], 2))))
        self.roll.append(180 / math.pi * math.atan(pose_mat[2][1] /pose_mat[2][2]))
        r_w = math.sqrt(abs(1+pose_mat[0][0]+pose_mat[1][1]+pose_mat[2][2]))/2
        self.r_w.append(r_w)
        self.r_x.append((pose_mat[2][1]-pose_mat[1][2])/(4*r_w))
        self.r_y.append((pose_mat[0][2]-pose_mat[2][0])/(4*r_w))
        self.r_z.append((pose_mat[1][0]-pose_mat[0][1])/(4*r_w))

def get_pose(vr_obj):
    return vr_obj.getDeviceToAbsoluteTrackingPose(openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount)


class vr_tracked_device():
    def __init__(self,vr_obj,index,device_class):
        self.device_class = device_class
        self.index = index
        self.vr = vr_obj
        self.event = VREvent_t()

    @lru_cache(maxsize=None)
    def get_serial(self):
        return self.vr.getStringTrackedDeviceProperty(self.index, openvr.Prop_SerialNumber_String)

    def get_model(self):
        return self.vr.getStringTrackedDeviceProperty(self.index, openvr.Prop_ModelNumber_String)

    def get_battery_percent(self):
        return self.vr.getFloatTrackedDeviceProperty(self.index, openvr.Prop_DeviceBatteryPercentage_Float)

    def is_charging(self):
        return self.vr.getBoolTrackedDeviceProperty(self.index, openvr.Prop_DeviceIsCharging_Bool)


    def sample(self,num_samples,sample_rate):
        interval = 1/sample_rate
        rtn = pose_sample_buffer()
        sample_start = time.time()
        for i in range(num_samples):
            start = time.time()
            pose = get_pose(self.vr)
            rtn.append(pose[self.index].mDeviceToAbsoluteTracking,time.time()-sample_start)
            sleep_time = interval- (time.time()-start)
            if sleep_time>0:
                time.sleep(sleep_time)
        return rtn

    def get_pose_euler(self, pose=None):
        if pose == None:
            pose = get_pose(self.vr)
        if pose[self.index].bPoseIsValid:
            return convert_to_euler(pose[self.index].mDeviceToAbsoluteTracking)
        else:
            return None

    def get_pose_matrix(self, pose=None):
        if pose == None:
            pose = get_pose(self.vr)
        if pose[self.index].bPoseIsValid:
            return pose[self.index].mDeviceToAbsoluteTracking
        else:
            return None

    def get_velocity(self, pose=None):
        if pose == None:
            pose = get_pose(self.vr)
        if pose[self.index].bPoseIsValid:
            return pose[self.index].vVelocity
        else:
            return None

    def get_angular_velocity(self, pose=None):
        if pose == None:
            pose = get_pose(self.vr)
        if pose[self.index].bPoseIsValid:
            return pose[self.index].vAngularVelocity
        else:
            return None

    def get_pose_quaternion(self, pose=None):
        if pose == None:
            pose = get_pose(self.vr)
        if pose[self.index].bPoseIsValid:
            return convert_to_quaternion(pose[self.index].mDeviceToAbsoluteTracking)
        else:
            return None

    def get_controller_inputs(self):
        # result, state = self.vr.getControllerState(self.index)
        # if not result:
        #     print(f"Failed to get controller state for device {self.index}")
        #     return None
        # print(f"Controller state for device {self.index}: {state}")
        return self.vr.enable
    def trigger_haptic_pulse(self, duration_micros=1000, axis_id=0):
        """
        Causes devices with haptic feedback to vibrate for a short time.
        """
        self.vr.triggerHapticPulse(self.index, axis_id, duration_micros)

class vr_tracking_reference(vr_tracked_device):
    def get_mode(self):
        return self.vr.getStringTrackedDeviceProperty(self.index,openvr.Prop_ModeLabel_String).decode('utf-8').upper()
    def sample(self,num_samples,sample_rate):
        print("Warning: Tracking References do not move, sample isn't much use...")


class triad_openvr():
    def __init__(self):
        # OpenVR 초기화
        self.vr = openvr.init(openvr.VRApplication_Other)
        self.vrsystem = openvr.VRSystem()

        # Tracker 정보 저장용
        self.object_names = {"Tracker": []}
        self.devices = {}
        self.device_index_map = {}

        # WebSocket 쓰레드 시작
        self.enable = False
        self.ws_thread = WebSocketThread("ws://127.0.0.1:27062", self)
        self.ws_thread.start()

        # 연결된 디바이스 확인 및 Tracker 추가
        poses = self.vr.getDeviceToAbsoluteTrackingPose(openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount)
        for i in range(openvr.k_unMaxTrackedDeviceCount):
            if poses[i].bDeviceIsConnected:
                device_class = self.vr.getTrackedDeviceClass(i)
                if device_class == openvr.TrackedDeviceClass_GenericTracker:
                    self.add_tracked_device(i)

    def __del__(self):
        openvr.shutdown()
        self.ws_thread.stop()
        self.ws_thread.join()

    def toggle_enable(self):
        """power_button 이벤트를 처리하여 enable 변수 반전"""
        self.enable = not self.enable
        print(f"Enable 상태 변경: {self.enable}")

    def add_tracked_device(self, tracked_device_index):
        device_class = self.vr.getTrackedDeviceClass(tracked_device_index)
        if device_class == openvr.TrackedDeviceClass_GenericTracker:
            device_name = "tracker_" + str(len(self.object_names["Tracker"]) + 1)
            self.object_names["Tracker"].append(device_name)
            self.devices[device_name] = vr_tracked_device(self.vr, tracked_device_index, "Tracker")
            self.device_index_map[tracked_device_index] = device_name

    def poll_vr_events(self):
        """
        VR 이벤트를 처리하며 새로운 Tracker를 감지하거나 제거
        """
        event = openvr.VREvent_t()
        while self.vrsystem.pollNextEvent(event):
            if event.eventType == openvr.VREvent_TrackedDeviceActivated:
                device_class = self.vr.getTrackedDeviceClass(event.trackedDeviceIndex)
                if device_class == openvr.TrackedDeviceClass_GenericTracker:
                    self.add_tracked_device(event.trackedDeviceIndex)
            elif event.eventType == openvr.VREvent_TrackedDeviceDeactivated:
                if event.trackedDeviceIndex in self.device_index_map:
                    self.remove_tracked_device(event.trackedDeviceIndex)

    def remove_tracked_device(self, tracked_device_index):
        if tracked_device_index in self.device_index_map:
            device_name = self.device_index_map[tracked_device_index]
            self.object_names["Tracker"].remove(device_name)
            del self.device_index_map[tracked_device_index]
            del self.devices[device_name]
        else:
            raise Exception(f"Tracked device index {tracked_device_index} not valid. Not removing.")

    def print_discovered_objects(self):
        """
        감지된 Tracker 목록을 출력
        """
        print(f"Found {len(self.object_names['Tracker'])} Trackers")
        for device in self.object_names["Tracker"]:
            print(f"  {device} (Serial: {self.devices[device].get_serial()}, Model: {self.devices[device].get_model()})")

    def get_tracker_poses(self):
        """
        모든 Tracker의 Pose를 반환
        """
        poses = self.vr.getDeviceToAbsoluteTrackingPose(openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount)
        tracker_poses = {}
        for tracker_name in self.object_names["Tracker"]:
            tracker = self.devices[tracker_name]
            tracker_poses[tracker_name] = tracker.get_pose_euler(poses)
        return tracker_poses

    def shutdown(self):
        """WebSocket 쓰레드 종료"""
        if self.ws_thread.is_alive():
            self.ws_thread.stop()
            self.ws_thread.join()