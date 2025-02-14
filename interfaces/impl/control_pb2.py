# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: control.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import control_msgs_pb2 as control__msgs__pb2
import config_msgs_pb2 as config__msgs__pb2
import common_msgs_pb2 as common__msgs__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='control.proto',
  package='Nrmk.IndyFramework',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\rcontrol.proto\x12\x12Nrmk.IndyFramework\x1a\x12\x63ontrol_msgs.proto\x1a\x11\x63onfig_msgs.proto\x1a\x11\x63ommon_msgs.proto2\xe7\x34\n\x07\x43ontrol\x12N\n\x0eGetControlInfo\x12\x19.Nrmk.IndyFramework.Empty\x1a\x1f.Nrmk.IndyFramework.ControlInfo\"\x00\x12[\n\x0f\x41\x63tivateIndySDK\x12\".Nrmk.IndyFramework.SDKLicenseInfo\x1a\".Nrmk.IndyFramework.SDKLicenseResp\"\x00\x12\x45\n\x05MoveJ\x12\x1c.Nrmk.IndyFramework.MoveJReq\x1a\x1c.Nrmk.IndyFramework.Response\"\x00\x12G\n\x06MoveJT\x12\x1d.Nrmk.IndyFramework.MoveJTReq\x1a\x1c.Nrmk.IndyFramework.Response\"\x00\x12\x45\n\x05MoveL\x12\x1c.Nrmk.IndyFramework.MoveLReq\x1a\x1c.Nrmk.IndyFramework.Response\"\x00\x12G\n\x06MoveLT\x12\x1d.Nrmk.IndyFramework.MoveLTReq\x1a\x1c.Nrmk.IndyFramework.Response\"\x00\x12\x45\n\x05MoveC\x12\x1c.Nrmk.IndyFramework.MoveCReq\x1a\x1c.Nrmk.IndyFramework.Response\"\x00\x12G\n\x06MoveCT\x12\x1d.Nrmk.IndyFramework.MoveCTReq\x1a\x1c.Nrmk.IndyFramework.Response\"\x00\x12U\n\rMoveJointTraj\x12$.Nrmk.IndyFramework.MoveJointTrajReq\x1a\x1c.Nrmk.IndyFramework.Response\"\x00\x12S\n\x0cMoveTaskTraj\x12#.Nrmk.IndyFramework.MoveTaskTrajReq\x1a\x1c.Nrmk.IndyFramework.Response\"\x00\x12M\n\tMoveGcode\x12 .Nrmk.IndyFramework.MoveGcodeReq\x1a\x1c.Nrmk.IndyFramework.Response\"\x00\x12S\n\x0cMoveConveyor\x12#.Nrmk.IndyFramework.MoveConveyorReq\x1a\x1c.Nrmk.IndyFramework.Response\"\x00\x12G\n\x06WaitIO\x12\x1d.Nrmk.IndyFramework.WaitIOReq\x1a\x1c.Nrmk.IndyFramework.Response\"\x00\x12K\n\x08WaitTime\x12\x1f.Nrmk.IndyFramework.WaitTimeReq\x1a\x1c.Nrmk.IndyFramework.Response\"\x00\x12S\n\x0cWaitProgress\x12#.Nrmk.IndyFramework.WaitProgressReq\x1a\x1c.Nrmk.IndyFramework.Response\"\x00\x12K\n\x08WaitTraj\x12\x1f.Nrmk.IndyFramework.WaitTrajReq\x1a\x1c.Nrmk.IndyFramework.Response\"\x00\x12O\n\nWaitRadius\x12!.Nrmk.IndyFramework.WaitRadiusReq\x1a\x1c.Nrmk.IndyFramework.Response\"\x00\x12M\n\tMoveJCond\x12 .Nrmk.IndyFramework.MoveJCondReq\x1a\x1c.Nrmk.IndyFramework.Response\"\x00\x12Q\n\x0eMoveLinearAxis\x12\x1f.Nrmk.IndyFramework.MoveAxisReq\x1a\x1c.Nrmk.IndyFramework.Response\"\x00\x12I\n\nStopMotion\x12\x1b.Nrmk.IndyFramework.StopCat\x1a\x1c.Nrmk.IndyFramework.Response\"\x00\x12K\n\x0bPauseMotion\x12\x1c.Nrmk.IndyFramework.PauseCat\x1a\x1c.Nrmk.IndyFramework.Response\"\x00\x12N\n\x11SetDirectTeaching\x12\x19.Nrmk.IndyFramework.State\x1a\x1c.Nrmk.IndyFramework.Response\"\x00\x12N\n\x11SetSimulationMode\x12\x19.Nrmk.IndyFramework.State\x1a\x1c.Nrmk.IndyFramework.Response\"\x00\x12S\n\x14SetCustomControlMode\x12\x1b.Nrmk.IndyFramework.IntMode\x1a\x1c.Nrmk.IndyFramework.Response\"\x00\x12P\n\x14GetCustomControlMode\x12\x19.Nrmk.IndyFramework.Empty\x1a\x1b.Nrmk.IndyFramework.IntMode\"\x00\x12T\n\x17SetFrictionCompensation\x12\x19.Nrmk.IndyFramework.State\x1a\x1c.Nrmk.IndyFramework.Response\"\x00\x12V\n\x1cGetFrictionCompensationState\x12\x19.Nrmk.IndyFramework.Empty\x1a\x19.Nrmk.IndyFramework.State\"\x00\x12K\n\x0bSetTactTime\x12\x1c.Nrmk.IndyFramework.TactTime\x1a\x1c.Nrmk.IndyFramework.Response\"\x00\x12H\n\x0bGetTactTime\x12\x19.Nrmk.IndyFramework.Empty\x1a\x1c.Nrmk.IndyFramework.TactTime\"\x00\x12\x44\n\x07Recover\x12\x19.Nrmk.IndyFramework.Empty\x1a\x1c.Nrmk.IndyFramework.Response\"\x00\x12\x43\n\x06Reboot\x12\x19.Nrmk.IndyFramework.Empty\x1a\x1c.Nrmk.IndyFramework.Response\"\x00\x12N\n\x11SetManualRecovery\x12\x19.Nrmk.IndyFramework.State\x1a\x1c.Nrmk.IndyFramework.Response\"\x00\x12O\n\x10MoveRecoverJoint\x12\x1b.Nrmk.IndyFramework.TargetJ\x1a\x1c.Nrmk.IndyFramework.Response\"\x00\x12O\n\rSearchProgram\x12\x1b.Nrmk.IndyFramework.Program\x1a\x1f.Nrmk.IndyFramework.ProgramInfo\"\x00\x12J\n\x0bPlayProgram\x12\x1b.Nrmk.IndyFramework.Program\x1a\x1c.Nrmk.IndyFramework.Response\"\x00\x12I\n\x0cPauseProgram\x12\x19.Nrmk.IndyFramework.Empty\x1a\x1c.Nrmk.IndyFramework.Response\"\x00\x12J\n\rResumeProgram\x12\x19.Nrmk.IndyFramework.Empty\x1a\x1c.Nrmk.IndyFramework.Response\"\x00\x12H\n\x0bStopProgram\x12\x19.Nrmk.IndyFramework.Empty\x1a\x1c.Nrmk.IndyFramework.Response\"\x00\x12\x45\n\tSendAlarm\x12\x1b.Nrmk.IndyFramework.Message\x1a\x19.Nrmk.IndyFramework.Empty\"\x00\x12J\n\x0eSendAnnotation\x12\x1b.Nrmk.IndyFramework.Message\x1a\x19.Nrmk.IndyFramework.Empty\"\x00\x12\x61\n\x11PlayTuningProgram\x12!.Nrmk.IndyFramework.TuningProgram\x1a\'.Nrmk.IndyFramework.CollisionThresholds\"\x00\x12N\n\x0fPlayProgramLine\x12\x1b.Nrmk.IndyFramework.Program\x1a\x1c.Nrmk.IndyFramework.Response\"\x00\x12`\n\x19SetModbusVariableNameList\x12&.Nrmk.IndyFramework.ModbusVariableList\x1a\x19.Nrmk.IndyFramework.Empty\"\x00\x12O\n\x13SetVariableNameList\x12\x1b.Nrmk.IndyFramework.AllVars\x1a\x19.Nrmk.IndyFramework.Empty\"\x00\x12O\n\x13GetVariableNameList\x12\x19.Nrmk.IndyFramework.Empty\x1a\x1b.Nrmk.IndyFramework.AllVars\"\x00\x12J\n\x0eSetIntVariable\x12\x1b.Nrmk.IndyFramework.IntVars\x1a\x19.Nrmk.IndyFramework.Empty\"\x00\x12J\n\x0eGetIntVariable\x12\x19.Nrmk.IndyFramework.Empty\x1a\x1b.Nrmk.IndyFramework.IntVars\"\x00\x12P\n\x11SetModbusVariable\x12\x1e.Nrmk.IndyFramework.ModbusVars\x1a\x19.Nrmk.IndyFramework.Empty\"\x00\x12P\n\x11GetModbusVariable\x12\x19.Nrmk.IndyFramework.Empty\x1a\x1e.Nrmk.IndyFramework.ModbusVars\"\x00\x12L\n\x0fSetBoolVariable\x12\x1c.Nrmk.IndyFramework.BoolVars\x1a\x19.Nrmk.IndyFramework.Empty\"\x00\x12L\n\x0fGetBoolVariable\x12\x19.Nrmk.IndyFramework.Empty\x1a\x1c.Nrmk.IndyFramework.BoolVars\"\x00\x12N\n\x10SetFloatVariable\x12\x1d.Nrmk.IndyFramework.FloatVars\x1a\x19.Nrmk.IndyFramework.Empty\"\x00\x12N\n\x10GetFloatVariable\x12\x19.Nrmk.IndyFramework.Empty\x1a\x1d.Nrmk.IndyFramework.FloatVars\"\x00\x12L\n\x0fSetJPosVariable\x12\x1c.Nrmk.IndyFramework.JPosVars\x1a\x19.Nrmk.IndyFramework.Empty\"\x00\x12L\n\x0fGetJPosVariable\x12\x19.Nrmk.IndyFramework.Empty\x1a\x1c.Nrmk.IndyFramework.JPosVars\"\x00\x12L\n\x0fSetTPosVariable\x12\x1c.Nrmk.IndyFramework.TPosVars\x1a\x19.Nrmk.IndyFramework.Empty\"\x00\x12L\n\x0fGetTPosVariable\x12\x19.Nrmk.IndyFramework.Empty\x1a\x1c.Nrmk.IndyFramework.TPosVars\"\x00\x12i\n\x11InverseKinematics\x12(.Nrmk.IndyFramework.InverseKinematicsReq\x1a(.Nrmk.IndyFramework.InverseKinematicsRes\"\x00\x12i\n\x11\x46orwardKinematics\x12(.Nrmk.IndyFramework.ForwardKinematicsReq\x1a(.Nrmk.IndyFramework.ForwardKinematicsRes\"\x00\x12~\n\x18\x43heckAproachRetractValid\x12/.Nrmk.IndyFramework.CheckAproachRetractValidReq\x1a/.Nrmk.IndyFramework.CheckAproachRetractValidRes\"\x00\x12l\n\x12GetPalletPointList\x12).Nrmk.IndyFramework.GetPalletPointListReq\x1a).Nrmk.IndyFramework.GetPalletPointListRes\"\x00\x12u\n\x15\x43\x61lculateRelativePose\x12,.Nrmk.IndyFramework.CalculateRelativePoseReq\x1a,.Nrmk.IndyFramework.CalculateRelativePoseRes\"\x00\x12{\n\x17\x43\x61lculateCurrentPoseRel\x12..Nrmk.IndyFramework.CalculateCurrentPoseRelReq\x1a..Nrmk.IndyFramework.CalculateCurrentPoseRelRes\"\x00\x12G\n\rPingFromConty\x12\x19.Nrmk.IndyFramework.Empty\x1a\x19.Nrmk.IndyFramework.Empty\"\x00\x12P\n\x0fGetTeleOpDevice\x12\x19.Nrmk.IndyFramework.Empty\x1a .Nrmk.IndyFramework.TeleOpDevice\"\x00\x12N\n\x0eGetTeleOpState\x12\x19.Nrmk.IndyFramework.Empty\x1a\x1f.Nrmk.IndyFramework.TeleOpState\"\x00\x12W\n\x13\x43onnectTeleOpDevice\x12 .Nrmk.IndyFramework.TeleOpDevice\x1a\x1c.Nrmk.IndyFramework.Response\"\x00\x12S\n\x16\x44isConnectTeleOpDevice\x12\x19.Nrmk.IndyFramework.Empty\x1a\x1c.Nrmk.IndyFramework.Response\"\x00\x12I\n\x0fReadTeleOpInput\x12\x19.Nrmk.IndyFramework.Empty\x1a\x19.Nrmk.IndyFramework.TeleP\"\x00\x12N\n\x0bStartTeleOp\x12\x1f.Nrmk.IndyFramework.TeleOpState\x1a\x1c.Nrmk.IndyFramework.Response\"\x00\x12G\n\nStopTeleOp\x12\x19.Nrmk.IndyFramework.Empty\x1a\x1c.Nrmk.IndyFramework.Response\"\x00\x12O\n\x0bSetPlayRate\x12 .Nrmk.IndyFramework.TelePlayRate\x1a\x1c.Nrmk.IndyFramework.Response\"\x00\x12L\n\x0bGetPlayRate\x12\x19.Nrmk.IndyFramework.Empty\x1a .Nrmk.IndyFramework.TelePlayRate\"\x00\x12R\n\x0fGetTeleFileList\x12\x19.Nrmk.IndyFramework.Empty\x1a\".Nrmk.IndyFramework.TeleOpFileList\"\x00\x12Q\n\x0eSaveTeleMotion\x12\x1f.Nrmk.IndyFramework.TeleFileReq\x1a\x1c.Nrmk.IndyFramework.Response\"\x00\x12Q\n\x0eLoadTeleMotion\x12\x1f.Nrmk.IndyFramework.TeleFileReq\x1a\x1c.Nrmk.IndyFramework.Response\"\x00\x12S\n\x10\x44\x65leteTeleMotion\x12\x1f.Nrmk.IndyFramework.TeleFileReq\x1a\x1c.Nrmk.IndyFramework.Response\"\x00\x12J\n\rEnableTeleKey\x12\x19.Nrmk.IndyFramework.State\x1a\x1c.Nrmk.IndyFramework.Response\"\x00\x12M\n\tMoveTeleJ\x12 .Nrmk.IndyFramework.MoveTeleJReq\x1a\x1c.Nrmk.IndyFramework.Response\"\x00\x12M\n\tMoveTeleL\x12 .Nrmk.IndyFramework.MoveTeleLReq\x1a\x1c.Nrmk.IndyFramework.Response\"\x00\x12G\n\x06MoveLF\x12\x1d.Nrmk.IndyFramework.MoveLFReq\x1a\x1c.Nrmk.IndyFramework.Response\"\x00\x12\x66\n\x1aGetTransformedFTSensorData\x12\x19.Nrmk.IndyFramework.Empty\x1a+.Nrmk.IndyFramework.TransformedFTSensorData\"\x00\x62\x06proto3'
  ,
  dependencies=[control__msgs__pb2.DESCRIPTOR,config__msgs__pb2.DESCRIPTOR,common__msgs__pb2.DESCRIPTOR,])



_sym_db.RegisterFileDescriptor(DESCRIPTOR)



_CONTROL = _descriptor.ServiceDescriptor(
  name='Control',
  full_name='Nrmk.IndyFramework.Control',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=96,
  serialized_end=6855,
  methods=[
  _descriptor.MethodDescriptor(
    name='GetControlInfo',
    full_name='Nrmk.IndyFramework.Control.GetControlInfo',
    index=0,
    containing_service=None,
    input_type=common__msgs__pb2._EMPTY,
    output_type=control__msgs__pb2._CONTROLINFO,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='ActivateIndySDK',
    full_name='Nrmk.IndyFramework.Control.ActivateIndySDK',
    index=1,
    containing_service=None,
    input_type=control__msgs__pb2._SDKLICENSEINFO,
    output_type=control__msgs__pb2._SDKLICENSERESP,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='MoveJ',
    full_name='Nrmk.IndyFramework.Control.MoveJ',
    index=2,
    containing_service=None,
    input_type=control__msgs__pb2._MOVEJREQ,
    output_type=common__msgs__pb2._RESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='MoveJT',
    full_name='Nrmk.IndyFramework.Control.MoveJT',
    index=3,
    containing_service=None,
    input_type=control__msgs__pb2._MOVEJTREQ,
    output_type=common__msgs__pb2._RESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='MoveL',
    full_name='Nrmk.IndyFramework.Control.MoveL',
    index=4,
    containing_service=None,
    input_type=control__msgs__pb2._MOVELREQ,
    output_type=common__msgs__pb2._RESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='MoveLT',
    full_name='Nrmk.IndyFramework.Control.MoveLT',
    index=5,
    containing_service=None,
    input_type=control__msgs__pb2._MOVELTREQ,
    output_type=common__msgs__pb2._RESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='MoveC',
    full_name='Nrmk.IndyFramework.Control.MoveC',
    index=6,
    containing_service=None,
    input_type=control__msgs__pb2._MOVECREQ,
    output_type=common__msgs__pb2._RESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='MoveCT',
    full_name='Nrmk.IndyFramework.Control.MoveCT',
    index=7,
    containing_service=None,
    input_type=control__msgs__pb2._MOVECTREQ,
    output_type=common__msgs__pb2._RESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='MoveJointTraj',
    full_name='Nrmk.IndyFramework.Control.MoveJointTraj',
    index=8,
    containing_service=None,
    input_type=control__msgs__pb2._MOVEJOINTTRAJREQ,
    output_type=common__msgs__pb2._RESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='MoveTaskTraj',
    full_name='Nrmk.IndyFramework.Control.MoveTaskTraj',
    index=9,
    containing_service=None,
    input_type=control__msgs__pb2._MOVETASKTRAJREQ,
    output_type=common__msgs__pb2._RESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='MoveGcode',
    full_name='Nrmk.IndyFramework.Control.MoveGcode',
    index=10,
    containing_service=None,
    input_type=control__msgs__pb2._MOVEGCODEREQ,
    output_type=common__msgs__pb2._RESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='MoveConveyor',
    full_name='Nrmk.IndyFramework.Control.MoveConveyor',
    index=11,
    containing_service=None,
    input_type=control__msgs__pb2._MOVECONVEYORREQ,
    output_type=common__msgs__pb2._RESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='WaitIO',
    full_name='Nrmk.IndyFramework.Control.WaitIO',
    index=12,
    containing_service=None,
    input_type=control__msgs__pb2._WAITIOREQ,
    output_type=common__msgs__pb2._RESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='WaitTime',
    full_name='Nrmk.IndyFramework.Control.WaitTime',
    index=13,
    containing_service=None,
    input_type=control__msgs__pb2._WAITTIMEREQ,
    output_type=common__msgs__pb2._RESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='WaitProgress',
    full_name='Nrmk.IndyFramework.Control.WaitProgress',
    index=14,
    containing_service=None,
    input_type=control__msgs__pb2._WAITPROGRESSREQ,
    output_type=common__msgs__pb2._RESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='WaitTraj',
    full_name='Nrmk.IndyFramework.Control.WaitTraj',
    index=15,
    containing_service=None,
    input_type=control__msgs__pb2._WAITTRAJREQ,
    output_type=common__msgs__pb2._RESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='WaitRadius',
    full_name='Nrmk.IndyFramework.Control.WaitRadius',
    index=16,
    containing_service=None,
    input_type=control__msgs__pb2._WAITRADIUSREQ,
    output_type=common__msgs__pb2._RESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='MoveJCond',
    full_name='Nrmk.IndyFramework.Control.MoveJCond',
    index=17,
    containing_service=None,
    input_type=control__msgs__pb2._MOVEJCONDREQ,
    output_type=common__msgs__pb2._RESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='MoveLinearAxis',
    full_name='Nrmk.IndyFramework.Control.MoveLinearAxis',
    index=18,
    containing_service=None,
    input_type=control__msgs__pb2._MOVEAXISREQ,
    output_type=common__msgs__pb2._RESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='StopMotion',
    full_name='Nrmk.IndyFramework.Control.StopMotion',
    index=19,
    containing_service=None,
    input_type=common__msgs__pb2._STOPCAT,
    output_type=common__msgs__pb2._RESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='PauseMotion',
    full_name='Nrmk.IndyFramework.Control.PauseMotion',
    index=20,
    containing_service=None,
    input_type=common__msgs__pb2._PAUSECAT,
    output_type=common__msgs__pb2._RESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='SetDirectTeaching',
    full_name='Nrmk.IndyFramework.Control.SetDirectTeaching',
    index=21,
    containing_service=None,
    input_type=common__msgs__pb2._STATE,
    output_type=common__msgs__pb2._RESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='SetSimulationMode',
    full_name='Nrmk.IndyFramework.Control.SetSimulationMode',
    index=22,
    containing_service=None,
    input_type=common__msgs__pb2._STATE,
    output_type=common__msgs__pb2._RESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='SetCustomControlMode',
    full_name='Nrmk.IndyFramework.Control.SetCustomControlMode',
    index=23,
    containing_service=None,
    input_type=common__msgs__pb2._INTMODE,
    output_type=common__msgs__pb2._RESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='GetCustomControlMode',
    full_name='Nrmk.IndyFramework.Control.GetCustomControlMode',
    index=24,
    containing_service=None,
    input_type=common__msgs__pb2._EMPTY,
    output_type=common__msgs__pb2._INTMODE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='SetFrictionCompensation',
    full_name='Nrmk.IndyFramework.Control.SetFrictionCompensation',
    index=25,
    containing_service=None,
    input_type=common__msgs__pb2._STATE,
    output_type=common__msgs__pb2._RESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='GetFrictionCompensationState',
    full_name='Nrmk.IndyFramework.Control.GetFrictionCompensationState',
    index=26,
    containing_service=None,
    input_type=common__msgs__pb2._EMPTY,
    output_type=common__msgs__pb2._STATE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='SetTactTime',
    full_name='Nrmk.IndyFramework.Control.SetTactTime',
    index=27,
    containing_service=None,
    input_type=common__msgs__pb2._TACTTIME,
    output_type=common__msgs__pb2._RESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='GetTactTime',
    full_name='Nrmk.IndyFramework.Control.GetTactTime',
    index=28,
    containing_service=None,
    input_type=common__msgs__pb2._EMPTY,
    output_type=common__msgs__pb2._TACTTIME,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='Recover',
    full_name='Nrmk.IndyFramework.Control.Recover',
    index=29,
    containing_service=None,
    input_type=common__msgs__pb2._EMPTY,
    output_type=common__msgs__pb2._RESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='Reboot',
    full_name='Nrmk.IndyFramework.Control.Reboot',
    index=30,
    containing_service=None,
    input_type=common__msgs__pb2._EMPTY,
    output_type=common__msgs__pb2._RESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='SetManualRecovery',
    full_name='Nrmk.IndyFramework.Control.SetManualRecovery',
    index=31,
    containing_service=None,
    input_type=common__msgs__pb2._STATE,
    output_type=common__msgs__pb2._RESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='MoveRecoverJoint',
    full_name='Nrmk.IndyFramework.Control.MoveRecoverJoint',
    index=32,
    containing_service=None,
    input_type=control__msgs__pb2._TARGETJ,
    output_type=common__msgs__pb2._RESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='SearchProgram',
    full_name='Nrmk.IndyFramework.Control.SearchProgram',
    index=33,
    containing_service=None,
    input_type=control__msgs__pb2._PROGRAM,
    output_type=control__msgs__pb2._PROGRAMINFO,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='PlayProgram',
    full_name='Nrmk.IndyFramework.Control.PlayProgram',
    index=34,
    containing_service=None,
    input_type=control__msgs__pb2._PROGRAM,
    output_type=common__msgs__pb2._RESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='PauseProgram',
    full_name='Nrmk.IndyFramework.Control.PauseProgram',
    index=35,
    containing_service=None,
    input_type=common__msgs__pb2._EMPTY,
    output_type=common__msgs__pb2._RESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='ResumeProgram',
    full_name='Nrmk.IndyFramework.Control.ResumeProgram',
    index=36,
    containing_service=None,
    input_type=common__msgs__pb2._EMPTY,
    output_type=common__msgs__pb2._RESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='StopProgram',
    full_name='Nrmk.IndyFramework.Control.StopProgram',
    index=37,
    containing_service=None,
    input_type=common__msgs__pb2._EMPTY,
    output_type=common__msgs__pb2._RESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='SendAlarm',
    full_name='Nrmk.IndyFramework.Control.SendAlarm',
    index=38,
    containing_service=None,
    input_type=common__msgs__pb2._MESSAGE,
    output_type=common__msgs__pb2._EMPTY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='SendAnnotation',
    full_name='Nrmk.IndyFramework.Control.SendAnnotation',
    index=39,
    containing_service=None,
    input_type=common__msgs__pb2._MESSAGE,
    output_type=common__msgs__pb2._EMPTY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='PlayTuningProgram',
    full_name='Nrmk.IndyFramework.Control.PlayTuningProgram',
    index=40,
    containing_service=None,
    input_type=control__msgs__pb2._TUNINGPROGRAM,
    output_type=config__msgs__pb2._COLLISIONTHRESHOLDS,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='PlayProgramLine',
    full_name='Nrmk.IndyFramework.Control.PlayProgramLine',
    index=41,
    containing_service=None,
    input_type=control__msgs__pb2._PROGRAM,
    output_type=common__msgs__pb2._RESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='SetModbusVariableNameList',
    full_name='Nrmk.IndyFramework.Control.SetModbusVariableNameList',
    index=42,
    containing_service=None,
    input_type=control__msgs__pb2._MODBUSVARIABLELIST,
    output_type=common__msgs__pb2._EMPTY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='SetVariableNameList',
    full_name='Nrmk.IndyFramework.Control.SetVariableNameList',
    index=43,
    containing_service=None,
    input_type=control__msgs__pb2._ALLVARS,
    output_type=common__msgs__pb2._EMPTY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='GetVariableNameList',
    full_name='Nrmk.IndyFramework.Control.GetVariableNameList',
    index=44,
    containing_service=None,
    input_type=common__msgs__pb2._EMPTY,
    output_type=control__msgs__pb2._ALLVARS,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='SetIntVariable',
    full_name='Nrmk.IndyFramework.Control.SetIntVariable',
    index=45,
    containing_service=None,
    input_type=control__msgs__pb2._INTVARS,
    output_type=common__msgs__pb2._EMPTY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='GetIntVariable',
    full_name='Nrmk.IndyFramework.Control.GetIntVariable',
    index=46,
    containing_service=None,
    input_type=common__msgs__pb2._EMPTY,
    output_type=control__msgs__pb2._INTVARS,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='SetModbusVariable',
    full_name='Nrmk.IndyFramework.Control.SetModbusVariable',
    index=47,
    containing_service=None,
    input_type=control__msgs__pb2._MODBUSVARS,
    output_type=common__msgs__pb2._EMPTY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='GetModbusVariable',
    full_name='Nrmk.IndyFramework.Control.GetModbusVariable',
    index=48,
    containing_service=None,
    input_type=common__msgs__pb2._EMPTY,
    output_type=control__msgs__pb2._MODBUSVARS,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='SetBoolVariable',
    full_name='Nrmk.IndyFramework.Control.SetBoolVariable',
    index=49,
    containing_service=None,
    input_type=control__msgs__pb2._BOOLVARS,
    output_type=common__msgs__pb2._EMPTY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='GetBoolVariable',
    full_name='Nrmk.IndyFramework.Control.GetBoolVariable',
    index=50,
    containing_service=None,
    input_type=common__msgs__pb2._EMPTY,
    output_type=control__msgs__pb2._BOOLVARS,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='SetFloatVariable',
    full_name='Nrmk.IndyFramework.Control.SetFloatVariable',
    index=51,
    containing_service=None,
    input_type=control__msgs__pb2._FLOATVARS,
    output_type=common__msgs__pb2._EMPTY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='GetFloatVariable',
    full_name='Nrmk.IndyFramework.Control.GetFloatVariable',
    index=52,
    containing_service=None,
    input_type=common__msgs__pb2._EMPTY,
    output_type=control__msgs__pb2._FLOATVARS,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='SetJPosVariable',
    full_name='Nrmk.IndyFramework.Control.SetJPosVariable',
    index=53,
    containing_service=None,
    input_type=control__msgs__pb2._JPOSVARS,
    output_type=common__msgs__pb2._EMPTY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='GetJPosVariable',
    full_name='Nrmk.IndyFramework.Control.GetJPosVariable',
    index=54,
    containing_service=None,
    input_type=common__msgs__pb2._EMPTY,
    output_type=control__msgs__pb2._JPOSVARS,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='SetTPosVariable',
    full_name='Nrmk.IndyFramework.Control.SetTPosVariable',
    index=55,
    containing_service=None,
    input_type=control__msgs__pb2._TPOSVARS,
    output_type=common__msgs__pb2._EMPTY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='GetTPosVariable',
    full_name='Nrmk.IndyFramework.Control.GetTPosVariable',
    index=56,
    containing_service=None,
    input_type=common__msgs__pb2._EMPTY,
    output_type=control__msgs__pb2._TPOSVARS,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='InverseKinematics',
    full_name='Nrmk.IndyFramework.Control.InverseKinematics',
    index=57,
    containing_service=None,
    input_type=control__msgs__pb2._INVERSEKINEMATICSREQ,
    output_type=control__msgs__pb2._INVERSEKINEMATICSRES,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='ForwardKinematics',
    full_name='Nrmk.IndyFramework.Control.ForwardKinematics',
    index=58,
    containing_service=None,
    input_type=control__msgs__pb2._FORWARDKINEMATICSREQ,
    output_type=control__msgs__pb2._FORWARDKINEMATICSRES,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='CheckAproachRetractValid',
    full_name='Nrmk.IndyFramework.Control.CheckAproachRetractValid',
    index=59,
    containing_service=None,
    input_type=control__msgs__pb2._CHECKAPROACHRETRACTVALIDREQ,
    output_type=control__msgs__pb2._CHECKAPROACHRETRACTVALIDRES,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='GetPalletPointList',
    full_name='Nrmk.IndyFramework.Control.GetPalletPointList',
    index=60,
    containing_service=None,
    input_type=control__msgs__pb2._GETPALLETPOINTLISTREQ,
    output_type=control__msgs__pb2._GETPALLETPOINTLISTRES,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='CalculateRelativePose',
    full_name='Nrmk.IndyFramework.Control.CalculateRelativePose',
    index=61,
    containing_service=None,
    input_type=control__msgs__pb2._CALCULATERELATIVEPOSEREQ,
    output_type=control__msgs__pb2._CALCULATERELATIVEPOSERES,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='CalculateCurrentPoseRel',
    full_name='Nrmk.IndyFramework.Control.CalculateCurrentPoseRel',
    index=62,
    containing_service=None,
    input_type=control__msgs__pb2._CALCULATECURRENTPOSERELREQ,
    output_type=control__msgs__pb2._CALCULATECURRENTPOSERELRES,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='PingFromConty',
    full_name='Nrmk.IndyFramework.Control.PingFromConty',
    index=63,
    containing_service=None,
    input_type=common__msgs__pb2._EMPTY,
    output_type=common__msgs__pb2._EMPTY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='GetTeleOpDevice',
    full_name='Nrmk.IndyFramework.Control.GetTeleOpDevice',
    index=64,
    containing_service=None,
    input_type=common__msgs__pb2._EMPTY,
    output_type=control__msgs__pb2._TELEOPDEVICE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='GetTeleOpState',
    full_name='Nrmk.IndyFramework.Control.GetTeleOpState',
    index=65,
    containing_service=None,
    input_type=common__msgs__pb2._EMPTY,
    output_type=control__msgs__pb2._TELEOPSTATE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='ConnectTeleOpDevice',
    full_name='Nrmk.IndyFramework.Control.ConnectTeleOpDevice',
    index=66,
    containing_service=None,
    input_type=control__msgs__pb2._TELEOPDEVICE,
    output_type=common__msgs__pb2._RESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='DisConnectTeleOpDevice',
    full_name='Nrmk.IndyFramework.Control.DisConnectTeleOpDevice',
    index=67,
    containing_service=None,
    input_type=common__msgs__pb2._EMPTY,
    output_type=common__msgs__pb2._RESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='ReadTeleOpInput',
    full_name='Nrmk.IndyFramework.Control.ReadTeleOpInput',
    index=68,
    containing_service=None,
    input_type=common__msgs__pb2._EMPTY,
    output_type=control__msgs__pb2._TELEP,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='StartTeleOp',
    full_name='Nrmk.IndyFramework.Control.StartTeleOp',
    index=69,
    containing_service=None,
    input_type=control__msgs__pb2._TELEOPSTATE,
    output_type=common__msgs__pb2._RESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='StopTeleOp',
    full_name='Nrmk.IndyFramework.Control.StopTeleOp',
    index=70,
    containing_service=None,
    input_type=common__msgs__pb2._EMPTY,
    output_type=common__msgs__pb2._RESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='SetPlayRate',
    full_name='Nrmk.IndyFramework.Control.SetPlayRate',
    index=71,
    containing_service=None,
    input_type=control__msgs__pb2._TELEPLAYRATE,
    output_type=common__msgs__pb2._RESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='GetPlayRate',
    full_name='Nrmk.IndyFramework.Control.GetPlayRate',
    index=72,
    containing_service=None,
    input_type=common__msgs__pb2._EMPTY,
    output_type=control__msgs__pb2._TELEPLAYRATE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='GetTeleFileList',
    full_name='Nrmk.IndyFramework.Control.GetTeleFileList',
    index=73,
    containing_service=None,
    input_type=common__msgs__pb2._EMPTY,
    output_type=control__msgs__pb2._TELEOPFILELIST,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='SaveTeleMotion',
    full_name='Nrmk.IndyFramework.Control.SaveTeleMotion',
    index=74,
    containing_service=None,
    input_type=control__msgs__pb2._TELEFILEREQ,
    output_type=common__msgs__pb2._RESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='LoadTeleMotion',
    full_name='Nrmk.IndyFramework.Control.LoadTeleMotion',
    index=75,
    containing_service=None,
    input_type=control__msgs__pb2._TELEFILEREQ,
    output_type=common__msgs__pb2._RESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='DeleteTeleMotion',
    full_name='Nrmk.IndyFramework.Control.DeleteTeleMotion',
    index=76,
    containing_service=None,
    input_type=control__msgs__pb2._TELEFILEREQ,
    output_type=common__msgs__pb2._RESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='EnableTeleKey',
    full_name='Nrmk.IndyFramework.Control.EnableTeleKey',
    index=77,
    containing_service=None,
    input_type=common__msgs__pb2._STATE,
    output_type=common__msgs__pb2._RESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='MoveTeleJ',
    full_name='Nrmk.IndyFramework.Control.MoveTeleJ',
    index=78,
    containing_service=None,
    input_type=control__msgs__pb2._MOVETELEJREQ,
    output_type=common__msgs__pb2._RESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='MoveTeleL',
    full_name='Nrmk.IndyFramework.Control.MoveTeleL',
    index=79,
    containing_service=None,
    input_type=control__msgs__pb2._MOVETELELREQ,
    output_type=common__msgs__pb2._RESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='MoveLF',
    full_name='Nrmk.IndyFramework.Control.MoveLF',
    index=80,
    containing_service=None,
    input_type=control__msgs__pb2._MOVELFREQ,
    output_type=common__msgs__pb2._RESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='GetTransformedFTSensorData',
    full_name='Nrmk.IndyFramework.Control.GetTransformedFTSensorData',
    index=81,
    containing_service=None,
    input_type=common__msgs__pb2._EMPTY,
    output_type=control__msgs__pb2._TRANSFORMEDFTSENSORDATA,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_CONTROL)

DESCRIPTOR.services_by_name['Control'] = _CONTROL

# @@protoc_insertion_point(module_scope)
