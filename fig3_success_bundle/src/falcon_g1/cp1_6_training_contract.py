"""Simulator-free observation contract shared by the future CP1.6 trainer."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
import numpy as np

from .cp1_policy import HISTORY_LENGTH, OBSERVATION_DIMS, OBSERVATION_ORDER, OBSERVATION_SCALES, POLICY_OBSERVATION_DIM, SINGLE_FRAME_DIM


def training_frame(fields: Mapping[str, np.ndarray]) -> np.ndarray:
    if set(fields) != set(OBSERVATION_ORDER):
        raise ValueError("training observation fields do not match official actor contract")
    pieces=[]
    for name in sorted(fields):
        value=np.asarray(fields[name],dtype=np.float32)
        if value.shape!=(OBSERVATION_DIMS[name],) or not np.isfinite(value).all(): raise ValueError(name)
        pieces.append(value*OBSERVATION_SCALES[name])
    return np.concatenate(pieces).astype(np.float32,copy=False)


@dataclass
class TrainingObservationHistory:
    frames: np.ndarray

    @classmethod
    def from_flat(cls, flat: np.ndarray) -> "TrainingObservationHistory":
        value=np.asarray(flat,dtype=np.float32)
        if value.shape!=(1,POLICY_OBSERVATION_DIM): raise ValueError(value.shape)
        return cls(value.reshape(HISTORY_LENGTH,SINGLE_FRAME_DIM).copy())

    def push(self, fields: Mapping[str,np.ndarray]) -> np.ndarray:
        self.frames[:-1]=self.frames[1:]; self.frames[-1]=training_frame(fields)
        return self.frames.reshape(1,POLICY_OBSERVATION_DIM).astype(np.float32,copy=False)


MODES=("STAND","LOW_SPEED_WALK","SUPPORTED_SPEED_WALK","TURN","TRANSITION")
SPEED_BINS=np.asarray([0.,.05,.10,.15,.20,.25,.30],dtype=np.float32)
YAW_BINS=np.asarray([0.,-.05,.05,-.10,.10,-.15,.15,-.25,.25],dtype=np.float32)
DIRECTIONS=np.asarray([[1,0],[-1,0],[0,1],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1]],dtype=np.float32)
DIRECTIONS[4:]/=np.sqrt(2.)


def sample_commands(rng: np.random.Generator, count: int) -> dict[str,np.ndarray]:
    mode_index=rng.integers(0,len(MODES),size=count); modes=np.asarray(MODES,dtype=object)[mode_index]
    speed=rng.choice(SPEED_BINS,size=count); direction=DIRECTIONS[rng.integers(0,len(DIRECTIONS),size=count)]
    command_xy=direction*speed[:,None]; yaw=rng.choice(YAW_BINS,size=count)
    stand=modes=="STAND"; command_xy[stand]=0.; yaw[stand]=0.
    low=modes=="LOW_SPEED_WALK"; command_xy[low]=direction[low]*rng.choice(SPEED_BINS[1:5],size=low.sum())[:,None]
    supported=modes=="SUPPORTED_SPEED_WALK"; command_xy[supported]=direction[supported]*rng.choice(SPEED_BINS[5:],size=supported.sum())[:,None]
    return {"mode":modes,"stand_flag":stand,"command_xy":command_xy.astype(np.float32),"command_yaw":yaw.astype(np.float32)}


def tracking_reward(error: np.ndarray, sigma: float) -> np.ndarray:
    if sigma<=0: raise ValueError("sigma must be positive and unit-specific")
    value=np.asarray(error,dtype=np.float32); return np.exp(-np.square(value/sigma))


def reward_terms(command_xy,command_yaw,velocity_xy,yaw_rate,*,velocity_sigma=.10,yaw_sigma=.10):
    command_xy=np.asarray(command_xy,dtype=np.float32); velocity_xy=np.asarray(velocity_xy,dtype=np.float32)
    result={"vx_tracking":tracking_reward(velocity_xy[...,0]-command_xy[...,0],velocity_sigma),"vy_tracking":tracking_reward(velocity_xy[...,1]-command_xy[...,1],velocity_sigma),"yaw_tracking":tracking_reward(np.asarray(yaw_rate)-np.asarray(command_yaw),yaw_sigma)}
    result["uncommanded_lateral_penalty"]=-np.abs(velocity_xy[...,1])*(np.abs(command_xy[...,1])<1e-8)
    result["uncommanded_yaw_penalty"]=-np.abs(yaw_rate)*(np.abs(command_yaw)<1e-8)
    return result
