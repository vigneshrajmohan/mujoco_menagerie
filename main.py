from dataclasses import dataclass
import numpy as np
from pathlib import Path
import enum
from tqdm import tqdm
import mujoco
import imageio

class Resolution(enum.Enum):
    SD = (480, 640)
    HD = (720, 1280)
    UHD = (2160, 3840)

def quartic(t: float) -> float:
    return 0 if abs(t) > 1 else (1 - t**2) ** 2

def blend_coef(t: float, duration: float, std: float) -> float:
    normalised_time = 2 * t / duration - 1
    return quartic(normalised_time / std)

def unit_smooth(normalised_time: float) -> float:
    return 1 - np.cos(normalised_time * 2 * np.pi)

def azimuth(time: float, duration: float, total_rotation: float, offset: float) -> float:
    return offset + unit_smooth(time / duration) * total_rotation

def save_frames_as_video(frames, filename='output.mp4', fps=60):
    writer = imageio.get_writer(filename, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

# Main script starts here
res = Resolution.SD
fps = 60
duration = 10.0
ctrl_rate = 2
ctrl_std = 0.05
total_rot = 60
blend_std = .8

h, w = res.value

model_dir = Path("universal_robots_ur5e")
model_xml = model_dir / "scene.xml"

model = mujoco.MjModel.from_xml_path(str(model_xml))
data = mujoco.MjData(model)

model.vis.global_.offheight = h
model.vis.global_.offwidth = w

renderer = mujoco.Renderer(model, height=h, width=w)

mujoco.mj_forward(model, data)
renderer.update_scene(data)
# Removed media.show_image(renderer.render())

frames = []

np.random.seed(12345)

vis = mujoco.MjvOption()
vis.geomgroup[2] = True
vis.geomgroup[3] = False

coll = mujoco.MjvOption()
coll.geomgroup[2] = False
coll.geomgroup[3] = True
coll.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = True

camera = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(model, camera)
camera.distance = 1
offset = model.vis.global_.azimuth

nsteps = int(np.ceil(duration / model.opt.timestep))
perturb = np.random.randn(nsteps, model.nu)
width = int(nsteps * ctrl_rate / duration)
kernel = np.exp(-0.5 * np.linspace(-3, 3, width) ** 2)
kernel /= np.linalg.norm(kernel)

for i in range(model.nu):
    perturb[:, i] = np.convolve(perturb[:, i], kernel, mode="same")

if model.nkey > 0:
    mujoco.mj_resetDataKeyframe(model, data, 0)
    ctrl0 = data.ctrl.copy()
else:
    mujoco.mj_resetData(model, data)
    ctrl0 = np.mean(model.actuator_ctrlrange, axis=1)

# Define control logic to reach the highest point
def get_high_point_control(model, data):
    # This is a placeholder function.
    # You need to define the actual control logic based on your robot's kinematics.
    high_point_ctrl = np.zeros(model.nu)  # Replace with actual control values for high point
    return high_point_ctrl

def get_home_position_control(model):
    # Assuming home position corresponds to zero control input
    return np.zeros(model.nu)

half_nsteps = nsteps // 2
for i in tqdm(range(nsteps), desc="Simulating"):
    if i < half_nsteps:  # Going to the high point in the first half of the simulation
        # Interpolate controls from home position to high point
        interp_factor = i / float(half_nsteps)
        data.ctrl[:] = ctrl0 * (1 - interp_factor) + get_high_point_control(model, data) * interp_factor
    else:  # Returning to home position in the second half
        # Interpolate controls from high point back to home position
        interp_factor = (i - half_nsteps) / float(half_nsteps)
        data.ctrl[:] = get_high_point_control(model, data) * (1 - interp_factor) + get_home_position_control(model) * interp_factor

    mujoco.mj_step(model, data)
    
    if len(frames) < data.time * fps:
        camera.azimuth = azimuth(data.time, duration, total_rot, offset)
        renderer.update_scene(data, camera, scene_option=vis)
        vispix = renderer.render().copy().astype(np.float32)
        
        renderer.update_scene(data, camera, scene_option=coll)
        collpix = renderer.render().copy().astype(np.float32)
        
        b = blend_coef(data.time, duration, blend_std)
        frame = (1 - b) * vispix + b * collpix
        frame = frame.astype(np.uint8)
        frames.append(frame)

# Save the video
save_frames_as_video(frames, filename='simulation_video.mp4', fps=fps)
