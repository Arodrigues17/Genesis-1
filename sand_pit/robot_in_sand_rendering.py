import numpy as np
import torch
import argparse

import genesis as gs

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--vis", action="store_true", default=False)
args = parser.parse_args()

########################## init ##########################
gs.init(
    backend=gs.gpu,
    seed=0,
    precision="32",
    logging_level="debug"
)
########################## create a scene ##########################
scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        res=(1920, 1080),
        camera_pos=(3, -1, 1.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=30,
        max_FPS=60,
    ),
    sim_options=gs.options.SimOptions(
    dt=1e-3,
    substeps=20, 
    ),
    vis_options=gs.options.VisOptions(
        visualize_mpm_boundary=True,
    ),
    mpm_options=gs.options.MPMOptions(
    lower_bound=(0.0, -.5, -0.1),
    upper_bound=(1.0, 0.5, 1.0),
    #particle_count_per_cell=8,  # see if this is actually a thing
    ),
    rigid_options=gs.options.RigidOptions(enable_collision=True),
    renderer=gs.renderers.RayTracer(  # type: ignore
        env_surface=gs.surfaces.Emission(
            emissive_texture=gs.textures.ImageTexture(
                image_path="textures/indoor_bright.png",
            ),
        ),
        env_radius=15.0,
        env_euler=(0, 0, 180),
        lights=[
            {"pos": (1.0, 0.0, 5.0), "radius": 2.0, "color": (15.0, 15.0, 15.0)},
            {"pos": (-1.0, 2.0, 3.0), "radius": 1.0, "color": (10.0, 10.0, 12.0)},
        ],
    ),
    show_viewer=True,
)
########################## entities ##########################
# Floor with metallic surface
plane = scene.add_entity(
    morph=gs.morphs.Plane(),
)

sand = scene.add_entity(
    material=gs.materials.MPM.Sand(),
    morph=gs.morphs.Box(
        pos=(0.65, 0.0, 0.4),
        size=(0.3, 0.3, 0.3),
    ),
    surface=gs.surfaces.Rough(
        color=(1.0, 0.9, 0.6, 1.0),
        vis_mode="particle",
    ),
)
bowl = scene.add_entity(
    morph=gs.morphs.Mesh(
        file="/home/anthony/dev/Genesis/sand_pit/soup_bowl.obj",
        pos=(0.65, 0.0, 0.0),
        scale=0.03,
        fixed=True,
        convexify=False,
        decimate = True,
        #friction=0.8
    ),
    surface=gs.surfaces.Default(
        color=(0.9, 0.8, 0.2, 1.0),
        vis_mode="collision",
    ),
)
# Franka robot with nice materials
franka = scene.add_entity(
    morph=gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
    # Note: The surface will be applied to all parts of the robot
    # For better results, you might want to apply different materials to different links
    surface=gs.surfaces.Iron(
        color=(0.95, 0.95, 0.95),
    ),
)

########################## cameras ##########################

ray_cam = scene.add_camera( #check for interference with regular camera
    res=(1600, 900),
    pos=(2.5, -2.0, 1.5),
    lookat=(0.65, 0.0, 0.3),
    fov=45,
    GUI=True,
    spp=128,  # Samples per pixel (increase for better quality, decrease for faster rendering)
)
########################## build ##########################
scene.build()
########################## Robot Control Logic ##########################
motors_dof = np.arange(7)
fingers_dof = np.arange(7, 9)
# Set control gains
franka.set_dofs_kp(
    np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
)
franka.set_dofs_kv(
    np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
)
franka.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
)
# Get the end-effector link
end_effector = franka.get_link('hand')
# Move to pre-grasp pose
qpos = franka.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.65, 0.0, 0.25]),
    quat=np.array([0, 1, 0, 0]),
)

# Gripper open pos
qpos[-2:] = 0.04
path = franka.plan_path(
    qpos_goal=qpos,
    num_waypoints=200,  # 2s duration
)

# Execute the planned path with ray tracing
step_counter = 0
for waypoint in path:
    franka.control_dofs_position(waypoint)
    scene.step()
    step_counter += 1
    
    # Render with ray tracing every few steps
    if step_counter % 5 == 0:
        ray_cam.render()
# Allow robot to reach the last waypoint
for i in range(100):
    scene.step()
    step_counter += 1
    if step_counter % 5 == 0:
        ray_cam.render()
        
# Reach
qpos = franka.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.65, 0.0, 0.130]),
    quat=np.array([0, 1, 0, 0]),
)
franka.control_dofs_position(qpos[:-2], motors_dof)
for i in range(100):
    scene.step()
    step_counter += 1
    if step_counter % 5 == 0:
        ray_cam.render()
# Grasp
franka.control_dofs_position(qpos[:-2], motors_dof)
franka.control_dofs_force(np.array([-0.5, -0.5]), fingers_dof)
for i in range(100):
    scene.step()
    step_counter += 1
    if step_counter % 5 == 0:
        ray_cam.render()
# Lift
qpos = franka.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.65, 0.0, 0.28]),
    quat=np.array([0, 1, 0, 0]),
)
franka.control_dofs_position(qpos[:-2], motors_dof)

# Render every frame during lift for smoother visuals
for i in range(200):
    scene.step()
    step_counter += 1
    if step_counter % 3 == 0:
        ray_cam.render()