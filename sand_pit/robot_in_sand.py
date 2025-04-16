import argparse
import numpy as np

import genesis as gs

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--vis", action="store_true", default=False)
args = parser.parse_args()
########################## init ##########################
gs.init(seed=0, precision="32", logging_level="debug")

scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=2e-3,
        substeps=10,
    ),
    vis_options=gs.options.VisOptions(
        visualize_mpm_boundary=True,
    ),
    mpm_options=gs.options.MPMOptions(
        lower_bound=(0.0, -.5, -0.1),
        upper_bound=(1.0, 0.5, 1.0),
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(0.8, -3, 1.42),
        camera_lookat=(0.5, 0.5, 0.4),
        camera_fov=30,
        max_FPS=60,
    ),
    show_viewer=True,
)
########################## entities ##########################
plane = scene.add_entity(
    morph=gs.morphs.Plane(),
)
sand = scene.add_entity(
    material=gs.materials.MPM.Sand(),
    morph=gs.morphs.Box(
        pos=(0.5, 0.0, 0.6),
        size=(0.2, 0.2, 0.2),
    ),
    surface=gs.surfaces.Rough(
        color=(1.0, 0.9, 0.6, 1.0),
        vis_mode="particle",
    ),
)
bowl = scene.add_entity(
    morph=gs.morphs.Mesh(
        file="/home/anthony/dev/Genesis/sand_pit/soup_bowl.obj",
        pos=(0.5, 0.0, 0.0),
        scale=0.03,
        fixed=True,
        convexify=False,
    ),
    surface=gs.surfaces.Plastic(
        color=(0.9, 0.8, 0.2, 1.0),
        vis_mode="collision",
    ),
)





franka = scene.add_entity(
gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)
########################## build ##########################
scene.build()

motors_dof = np.arange(7)
fingers_dof = np.arange(7, 9)

# set control gains
# Note: the following values are tuned for achieving best behavior with Franka
# Typically, each new robot would have a different set of parameters.
# Sometimes high-quality URDF or XML file would also provide this and will be parsed.
franka.set_dofs_kp(
    np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
)
franka.set_dofs_kv(
    np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
)
franka.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    np.array([ 87,  87,  87,  87,  12,  12,  12,  100,  100]),
)

# get the end-effector link
end_effector = franka.get_link('hand')

# move to pre-grasp pose
qpos = franka.inverse_kinematics(
    link = end_effector,
    pos  = np.array([0.65, 0.0, 0.25]),
    quat = np.array([0, 1, 0, 0]),
)
# gripper open pos
qpos[-2:] = 0.04
path = franka.plan_path(
    qpos_goal     = qpos,
    num_waypoints = 200, # 2s duration
)
# execute the planned path
for waypoint in path:
    franka.control_dofs_position(waypoint)
    scene.step()

# allow robot to reach the last waypoint
for i in range(100):
    scene.step()
# reach
qpos = franka.inverse_kinematics(
    link = end_effector,
    pos  = np.array([0.65, 0.0, 0.130]),
    quat = np.array([0, 1, 0, 0]),
)
franka.control_dofs_position(qpos[:-2], motors_dof)
for i in range(100):
    scene.step()

# grasp
franka.control_dofs_position(qpos[:-2], motors_dof)
franka.control_dofs_force(np.array([-0.5, -0.5]), fingers_dof)

for i in range(100):
    scene.step()

# lift
qpos = franka.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.65, 0.0, 0.28]),
    quat=np.array([0, 1, 0, 0]),
)
franka.control_dofs_position(qpos[:-2], motors_dof)
for i in range(200):
    scene.step()