import genesis as gs
import numpy as np

gs.init(seed=0, precision="32", backend=gs.cuda)

scene = gs.Scene(
    show_viewer = True,
    viewer_options = gs.options.ViewerOptions(
        res           = (1280, 960),
        camera_pos    = (1.5, 0.0, 1.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 40,
        max_FPS       = 60,
    ),
    vis_options = gs.options.VisOptions(
        show_world_frame = True,
        world_frame_size = 1.0,
        show_link_frame  = False,
        show_cameras     = False,
        plane_reflection = False,
        ambient_light    = (0.3, 0.3, 0.3),
    ),
    renderer=gs.renderers.Rasterizer(),
)

plane = scene.add_entity(
    gs.morphs.Plane(),
)
franka = scene.add_entity(
    gs.morphs.MJCF(file='/home/anthony/dev/Genesis-sim/genesis/assets/xml/franka_emika_panda/panda_shovel.xml'),
)

cam = scene.add_camera(
    res    = (2560, 1440),
    pos    = (1.5, 1.5, 0.75),
    lookat = (0, 0, 0.5),
    fov    = 60,
    GUI    = False,
)

scene.build()

# render rgb, depth, segmentation, and normal
# rgb, depth, segmentation, normal = cam.render(rgb=True, depth=True, segmentation=True, normal=True)

cam.start_recording()
#cam.set_pose(
#    pos    = (3.0, 3.0, 1.5),
#    lookat = (0, 0, 0.5),
#)

motors_dof = np.arange(7)

jnt_names = [
    'joint1',
    'joint2',
    'joint3',
    'joint4',
    'joint5',
    'joint6',
    'joint7',
]
dofs_idx = [franka.get_joint(name).dof_idx_local for name in jnt_names]

# Hard reset
for i in range(100):
    if i < 100:
        franka.set_dofs_position(np.array([1, 1, 0, 0, 0, 1, 0]), dofs_idx)
    # elif i < 10000:
    #     franka.set_dofs_position(np.array([-1, 0.8, 1, -2, 1, 0.5, -0.5]), dofs_idx)
    # else:
    #     franka.set_dofs_position(np.array([0, 0, 0, 0, 0, 0, 0]), dofs_idx)

    scene.step()
    cam.render()


cam.stop_recording(save_to_filename='video.mp4', fps=60)