import genesis as gs
import numpy as np
gs.init(seed=0, precision="32", backend=gs.cuda)

offset = -.2

scene = gs.Scene(
    show_viewer = True,

    sim_options=gs.options.SimOptions(
    dt=2e-3,
    substeps=10,
    ),
    mpm_options=gs.options.MPMOptions(
        lower_bound=(0.0, -.5, -0.1),
        upper_bound=(1.0, 0.5, 1.0),
    ),

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
        visualize_mpm_boundary=True,
    ),
    renderer=gs.renderers.Rasterizer(),
)

plane = scene.add_entity(
    gs.morphs.Plane(),
)
franka = scene.add_entity(
    gs.morphs.MJCF(file='/home/anthony/dev/Genesis-sim/genesis/assets/xml/franka_emika_panda/panda_shovel.xml'),
)

sand = scene.add_entity(
    material=gs.materials.MPM.Sand(),
    morph=gs.morphs.Box(
        pos=(0.4, 0.0, 0.3),
        size=(0.3, 0.3, 0.3),
    ),
    surface=gs.surfaces.Rough(
        color=(1.0, 0.9, 0.6, 1.0),
        vis_mode="particle",
    ),
)

bowl = scene.add_entity(
    morph=gs.morphs.Mesh(
        file="/home/anthony/dev/Genesis-sim/sand_pit/soup_bowl.obj",
        pos=(0.4, 0.0, 0.0),
        scale=0.03,
        fixed=True,
        convexify=False,
    ),
    surface=gs.surfaces.Plastic(
        color=(0.9, 0.8, 0.2, 1.0),
        vis_mode="collision",
    ),
)
cam = scene.add_camera(
    res    = (1280, 720),
    pos    = (1.5, 1.5, 0.75),
    lookat = (0, 0, 0.5),
    fov    = 60,
    GUI    = False,
)

scene.build()

def execute_digging_motion_with_predefined_positions(franka, scene):
    # Setup joint names and get DOF indices
    
    jnt_names = [
        'joint1',
        'joint2',
        'joint3',
        'joint4',
        'joint5',
        'joint6',
        'joint7'
    ]
    dofs_idx = [franka.get_joint(name).dof_idx_local for name in jnt_names]
    
    # Configure control parameters
    franka.set_dofs_kp(
        kp=np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000]),
        dofs_idx_local=dofs_idx,
    )
    franka.set_dofs_kv(
        kv=np.array([450, 450, 350, 350, 200, 200, 200]),
        dofs_idx_local=dofs_idx,
    )
    franka.set_dofs_force_range(
        lower=np.array([-87, -87, -87, -87, -12, -12, -12]),
        upper=np.array([87, 87, 87, 87, 12, 12, 12]),
        dofs_idx_local=dofs_idx,
    )
    
    # Define digging motion using joint positions
    # These would need to be adjusted based on your robot's configuration
    digging_joint_positions = [
        # Position above digging spot
        np.array([0.0, -0.3, 0.0, -2.0, 0.0, 2.0, 0.8+offset]),
        
        # Insert shovel into material
        np.array([0.0, 0.0, 0.0, -2.2, 0.0, 2.2, 0.8+offset]),
        
        # Scoop motion (tilt the shovel)
        np.array([0.0, 0.0, 0.0, -2.2, 0.0, 1.8, 0.5+offset]),
        
        # Lift filled shovel
        np.array([0.0, -0.5, 0.0, -1.8, 0.0, 1.8, 0.5+offset]),
        
        # Move to dumping location
        np.array([1.0, -0.5, 0.0, -1.8, 0.0, 1.8, 0.5+offset]),
        
        # Empty the shovel (tilt in opposite direction)
        np.array([1.0, -0.5, 0.0, -1.8, 0.0, 2.5, 0.8+offset]),
        
        # Return to starting position
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0+offset])
    ]
    
    # Execute digging sequence
    for waypoint_idx, joint_position in enumerate(digging_joint_positions):
        print(f"Moving to waypoint {waypoint_idx+1}/{len(digging_joint_positions)}")
        
        # Get current joint positions and convert to numpy array
        current_positions_tensor = franka.get_dofs_position(dofs_idx)
        current_positions = current_positions_tensor.cpu().numpy()  # Transfer tensor to CPU first

        
        # Steps for this waypoint
        steps = 100
        
        # Move to target position gradually
        for step in range(steps):
            # Interpolate joint positions
            alpha = (step + 1) / steps
            interp_positions = current_positions + alpha * (joint_position - current_positions)
            
            # Apply joint positions
            franka.control_dofs_position(interp_positions, dofs_idx)
            
            # Step the simulation
            scene.step()
            cam.render()
        
        # Hold at waypoint
        for _ in range(30):
            franka.control_dofs_position(joint_position, dofs_idx)
            scene.step()
            cam.render()
    
    # Return to home position
    home_position = np.array([0, 0, 0, 0, 0, 0, 0])
    for _ in range(100):
        franka.control_dofs_position(home_position, dofs_idx)
        scene.step()
        cam.render()

# Start camera recording (if needed)
cam.start_recording()

# Execute the digging motion
execute_digging_motion_with_predefined_positions(franka, scene)

# Stop camera recording and save to file
cam.stop_recording(save_to_filename='video.mp4', fps=60)