import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import DCMotorCfg, DelayedPDActuatorCfg
import os

AmazingCfg = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.getcwd() + "/source/amazing/amazing/assets/15_Test1.usd",
        # usd_path="/home/rl/jacksit/newton/RmLab/source/RmLab/RmLab/assets/dogleg/dogleg4/dogleg_converter/dogleg4.usda",
        scale=(1.0, 1.0, 1.0),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,       # for simulating collision of gimbal and dogleg
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
            fix_root_link=False,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(
        #     mesh_collision_property=sim_utils.ConvexDecompositionPropertiesCfg(),
        #     collision_enabled=True,
        # )
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "wheel": DCMotorCfg(    # velocity contorl
            joint_names_expr=[".*_wheel"],
            velocity_limit_sim=1e9,
            effort_limit_sim=1e9,
            stiffness={
                ".*": 0.0,      # set to zero for using velocity control
            },
            effort_limit={
                ".*": 40.0,
            },
            saturation_effort=60.0,
            velocity_limit=20.0,   # TODO: need check for real velocity
            damping={
                ".*": 100.0,
            },
            armature=0.01,
            friction=0.0,
        ),
        "calf": DelayedPDActuatorCfg(    # position contorl
            joint_names_expr=[".*_calf"],
            velocity_limit_sim=1e9,
            effort_limit_sim=1e9,
            stiffness={
                ".*": 70.0,      # set to zero for using velocity control
            },
            effort_limit={
                ".*": 40.0,
            },
            velocity_limit=20.0,   # TODO: need check for real velocity
            min_delay=0,  # physics time steps (min: 5.0 * 0 = 0.0ms)
            max_delay=4,  # physics time steps (max: 5.0 * 4 = 20.0ms)
            damping={
                ".*": 0.7,
            },
            armature=0.01,
            friction=0.0,
        ),
        "thigh": DelayedPDActuatorCfg(    # position contorl
            joint_names_expr=[".*_thigh"],
            velocity_limit_sim=1e9,
            effort_limit_sim=1e9,
            stiffness={
                ".*": 70.0,      # set to zero for using velocity control
            },
            effort_limit={
                ".*": 40.0,
            },
            velocity_limit=20.0,   # TODO: need check for real velocity
            min_delay=0,  # physics time steps (min: 5.0 * 0 = 0.0ms)
            max_delay=4,  # physics time steps (max: 5.0 * 4 = 20.0ms)
            damping={
                ".*": 0.7,
            },
            armature=0.01,
            friction=0.0,
        ),
        "abd": DelayedPDActuatorCfg(    # position contorl
            joint_names_expr=[".*_abd"],
            velocity_limit_sim=1e9,
            effort_limit_sim=1e9,
            stiffness={
                ".*": 70.0,      # set to zero for using velocity control
            },
            effort_limit={
                ".*": 40.0,
            },
            velocity_limit=20.0,   # TODO: need check for real velocity
            min_delay=0,  # physics time steps (min: 5.0 * 0 = 0.0ms)
            max_delay=4,  # physics time steps (max: 5.0 * 4 = 20.0ms)
            damping={
                ".*": 0.7,
            },
            armature=0.01,
            friction=0.0,
        ),
        "arm": DelayedPDActuatorCfg(    # position contorl
            joint_names_expr=["arm_.*"],
            velocity_limit_sim=1e9,
            effort_limit_sim=1e9,
            stiffness={
                ".*": 70.0,      # set to zero for using velocity control
            },
            effort_limit={
                ".*": 40.0,
            },
            velocity_limit=5.0,   # TODO: need check for real velocity
            min_delay=0,  # physics time steps (min: 5.0 * 0 = 0.0ms)
            max_delay=4,  # physics time steps (max: 5.0 * 4 = 20.0ms)
            damping={
                ".*": 5.0,
            },
            armature=0.01,
            friction=0.0,
        ),
    },
)