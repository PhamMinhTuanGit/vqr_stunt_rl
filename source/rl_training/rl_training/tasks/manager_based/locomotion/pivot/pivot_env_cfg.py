from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from ....velocity import mdp
from .flat_env_cfg import VQRWheelFlatEnvCfg     # ★ đổi đúng tên class flat của đồng nghiệp bạn

# ★ SỬA TÊN KHỚP / BODY Ở ĐÂY NẾU VQR KHÁC M20 --------------------
FRONT_WHEELS = ["fl_wheel", "fr_wheel"]
REAR_WHEELS  = ["hl_wheel", "hr_wheel"]
FRONT_ARM_J  = ["fl_hipy", "fr_hipy", "fl_knee", "fr_knee"]
ALL_WHEEL_J  = [".*_wheel"]
BASE_LINK    = "base_link"
# -----------------------------------------------------------------


@configclass
class VQRWheelPivotEnvCfg(VQRWheelFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 20.0
        self.scene.num_envs = 4096
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        if getattr(self.scene, "height_scanner", None) is not None:
            self.scene.height_scanner = None
            self.observations.policy.height_scan = None

        # ---------------- COMMANDS ----------------
        self.commands.base_velocity = None
        self.commands.pivot = mdp.PivotCommandCfg(
            asset_name="robot",
            resampling_time_range=(6.0, 10.0),
            ang_vel_z=(0.0, 0.0),          # curriculum sẽ mở rộng
            pitch_target=(0.9, 1.2),
            gesture_freq=(0.5, 1.5),
            rel_standing_envs=0.2,
            debug_vis=False,
        )

        # ---------------- OBSERVATIONS ----------------
        P = self.observations.policy
        P.velocity_commands = ObsTerm(func=mdp.generated_commands,
                                      params={"command_name": "pivot"})
        P.base_lin_vel = None                        # khó ước lượng khi wheelie, bỏ khỏi actor
        P.base_ang_vel.scale = 0.1
        P.wheel_contact = ObsTerm(
            func=mdp.contact_forces_binary if hasattr(mdp, "contact_forces_binary") else mdp.undesired_contacts,
            params={"sensor_cfg": SceneEntityCfg("contact_forces",
                                                 body_names=FRONT_WHEELS + REAR_WHEELS),
                    "threshold": 1.0},
        )
        P.enable_corruption = True

        # ---------------- REWARDS ----------------
        R = self.rewards
        for t in ["track_lin_vel_xy_exp", "track_ang_vel_z_exp", "feet_air_time",
                  "flat_orientation_l2", "base_height_l2", "lin_vel_z_l2", "ang_vel_xy_l2"]:
            if hasattr(R, t):
                setattr(R, t, None)

        R.pivot_pitch = RewTerm(func=mdp.pivot_pitch_tracking, weight=2.0,
                                params={"std": 0.25, "command_name": "pivot"})
        R.front_lift  = RewTerm(func=mdp.front_wheels_height, weight=1.5,
                                params={"target_h": 0.45,
                                        "asset_cfg": SceneEntityCfg("robot", body_names=FRONT_WHEELS)})
        R.rear_only   = RewTerm(func=mdp.rear_only_contact, weight=0.5,
                                params={"front_cfg": SceneEntityCfg("contact_forces", body_names=FRONT_WHEELS),
                                        "rear_cfg":  SceneEntityCfg("contact_forces", body_names=REAR_WHEELS)})
        R.pivot_yaw   = RewTerm(func=mdp.pivot_yaw_tracking, weight=1.5,
                                params={"std": 0.6, "command_name": "pivot"})
        R.stay        = RewTerm(func=mdp.stay_in_place, weight=1.0, params={"std": 0.35})
        R.roll_flat   = RewTerm(func=mdp.flat_roll_only, weight=-2.0)
        R.alive       = RewTerm(func=mdp.is_alive, weight=1.0)

        R.gesture     = RewTerm(func=mdp.gesture_tracking, weight=0.0,   # bật ở stage 3
                                params={"std": 0.5, "amp": 0.6, "command_name": "pivot",
                                        "asset_cfg": SceneEntityCfg("robot", joint_names=FRONT_ARM_J)})
        R.show_spin   = RewTerm(func=mdp.front_wheel_spin, weight=0.0,   # bật ở stage 3
                                params={"target_speed": 12.0,
                                        "asset_cfg": SceneEntityCfg("robot", joint_names=FRONT_WHEELS)})

        R.action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
        R.dof_acc_l2     = RewTerm(func=mdp.joint_acc_l2,   weight=-2.5e-7)
        R.dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
        R.undesired_contacts = RewTerm(
            func=mdp.undesired_contacts, weight=-1.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces",
                                                 body_names=[BASE_LINK, ".*_knee"]), "threshold": 1.0})

        # ---------------- TERMINATIONS ----------------
        self.terminations.base_contact = DoneTerm(
            func=mdp.illegal_contact, params={"threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[BASE_LINK])})
        self.terminations.bad_roll = DoneTerm(
            func=mdp.bad_orientation, params={"limit_angle": 0.6})   # nếu term này tính cả pitch -> dùng hàm riêng
        self.terminations.pitch_collapsed = DoneTerm(
            func=mdp.pitch_collapsed, params={"min_pitch": 0.3, "grace_s": 1.5})
        self.terminations.drift = DoneTerm(func=mdp.drifted_away, params={"max_dist": 1.2})

        # ---------------- EVENTS ----------------
        # ★ QUAN TRỌNG NHẤT: reset 50% env thẳng vào tư thế wheelie
        self.events.reset_base.params["pose_range"]["pitch"] = (0.0, 1.1)
        self.events.reset_base.params["pose_range"]["yaw"]   = (-3.14, 3.14)
        self.events.reset_base.params["velocity_range"] = {k: (0.0, 0.0) for k in
                                                           ["x","y","z","roll","pitch","yaw"]}
        self.events.push_robot.params["velocity_range"] = {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}
        self.events.push_robot.interval_range_s = (8.0, 12.0)
        self.events.physics_material.params["static_friction_range"]  = (0.5, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.4, 1.0)
        self.events.add_base_mass.params["mass_distribution_params"]  = (-2.0, 2.0)

        # ---------------- CURRICULUM ----------------
        self.curriculum.pivot_speed = CurrTerm(func=mdp.pivot_yaw_curriculum,
                                               params={"command_name": "pivot",
                                                       "max_yaw": 3.0, "step": 0.25})


@configclass
class VQRWheelPivotEnvCfg_PLAY(VQRWheelPivotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.commands.pivot.ang_vel_z = (-3.0, 3.0)