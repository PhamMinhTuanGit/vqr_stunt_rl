"""Isolated, scratch-only POS curriculum with phase-gated skill acquisition."""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from rl_training.tasks.manager_based.locomotion.velocity.mdp import yaw_pos_skill_curriculums as skill
from rl_training.tasks.manager_based.locomotion.velocity.mdp import yaw_pos_skill_dr as dr
from rl_training.tasks.manager_based.locomotion.velocity.mdp import yaw_pos_skill_rewards as telemetry
from rl_training.tasks.manager_based.locomotion.velocity.mdp import wheel_contact_kinematics
from rl_training.tasks.manager_based.locomotion.velocity.mdp.yaw_pos_skill_noise import SkillNoiseCfg
from rl_training.tasks.manager_based.locomotion.velocity.velocity_yaw_env_cfg import EventCfg

from .yaw_env_cfg import VQRWheelYawTerminationsCfg
from .yaw_env_pos_cfg import VQRWheelFlatEnvPOSCfg, VQRWheelYawPosRewardsCfg


@configclass
class VQRWheelYawPosSkillRewardsCfg(VQRWheelYawPosRewardsCfg):
    def __post_init__(self):
        super().__post_init__()
        # The wrapper only records anchor coverage and returns the original reward.
        self.neutral_position.func = telemetry.skill_neutral_position


@configclass
class VQRWheelYawPosSkillCurriculumCfg:
    task_levels = CurrTerm(func=skill.skill_task_levels)


@configclass
class VQRWheelYawPosSkillTerminationsCfg(VQRWheelYawTerminationsCfg):
    skill_promotion_reset = DoneTerm(func=skill.promotion_reset, time_out=True)


@configclass
class VQRWheelYawPosSkillEventsCfg(EventCfg):
    skill_physical_reset = EventTerm(func=dr.reset_physical, mode="reset")
    skill_delay_reset = EventTerm(func=dr.reset_delay, mode="reset")


@configclass
class VQRWheelFlatEnvPOSSkillCfg(VQRWheelFlatEnvPOSCfg):
    rewards: VQRWheelYawPosSkillRewardsCfg = VQRWheelYawPosSkillRewardsCfg()
    curriculum: VQRWheelYawPosSkillCurriculumCfg = VQRWheelYawPosSkillCurriculumCfg()
    terminations: VQRWheelYawPosSkillTerminationsCfg = VQRWheelYawPosSkillTerminationsCfg()
    events: VQRWheelYawPosSkillEventsCfg = VQRWheelYawPosSkillEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Name and function must describe the same eight critic channels.
        # Keep the inherited POS task untouched and make the skill contract explicit.
        critic = self.observations.critic
        if getattr(critic, "contact_velocity", None) is not None:
            raise ValueError("POS skill critic must use rolling_lateral_contact_velocity only.")
        critic.rolling_lateral_contact_velocity.func = wheel_contact_kinematics.rolling_lateral_contact_velocity
        # The five old startup randomizers are replaced by a stage-controlled
        # physical reset; they must not run before Phase A is initialized.
        for name in (
            "randomize_rigid_body_material", "randomize_rigid_body_mass",
            "randomize_rigid_body_mass_base", "randomize_rigid_body_inertia",
            "randomize_com_positions",
        ):
            setattr(self.events, name, None)
        # The first environment reset precedes the runner hook. Start its
        # reset/interval events at nominal settings as well.
        self.events.randomize_apply_external_force_torque.params.update(
            force_range=(0.0, 0.0), torque_range=(0.0, 0.0)
        )
        self.events.randomize_actuator_gains.params.update(
            stiffness_distribution_params=(1.0, 1.0), damping_distribution_params=(1.0, 1.0)
        )
        self.events.randomize_push_robot.params["velocity_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0)
        }
        self.events.randomize_reset_base.params["pose_range"].update(
            roll=(0.0, 0.0), pitch=(0.0, 0.0)
        )
        self.events.randomize_reset_base.params["velocity_range"].update(
            x=(0.0, 0.0), y=(0.0, 0.0), z=(0.0, 0.0),
            roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        )
        policy = self.observations.policy
        policy.base_ang_vel.noise = SkillNoiseCfg(bound=0.4)
        policy.projected_gravity.noise = SkillNoiseCfg(bound=0.1)
        policy.joint_pos.noise = SkillNoiseCfg(bound=0.02, bias_bound=0.1)
        policy.joint_vel.noise = SkillNoiseCfg(bound=3.0, joint_names=tuple(self.joint_names))
