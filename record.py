from pathlib import Path
import random
import time

import glfw
import mujoco
import mujoco.viewer
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


# Configuration
MODEL_PATH = Path(__file__).parent / "so101" / "scene.xml"
DATASET_PATH = Path(__file__).parent / "data" / "so101_pick_cube"

REPO_ID = "local/so101_pick_cube"
TASK = "Pick up the red cube and put it inside the box"

FPS = 30
STEP_SIZE = 0.05

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

CUBE_X_RANGE = (0.20, 0.35)
CUBE_Y_RANGE = (-0.15, 0.15)

JOINT_KEYS = {
    "shoulder_pan": (glfw.KEY_RIGHT, glfw.KEY_LEFT),
    "shoulder_lift": (glfw.KEY_UP, glfw.KEY_DOWN),
    "elbow_flex": (glfw.KEY_K, glfw.KEY_L),
    "wrist_flex": (glfw.KEY_SEMICOLON, glfw.KEY_APOSTROPHE),
    "wrist_roll": (glfw.KEY_MINUS, glfw.KEY_EQUAL),
    "gripper": (glfw.KEY_PERIOD, glfw.KEY_COMMA),
}

JOINT_NAMES = list(JOINT_KEYS.keys())


class SO101Simulator:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)

        self.renderer = mujoco.Renderer(
            self.model,
            height=IMAGE_HEIGHT,
            width=IMAGE_WIDTH,
        )

        self.actuators = {}
        self.qpos_addresses = {}
        self.targets = {}

        for name in JOINT_NAMES:
            actuator_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
            )
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)

            if actuator_id < 0 or joint_id < 0:
                raise ValueError(f"Missing actuator or joint: {name}")

            self.actuators[name] = actuator_id
            self.qpos_addresses[name] = self.model.jnt_qposadr[joint_id]

        cube_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free"
        )

        if cube_joint_id < 0:
            raise ValueError("Could not find the cube_free joint.")

        self.cube_qpos_address = self.model.jnt_qposadr[cube_joint_id]

        self.initial_qpos = self.data.qpos.copy()
        self.initial_cube_z = self.initial_qpos[self.cube_qpos_address + 2]

        self.initial_targets = {
            name: self.data.qpos[address]
            for name, address in self.qpos_addresses.items()
        }

        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.initial_qpos

        # Randomize cube position
        self.data.qpos[self.cube_qpos_address] = random.uniform(*CUBE_X_RANGE)
        self.data.qpos[self.cube_qpos_address + 1] = random.uniform(*CUBE_Y_RANGE)
        self.data.qpos[self.cube_qpos_address + 2] = self.initial_cube_z

        self.targets = self.initial_targets.copy()

        for name, target in self.targets.items():
            self.data.ctrl[self.actuators[name]] = target

        mujoco.mj_forward(self.model, self.data)

    def handle_key(self, keycode):
        for name, (positive, negative) in JOINT_KEYS.items():
            if keycode == positive:
                self.move_joint(name, 1)
                return

            if keycode == negative:
                self.move_joint(name, -1)
                return

        if keycode == glfw.KEY_9:
            self.targets = self.initial_targets.copy()

    def move_joint(self, name, direction):
        self.targets[name] += direction * STEP_SIZE

        actuator_id = self.actuators[name]
        lower, upper = self.model.actuator_ctrlrange[actuator_id]

        self.targets[name] = np.clip(self.targets[name], lower, upper)

    def step(self):
        for name, target in self.targets.items():
            self.data.ctrl[self.actuators[name]] = target

        mujoco.mj_step(self.model, self.data)

    def get_observation(self):
        state = np.array(
            [self.data.qpos[self.qpos_addresses[name]] for name in JOINT_NAMES],
            dtype=np.float32,
        )

        self.renderer.update_scene(self.data, camera="top_cam")
        top_image = self.renderer.render()

        self.renderer.update_scene(self.data, camera="wrist_cam")
        wrist_image = self.renderer.render()

        return {
            "observation.state": state,
            "observation.images.top": top_image,
            "observation.images.wrist": wrist_image,
        }

    def get_action(self):
        return np.array(
            [self.targets[name] for name in JOINT_NAMES],
            dtype=np.float32,
        )

    def close(self):
        self.renderer.close()


class DatasetRecorder:
    def __init__(self):
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (len(JOINT_NAMES),),
                "names": JOINT_NAMES,
            },
            "action": {
                "dtype": "float32",
                "shape": (len(JOINT_NAMES),),
                "names": JOINT_NAMES,
            },
            "observation.images.top": {
                "dtype": "video",
                "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.images.wrist": {
                "dtype": "video",
                "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
                "names": ["height", "width", "channels"],
            },
        }

        # Open an existing dataset or create a new one.
        info_path = DATASET_PATH / "meta" / "info.json"

        if info_path.exists():
            self.dataset = LeRobotDataset.resume(
                repo_id=REPO_ID,
                root=DATASET_PATH,
            )
        else:
            self.dataset = LeRobotDataset.create(
                repo_id=REPO_ID,
                root=DATASET_PATH,
                robot_type="so101",
                fps=FPS,
                features=features,
                use_videos=True,
            )

        self.frame_count = 0

    def add_frame(self, observation, action):
        frame = {
            **observation,
            "action": action,
            "task": TASK,
        }

        self.dataset.add_frame(frame)
        self.frame_count += 1

    def save_episode(self):
        if self.frame_count == 0:
            print("No frames recorded. Nothing to save.")
            return False

        self.dataset.save_episode()
        print(f"Episode saved! Total episodes: {self.dataset.num_episodes}")
        return True

    def discard_episode(self):
        self.dataset.clear_episode_buffer()
        print("Demonstration discarded.")

    def close(self):
        self.dataset.finalize()


def main():
    simulator = SO101Simulator()
    recorder = DatasetRecorder()

    # The outcome is either "save", "discard", or None.
    outcome = {"value": None}

    def key_callback(keycode):
        if keycode == glfw.KEY_P:
            outcome["value"] = "save"

        elif keycode == glfw.KEY_ESCAPE:
            outcome["value"] = "discard"

        else:
            simulator.handle_key(keycode)

    frame_period = 1.0 / FPS
    frame_accumulator = 0.0

    print("SO-101 Demonstration Recorder")
    print("-----------------------------")
    print(f"Task: {TASK}")
    print(f"Dataset: {DATASET_PATH}")
    print()
    print("ENTER the workspace and start teleoperating right away.")
    print()
    print("Arrow keys: Shoulder pan and lift")
    print("K / L: Elbow flex")
    print("; / ': Wrist flex")
    print("- / =: Wrist roll")
    print(". / ,: Gripper")
    print("0: Reset robot targets")
    print()
    print("P: Save demonstration and exit")
    print("ESC: Discard demonstration and exit")
    print()

    try:
        with mujoco.viewer.launch_passive(
            simulator.model,
            simulator.data,
            key_callback=key_callback,
        ) as viewer:
            while viewer.is_running() and outcome["value"] is None:
                with viewer.lock():
                    simulator.step()

                    frame_accumulator += simulator.model.opt.timestep

                    if frame_accumulator >= frame_period:
                        frame_accumulator -= frame_period

                        observation = simulator.get_observation()
                        action = simulator.get_action()

                        recorder.add_frame(observation, action)

                viewer.sync()
                time.sleep(simulator.model.opt.timestep)

        if outcome["value"] == "save":
            recorder.save_episode()
        else:
            recorder.discard_episode()

    except KeyboardInterrupt:
        print("Interrupted. Discarding demonstration.")
        recorder.discard_episode()

    finally:
        recorder.close()
        simulator.close()

        print("Recording finished.")
        print(f"Saved episodes: {recorder.dataset.num_episodes}")
        print(f"Dataset location: {DATASET_PATH}")


if __name__ == "__main__":
    main()
