# SO-101 Simulator
Train your own SO-101 robotic arm without the robot!

Code is provided as MIT license. The assets inside `so101` were copied from [https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101), and are provided under the Apache 2.0 license.

## Instructions

1. Install `uv`: [https://docs.astral.sh/uv/getting-started/installation/#installation-methods](https://docs.astral.sh/uv/getting-started/installation/#installation-methods)

2. Run the recording script: `uv run mjpython record.py`.

If you get an error like the below:
```
failed to dlopen path '/Users/user/so-101-simulator/.venv/bin/python3': dlopen(/Users/user/so-101-simulator/.venv/bin/python3, 0x000A): Library not loaded: @executable_path/../lib/libpython3.13.dylib
  Referenced from: <4C4C4410-5555-3144-A1BC-E27CE9ED6B8A> /Users/user/.local/share/uv/python/cpython-3.13.5-macos-aarch64-none/bin/python3.13
  Reason: tried: '/Users/user/so-101-simulator/.venv/lib/python3.13/site-packages/mujoco/MuJoCo_(mjpython).app/Contents/lib/libpython3.13.dylib' (no such file), '/Users/user/so-101-simulator/.venv/bin/../lib/libpython3.13.dylib' (no such file), '/usr/local/lib/libpython3.13.dylib' (no such file), '/usr/lib/libpython3.13.dylib' (no such file, not in dyld cache)
```
Run:
```bash
mkdir -p .venv/lib

ln -sf \
  "$(uv run python -c 'import sys; print(sys.base_prefix)')/lib/libpython3.13.dylib" \
  .venv/lib/libpython3.13.dylib
```

The first time you run it, it may take a minute or two to start.

Once the simulation window runs, you can control it using the following keys:
* Arrow keys: Shoulder pan and lift
* K / L: Elbow flex
* ; / ': Wrist flex
* - / =: Wrist roll
* . / ,: Gripper
* 9: Reset robot targets
* P: Save demonstration and exit
* ESC: Discard demonstration and exit

The key bindings are a bit arbitrary, I know. They were picked to not conflict with MuJoCo's key bindings.

3. After you get the hang of the robot, control it and move the red cube into the box. Once you're done, press `P` to save your movements. The simulation records your movements, as well as two video feeds: one from a camera on the robot arm, the other from a(n invisible) camera on top of the robot.

Repeat this step 20-50x. The more, the better your robot will be.

If you make a mistake and want to delete some episodes, run:
```bash
uv run lerobot-edit-dataset \
    --repo_id local/so101_pick_cube \
    --root ./data/so101_pick_cube \
    --new_root ./data/so101_pick_cube \
    --operation.type delete_episodes \
    --operation.episode_indices "[1, 3, 7, 10]"
```
