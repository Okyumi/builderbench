# Teleop

We provide two scripts for manually controlling the robot in BuilderBench.

## Keyboard Teleop
``teleop_keyboard.py``

To control the robot end-effector via keyboard.

```bash
python teleop/teleop_keyboard.py --level-id cube-1-task-1
```

**Key bindings:**

| Key | Action |
|-----|--------|
| `Up` / `Down` | forward / back (X axis) |
| `Left` / `Right` | left / right (Y axis) |
| `w` / `s` | up / down (Z axis) |
| `a` / `d` | yaw counterclock-wise / yaw clock-wise |
| `z` / `x` | open gripper / close gripper |
| `End` | reset environment |
| `Esc` | quit |

---

## GUI Teleop
``teleop_gui.py``

To control the robot via a Tkinter GUI with sliders. Each slider sets the target end-effector state directly.

```bash
python teleop/teleop_gui.py --level-id cube-3-task-1
```

**Sliders:**

| Slider | Description |
|--------|-------------|
| `x / y / z` | target end-effector position (meters) |
| `yaw` | target end-effector yaw (radians) |
| `gripper` | gripper opening (0 = closed, 1 = open) |

A "Reset Env" button resets the environment manually. The episode never terminates automatically.

---

## Common options

| Flag | Default | Description |
|------|---------|-------------|
| `--level-id` | `cube-1-task-1` | Task/level to load |
| `--seed` | `0` | Random seed for reset |
