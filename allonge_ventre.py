from pypot.vrep import from_vrep
from poppy_humanoid import PoppyHumanoid
poppy = PoppyHumanoid(simulator='vrep')

def pause():
    programPause = input("Press the <ENTER> key to continue...")
    return

poppy.l_shoulder_y.goal_position = -90.
poppy.r_shoulder_y.goal_position = -90.
pause()
poppy.bust_y.goal_position = 20.
poppy.abs_y.goal_position = 20.
pause()
poppy.l_knee_y.goal_position = 100.
poppy.r_knee_y.goal_position = 100.
poppy.r_hip_y.goal_position = -30.
poppy.l_hip_y.goal_position = -30.
pause()
poppy.bust_y.goal_position = 20.
poppy.abs_y.goal_position = 20.
pause()
poppy.l_shoulder_y.goal_position = -60.
poppy.r_shoulder_y.goal_position = -60.
poppy.l_knee_y.goal_position = 70.
poppy.r_knee_y.goal_position = 70.
poppy.r_hip_y.goal_position = -20.
poppy.l_hip_y.goal_position = -20.
pause()
poppy.l_shoulder_y.goal_position = 0.
poppy.r_shoulder_y.goal_position = 0.
poppy.l_knee_y.goal_position = 0.
poppy.r_knee_y.goal_position = 0.
poppy.r_hip_y.goal_position = 0.
poppy.l_hip_y.goal_position = 0.
pause()

poppy.reset_simulation()