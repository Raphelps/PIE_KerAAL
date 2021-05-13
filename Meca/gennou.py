from pypot.vrep import from_vrep
from poppy_humanoid import PoppyHumanoid
poppy = PoppyHumanoid(simulator='vrep')
import time
import math
def pause():
    programPause = input("Press the <ENTER> key to continue...")
    return

poppy.l_hip_y.goto_position(-60,2)
poppy.r_hip_y.goto_position(-60,2)

poppy.l_knee_y.goto_position(170,2)
poppy.r_knee_y.goto_position(170,2)

poppy.l_ankle_y.goto_position(65,2)
poppy.r_ankle_y.goto_position(65,2)

pause()
