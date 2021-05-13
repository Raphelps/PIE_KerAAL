from pypot.vrep import from_vrep
from poppy_humanoid import PoppyHumanoid
poppy = PoppyHumanoid(simulator='vrep')
import time
import math
def pause():
    programPause = input("Press the <ENTER> key to continue...")
    return
#poppy.r_knee_y.goto_position(90,2,wait=False)
#poppy.l_knee_y.goto_position(90,2,wait=False)

poppy.r_hip_y.goto_position(-90,2,wait=False)
poppy.l_hip_y.goto_position(-90,2,wait=True)
pause()
