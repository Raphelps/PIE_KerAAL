from pypot.vrep import from_vrep
from poppy_humanoid import PoppyHumanoid
poppy = PoppyHumanoid(simulator='vrep')
import time
import math
def pause():
    programPause = input("Press the <ENTER> key to continue...")
    return
pause()
poppy.l_hip_x.goto_position(5,2,wait=False)
poppy.r_hip_x.goto_position(-5,2,wait = True)
pause()
#l_arm on hip
poppy.l_shoulder_x.goto_position(45,2,wait= False)
poppy.l_arm_z.goto_position(-80,2,wait=False)
poppy.l_elbow_y.goto_position(-90,2,wait = True)
pause
