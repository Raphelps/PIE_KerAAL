from pypot.vrep import from_vrep
from poppy_humanoid import PoppyHumanoid
poppy = PoppyHumanoid(simulator='vrep')

def pause():
    programPause = input("Press the <ENTER> key to continue...")
    return
poppy.bust_y.goal_position = 20.
pause()
poppy.abs_y.goal_position = 20.

#a = get_moving_speed(36)
#print(a)


#for m in poppy.l_arm:
#    m.goal_position = 30. 
#pause()
#for m in poppy.r_arm:
#    m.goal_position = -30. 

pause()
poppy.r_knee_y.goal_position = 40.
poppy.l_knee_y.goal_position = 40.
poppy.r_hip_y.goal_position = -40.
poppy.l_hip_y.goal_position = -40.
pause()
poppy.r_hip_y.goal_position = -60.
poppy.l_hip_y.goal_position = -60.
pause()
poppy.r_hip_y.goal_position = -90.
poppy.l_hip_y.goal_position = -90.
poppy.r_knee_y.goal_position = 0.
poppy.l_knee_y.goal_position = 0.
poppy.bust_y.goal_position = 0.
poppy.abs_y.goal_position = 0.
pause()

poppy.reset_simulation()
