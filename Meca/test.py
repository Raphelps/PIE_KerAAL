from pypot.vrep import from_vrep
from poppy_humanoid import PoppyHumanoid
poppy = PoppyHumanoid(simulator='vrep')
import time
import math
def pause():
    programPause = input("Press the <ENTER> key to continue...")
    return
poppy.head_z.goal_position = 90

pause()
for m in poppy.l_arm:
    m.goal_position = 30 
pause()

amp = 30 #amplitude en degre
freq = 0.5 #frequence dans hertz

t0 = time.time()
while True:
    t = time.time()
    if t-t0>10 :
        break
    poppy.head_z.goal_position = amp*math.sin(2*3.14*freq*t)
    time.sleep(0.05)

t0 = time.time()
goal = -90
p0 = poppy.r_shoulder_y.present_position

m = (goal - p0)*freq
new_target = p0

while  True:
    t1 = time.time()
    t = t1-t0
    new_target = m*t +p0
    if abs(new_target - goal)< 3:
        poppy.r_shoulder_y.goal_position = goal
        break
    poppy.r_shoulder_y.goal_position = new_target
    time.sleep(0.05)
