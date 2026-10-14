# One overlay slot of a ped's animation layer, as rendered this frame
string slot                  # head | arm | body
string kind                  # look | point | halt | clip, empty = raw overlay
string channel               # wire slot served (arm_l, halt, ...), empty = raw
string phase                 # requested|ramp|hold|transition|release, "" = raw
string clip                  # body clip name, empty otherwise
string animation             # AnimationManager clip the slot plays
float32 playhead             # seconds into that clip
float32 duration             # clip length in seconds
float32 weight               # blend weight after the fade envelope, 0..1
bool loop
