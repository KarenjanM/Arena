# Exact name as published on arena_peds.
string pedestrian_name

# attend | ready | reach | take | retract
string phase

# Object centre in the pedestrian stream's map frame.
geometry_msgs/Point at

# Profile key, used by ready, reach and take.
string object_kind
