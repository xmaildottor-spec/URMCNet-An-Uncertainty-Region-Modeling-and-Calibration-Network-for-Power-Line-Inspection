from pymavlink import mavutil
import json
import time

'''
PART 2 —— ArduPilot：pymavlink Mission Upload
'''


def load_waypoints(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def upload_to_ardupilot(json_path="uav_waypoints.json"):
    # 连接（SITL）
    master = mavutil.mavlink_connection("udp:127.0.0.1:14550")
    master.wait_heartbeat()
    print("Heartbeat Received. Connected to ArduPilot")

    wps = load_waypoints(json_path)

    master.waypoint_clear_all_send()
    time.sleep(1)

    master.waypoint_count_send(len(wps))

    for i, wp in enumerate(wps):
        frame = mavutil.mavlink.MAV_FRAME_LOCAL_NED
        master.mav.mission_item_send(
            master.target_system,
            master.target_component,
            i,
            frame,
            mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            0, 0,
            0, 0, 0,
            wp["x"],    # North
            wp["y"],    # East
            -wp["z"]    # Altitude (Up positive)
        )
        print(f"Sent waypoint {i}")
        time.sleep(0.1)

    print("Mission upload complete!")

if __name__ == "__main__":
    upload_to_ardupilot("uav_waypoints.json")
