import json
import asyncio
from mavsdk import System
from mavsdk.mission import MissionItem, MissionPlan

'''
PART 1 —— MAVLink / PX4

'''


def load_waypoints(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        wp = json.load(f)
    return wp

async def upload_mission(json_path="uav_waypoints.json"):
    drone = System()
    print("Connecting to PX4...")
    await drone.connect(system_address="udp://:14540")  # SITL 

    print("Waiting for connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-- Connected to drone!")
            break

    # 读取航点
    wps = load_waypoints(json_path)

    mission_items = []
    for i, wp in enumerate(wps):
        mission_items.append(
            MissionItem(
                wp["x"],     # NED-X (North)
                wp["y"],     # NED-Y (East)
                -wp["z"],    # NED-Z (Up=negative, so negation)
                3.0,         # speed m/s
                True,        # is_fly_through
                float('nan'),
                float('nan'),
                wp["yaw"],   # yaw angle
                MissionItem.CameraAction.NONE,
                0.0, 0.0, 0.0
            )
        )

    mission_plan = MissionPlan(mission_items)

    print("Uploading mission...")
    await drone.mission.upload_mission(mission_plan)
    print("Mission uploaded!")

    print("Arming...")
    await drone.action.arm()

    print("Starting mission...")
    await drone.mission.start_mission()

if __name__ == "__main__":
    asyncio.run(upload_mission("uav_waypoints.json"))
