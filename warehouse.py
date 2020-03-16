import typing

import numpy as np
import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import Box2D
from Box2D.b2 import (
    world,
    circleShape,
    polygonShape,
    dynamicBody,
)

# Engineering notes:
#   - Each agent is identified by int[0, NUM_AGENTS) in string type.
#   - Zero coordinate for the env is on the bottom left, this is then transformed to
#     top left for rendering.
#
# |B|x| | |x|x| | |x|x| | |x|x| | |x|B|

# Environment
AREA_DIMENSION_M: float = 16.0
BORDER_WIDTH_M: float = 1.0
WORLD_DIMENSION_M: float = AREA_DIMENSION_M + 2 * BORDER_WIDTH_M
PICKUP_RACKS_ARRANGEMENT: typing.List[float] = [5.0, 9.0, 13.0]
FRAMES_PER_SECOND: int = 20

NUM_AGENTS: int = (len(PICKUP_RACKS_ARRANGEMENT) + 1) ** 2
DELIVERY_QUEUE_LEN: int = 20

# Rendering
B2_VEL_ITERS: int = 10
B2_POS_ITERS: int = 10
PIXELS_PER_METER: int = 30
VIEWPORT_DIMENSION_PX: int = int(WORLD_DIMENSION_M) * PIXELS_PER_METER

AGENT_COLOR: typing.Tuple[float, float, float] = (0.0, 0.0, 0.0)
BORDER_COLOR: typing.Tuple[float, float, float] = (0.5, 0.5, 0.5)
PICKUP_POINT_COLOR: typing.Tuple[float, float, float] = (0.8, 0.8, 0.8)
DELIVERY_POINT_COLOR: typing.Tuple[float, float, float] = (0.8, 0.8, 0.8)


class Warehouse(MultiAgentEnv):
    def __init__(self) -> None:
        super(Warehouse, self).__init__()

        self.metadata = {
            "render.modes": ["human"],
            "video.frames_per_second": FRAMES_PER_SECOND,
        }
        self.reward_range = (-np.inf, -np.inf)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "self_position": gym.spaces.Box(
                    low=-AREA_DIMENSION_M, high=AREA_DIMENSION_M, shape=(2,), dtype=np.float32,
                ),
                "pickup_positions": gym.spaces.Box(
                    low=-AREA_DIMENSION_M,
                    high=AREA_DIMENSION_M,
                    shape=(DELIVERY_QUEUE_LEN, 2),
                    dtype=np.float32,
                ),
                "delivery_positions": gym.spaces.Box(
                    low=-AREA_DIMENSION_M,
                    high=AREA_DIMENSION_M,
                    shape=(DELIVERY_QUEUE_LEN, 2),
                    dtype=np.float32,
                ),
            }
        )

        self._viewer: gym.Viewer = None
        self._world = world(gravity=(0, 0), doSleep=False)
        self._agent_bodies: typing.List[dynamicBody] = []
        self._border_bodies: typing.List[dynamicBody] = []
        self._pickup_points: np.ndarray = None
        self._delivery_points: np.ndarray = None

    def reset(self) -> typing.Dict[str, gym.spaces.Dict]:
        self._agent_bodies = []
        racks_diff = (PICKUP_RACKS_ARRANGEMENT[1] - PICKUP_RACKS_ARRANGEMENT[0]) / 2
        arrangement = [
            PICKUP_RACKS_ARRANGEMENT[0] - racks_diff,
            *[x + racks_diff for x in PICKUP_RACKS_ARRANGEMENT],
        ]
        for x in arrangement:
            for y in arrangement:
                body = self._world.CreateDynamicBody(position=(x, y))
                _ = body.CreateCircleFixture(radius=0.3, density=1.0, friction=0.0)
                self._agent_bodies.append(body)

        self._border_bodies = [
            self._world.CreateStaticBody(
                position=(WORLD_DIMENSION_M / 2, BORDER_WIDTH_M / 2),
                shapes=polygonShape(box=(WORLD_DIMENSION_M / 2, BORDER_WIDTH_M / 2)),
            ),
            self._world.CreateStaticBody(
                position=(WORLD_DIMENSION_M / 2, WORLD_DIMENSION_M - BORDER_WIDTH_M / 2,),
                shapes=polygonShape(box=(WORLD_DIMENSION_M / 2, BORDER_WIDTH_M / 2)),
            ),
            self._world.CreateStaticBody(
                position=(BORDER_WIDTH_M / 2, WORLD_DIMENSION_M / 2,),
                shapes=polygonShape(box=(BORDER_WIDTH_M / 2, WORLD_DIMENSION_M / 2)),
            ),
            self._world.CreateStaticBody(
                position=(WORLD_DIMENSION_M - BORDER_WIDTH_M / 2, WORLD_DIMENSION_M / 2,),
                shapes=polygonShape(box=(BORDER_WIDTH_M / 2, WORLD_DIMENSION_M / 2)),
            ),
        ]

        pickup_points = []
        for x in PICKUP_RACKS_ARRANGEMENT:
            for y in PICKUP_RACKS_ARRANGEMENT:
                pickup_points.extend(
                    [
                        [x - 0.5, y - 0.5],
                        [x + 0.5, y - 0.5],
                        [x + 0.5, y + 0.5],
                        [x - 0.5, y + 0.5],
                    ]
                )
        self._pickup_points = np.array(pickup_points, dtype=np.float32)

        delivery_points = []
        for val in range(int(AREA_DIMENSION_M)):
            delivery_points.extend(
                [
                    [BORDER_WIDTH_M + val + 0.5, BORDER_WIDTH_M + 0.5],
                    [BORDER_WIDTH_M + val + 0.5, WORLD_DIMENSION_M - BORDER_WIDTH_M - 0.5,],
                    [BORDER_WIDTH_M + 0.5, BORDER_WIDTH_M + val + 0.5],
                    [WORLD_DIMENSION_M - BORDER_WIDTH_M - 0.5, BORDER_WIDTH_M + val + 0.5,],
                ]
            )
        self._delivery_points = np.array(delivery_points, dtype=np.float32)

        return {
            str(i): {
                "self_position": np.array(self._agent_bodies[i].position, dtype=np.float32),
                "pickup_positions": np.zeros((DELIVERY_QUEUE_LEN, 2)),
                "delivery_positions": np.zeros((DELIVERY_QUEUE_LEN, 2)),
            }
            for i in range(NUM_AGENTS)
        }

    def step(
        self, action_dict: typing.Dict[str, np.ndarray]
    ) -> typing.Tuple[
        typing.Dict[str, gym.spaces.Dict],
        typing.Dict[str, float],
        typing.Dict[str, bool],
        typing.Dict[str, typing.Dict[str, str]],
    ]:
        for key, value in action_dict.items():
            self._agent_bodies[int(key)].linearVelocity = value.tolist()

        self._world.Step(1.0 / FRAMES_PER_SECOND, 10, 10)

        observations = {
            str(i): {
                "self_position": np.array(self._agent_bodies[i].position, dtype=np.float32),
                "pickup_positions": np.zeros((DELIVERY_QUEUE_LEN, 2)),
                "delivery_positions": np.zeros((DELIVERY_QUEUE_LEN, 2)),
            }
            for i in range(NUM_AGENTS)
        }
        rewards = {f"{i}": 0.0 for i in range(NUM_AGENTS)}
        dones = {f"{i}": False for i in range(NUM_AGENTS)}
        dones["__all__"] = False
        infos = {f"{i}": {"test": "test"} for i in range(NUM_AGENTS)}
        return observations, rewards, dones, infos

    def render(self, mode: str = "human") -> None:
        from gym.envs.classic_control import rendering

        if mode != "human":
            super(Warehouse, self).render(mode=mode)

        if self._viewer is None:
            self._viewer = rendering.Viewer(VIEWPORT_DIMENSION_PX, VIEWPORT_DIMENSION_PX)

        for body in self._border_bodies:
            for fixture in body.fixtures:
                self._viewer.draw_polygon(
                    [fixture.body.transform * v * PIXELS_PER_METER for v in fixture.shape.vertices],
                    color=BORDER_COLOR,
                )

        for point in self._pickup_points:
            self._viewer.draw_polygon(
                [
                    ((point[0] - 0.4) * PIXELS_PER_METER, (point[1] - 0.4) * PIXELS_PER_METER,),
                    ((point[0] + 0.4) * PIXELS_PER_METER, (point[1] - 0.4) * PIXELS_PER_METER,),
                    ((point[0] + 0.4) * PIXELS_PER_METER, (point[1] + 0.4) * PIXELS_PER_METER,),
                    ((point[0] - 0.4) * PIXELS_PER_METER, (point[1] + 0.4) * PIXELS_PER_METER,),
                ],
                color=PICKUP_POINT_COLOR,
            )

        for point in self._delivery_points:
            self._viewer.draw_polygon(
                [
                    ((point[0] - 0.4) * PIXELS_PER_METER, (point[1] - 0.4) * PIXELS_PER_METER,),
                    ((point[0] + 0.4) * PIXELS_PER_METER, (point[1] - 0.4) * PIXELS_PER_METER,),
                    ((point[0] + 0.4) * PIXELS_PER_METER, (point[1] + 0.4) * PIXELS_PER_METER,),
                    ((point[0] - 0.4) * PIXELS_PER_METER, (point[1] + 0.4) * PIXELS_PER_METER,),
                ],
                color=DELIVERY_POINT_COLOR,
            )

        for body in self._agent_bodies:
            for fixture in body.fixtures:
                self._viewer.draw_circle(
                    fixture.shape.radius * PIXELS_PER_METER, 30, color=AGENT_COLOR
                ).add_attr(
                    rendering.Transform(
                        translation=fixture.body.transform * fixture.shape.pos * PIXELS_PER_METER
                    )
                )

        self._viewer.render()

    def close(self) -> None:
        pass
