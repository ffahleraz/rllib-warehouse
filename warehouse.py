import typing

import numpy as np
import gym
from gym.envs.classic_control import rendering
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import Box2D
from Box2D.b2 import (
    world,
    polygonShape,
    circleShape,
    staticBody,
    dynamicBody,
)


# Env
N_AGENTS: int = 10
N_PICKUP_POINTS: int = 10
N_DELIVERY_POINTS: int = 10
AREA_DIMENSION: float = 10.0

# Box2D
B2_VEL_ITERS: int = 10
B2_POS_ITERS: int = 10
FRAMES_PER_SECOND: int = 50
PIXELS_PER_METER: int = 50
VIEWPORT_DIMENSION: int = int(AREA_DIMENSION) * PIXELS_PER_METER


class Warehouse(MultiAgentEnv):
    def __init__(self) -> None:
        super(Warehouse, self).__init__()

        self.metadata = {
            "render.modes": ["human"],
            "video.frames_per_second": FRAMES_PER_SECOND,
        }
        self.reward_range = (-np.inf, -np.inf)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "self_position": gym.spaces.Box(
                    low=AREA_DIMENSION,
                    high=AREA_DIMENSION,
                    shape=(2,),
                    dtype=np.float32,
                ),
                "pickup_points": gym.spaces.Dict(
                    {
                        "availability": gym.spaces.MultiBinary(N_PICKUP_POINTS),
                        "positions": gym.spaces.Box(
                            low=AREA_DIMENSION,
                            high=AREA_DIMENSION,
                            shape=(N_PICKUP_POINTS, 2),
                            dtype=np.float32,
                        ),
                    }
                ),
                "delivery_points": gym.spaces.Dict(
                    {
                        "availability": gym.spaces.MultiBinary(N_DELIVERY_POINTS),
                        "positions": gym.spaces.Box(
                            low=AREA_DIMENSION,
                            high=AREA_DIMENSION,
                            shape=(N_DELIVERY_POINTS, 2),
                            dtype=np.float32,
                        ),
                    }
                ),
            }
        )

        self._world = world(gravity=(0, 0), doSleep=True)
        self._viewer: gym.Viewer = None

    def _setup(self) -> None:
        # self._agent_bodies: typing.List[dynamicBody] = []
        # for _ in range(N_AGENTS):
        #     body = self._world.CreateDynamicBody(position=(2, 5))

        body = self._world.CreateDynamicBody(position=(2, 5))
        circle = body.CreateCircleFixture(radius=0.4, density=1, friction=0.5)

    def _draw_circle(self, body: Box2D.b2Body, fixture: Box2D.b2Fixture) -> None:
        # position = body.transform * fixture.shape.pos * PIXELS_PER_METER
        # position = (position[0], VIEWPORT_DIMENSION - position[1])
        self._viewer.draw_circle(
            fixture.shape.radius * PIXELS_PER_METER, 30, color=(0, 0, 0)
        ).add_attr(
            rendering.Transform(
                translation=fixture.body.transform
                * fixture.shape.pos
                * PIXELS_PER_METER
            )
        )
        print("rad", fixture.shape.radius)
        print("pos", fixture.body.transform * fixture.shape.pos)
        self._viewer.draw_circle()

    def reset(self) -> typing.Dict[str, gym.spaces.Dict]:
        self._counter = 1

        self._setup()

        return {f"{i}": self.observation_space.sample() for i in range(N_AGENTS)}

    def step(
        self, action_dict: typing.Dict[str, gym.spaces.Box]
    ) -> typing.Tuple[
        typing.Dict[str, gym.spaces.Dict],
        typing.Dict[str, float],
        typing.Dict[str, bool],
        typing.Dict[str, typing.Dict[str, str]],
    ]:
        self._counter += 1

        self._world.Step(1.0 / FRAMES_PER_SECOND, 10, 10)

        observations = {
            f"{i}": self.observation_space.sample() for i in range(N_AGENTS)
        }
        rewards = {f"{i}": 0.0 for i in range(N_AGENTS)}
        dones = {f"{i}": False for i in range(N_AGENTS)}
        dones["__all__"] = self._counter % 1000 == 0
        infos = {f"{i}": {"test": "test"} for i in range(N_AGENTS)}
        return observations, rewards, dones, infos

    def render(self, mode: str = "human") -> None:
        if mode != "human":
            super(Warehouse, self).render(mode=mode)

        if self._viewer is None:
            self._viewer = rendering.Viewer(VIEWPORT_DIMENSION, VIEWPORT_DIMENSION)

        for body in self._world.bodies:
            for fixture in body.fixtures:
                self._draw_circle(body, fixture)

        self._viewer.render()
        print(self._counter)

    def close(self) -> None:
        pass
