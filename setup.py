import setuptools

setuptools.setup(
    name="rllib-warehouse",
    version="0.1",
    description="Multi-agent warehouse environment for reinforcement learning.",
    url="https://github.com/ffahleraz/rllib-warehouse",
    author="Faza Fahleraz",
    author_email="ffahleraz@gmail.com",
    license="MIT",
    package_data={"warehouse": ["py.typed"]},
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    install_requires=["numpy", "box2d-py", "gym", "ray"],
)
