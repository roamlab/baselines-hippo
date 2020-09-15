from setuptools import setup, find_packages

setup(
    name="baselines-hippo",
    version="0.0.1",
    author="Gagan Khandate",
    description= "Hindsight in PPO",
    url="https://github.com/roamlab/baselines-hippo/",
    packages=[package for package in find_packages() if package.startswith('hippo')],
    )