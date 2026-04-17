from setuptools import setup


setup(
    name='crowdnav',
    version='0.0.1',
    python_requires='>=3.10,<3.11',
    packages=[
        'crowd_nav',
        'crowd_nav.cli',
        'crowd_nav.configs',
        'crowd_nav.gui',
        'crowd_nav.gui.controllers',
        'crowd_nav.gui.workers',
        'crowd_nav.planner',
        'crowd_nav.policy',
        'crowd_nav.utils',
        'crowd_sim',
        'crowd_sim.envs',
        'crowd_sim.envs.policy',
        'crowd_sim.envs.utils',
    ],
    install_requires=[
        'cython<3',
        'gitpython',
        'gym==0.15.7',
        'matplotlib',
        'numpy',
        'scipy',
        'torch',
        'torchvision',
    ],
    extras_require={
        'gui': [
            'Pillow',
            'PyQt5',
        ],
        'test': [
            'Pillow',
            'PyQt5',
            'pylint',
            'pytest',
            'pytest-qt',
        ],
    },
)
