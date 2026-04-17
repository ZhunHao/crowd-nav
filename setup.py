from setuptools import setup


setup(
    name='crowdnav',
    version='0.0.1',
    packages=[
        'crowd_nav',
        'crowd_nav.configs',
        'crowd_nav.gui',
        'crowd_nav.gui.controllers',
        'crowd_nav.gui.workers',
        'crowd_nav.policy',
        'crowd_nav.utils',
        'crowd_sim',
        'crowd_sim.envs',
        'crowd_sim.envs.policy',
        'crowd_sim.envs.utils',
    ],
    install_requires=[
        'gitpython',
        'gym',
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
