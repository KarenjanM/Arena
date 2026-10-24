import os

from setuptools import find_packages, setup

package_name = 'arena_mcp'

setup(
    name=package_name,
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        (os.path.join('share', package_name), ['package.xml']),
    ],
    zip_safe=True,
    author='voshch',
    author_email='voshch@arena-rosnav.org',
    maintainer='voshch',
    maintainer_email='voshch@arena-rosnav.org',
    description='One MCP server for Arena',
    license='MIT',
    entry_points={
        'console_scripts': [
            'arena_mcp = arena_mcp.__main__:main',
        ],
        'arena_mcp.domains': [
            'jobs = arena_mcp.domains.jobs:domain',
            'system = arena_mcp.domains.system:domain',
            'workspace = arena_mcp.domains.workspace:domain',
            'ros = arena_mcp.domains.ros:domain',
        ],
    },
)
