from setuptools import find_packages, setup

package_name = 'stress_inference_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your name',
    maintainer_email='your email ID',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'stress_inference_realtime = stress_inference_pkg.stress_inference_realtime:main',
        'stress_inference_realtime_al = stress_inference_pkg.stress_inference_realtime_al:main',
        'mock_publisher = stress_inference_pkg.mock_realtime_publisher:main',
        'mock_wesad_publisher = stress_inference_pkg.wesad_publisher:main',
        'emobit_publisher = stress_inference_pkg.emobit_publisher:main',
        'gui = stress_inference_pkg.gui_node:main',
        'rq1 = stress_inference_pkg.rq1_analysis_raw:main',
        ],
    },
)
