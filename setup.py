from setuptools import setup, find_packages

setup(
    name="marg-one",
    version="0.1.0",
    description="MARG-One: Modular Multimodal AI & Robotics Vision Subsystem",
    author="Arnav Garg",
    author_email="gargarnav03@gmail.com",
    url="https://github.com/Arnav-G-null/MARG-One",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "mediapipe>=0.10.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
    ],
    entry_points={
        "console_scripts": [
            "marg-vision=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
)
