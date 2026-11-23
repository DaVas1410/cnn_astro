"""Setup script for Fractal Classifier package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / 'README.md'
long_description = ''
if readme_file.exists():
    long_description = readme_file.read_text(encoding='utf-8')

# Read requirements
requirements_file = Path(__file__).parent / 'requirements.txt'
requirements = []
if requirements_file.exists():
    requirements = requirements_file.read_text().strip().split('\n')

setup(
    name='fractal-classifier',
    version='0.1.0',
    author='Fractal Cloud Research Team',
    description='Deep learning framework for fractal cloud classification',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/fractal-classifier',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Astronomy',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.0',
            'pytest-cov>=4.0',
            'black>=22.0',
            'flake8>=5.0',
            'mypy>=0.990',
        ],
        'docs': [
            'sphinx>=5.0',
            'sphinx-rtd-theme>=1.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'fractal-train=fractal_classifier.scripts.train:main',
            'fractal-evaluate=fractal_classifier.scripts.evaluate:main',
            'fractal-analyze=fractal_classifier.scripts.analyze:main',
        ],
    },
    include_package_data=True,
    package_data={
        'fractal_classifier': [
            'config/default_config.yaml',
        ],
    },
    zip_safe=False,
)
