import setuptools
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='matching_bandit',
    version='0.0.1',
    packages=find_packages(),
    license='MIT',
    description='An OpenAI gym environment for matching bandit algorithms.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Jialin Yi',
    url = 'https://gitlab.com/roka/matching-bandit',
    keywords = ['openai', 'openai-gym', 'gym', 'bandit', 'matching'],
    install_requires=[
        'gym',
        'numpy',
        'matplotlib',
        'pytest',
        'pytest-cov',
        'codecov'
    ],
    python_requires='>=3.5',
    classifiers=[
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8'        
    ],
)
