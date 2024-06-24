

from setuptools import setup, find_packages


setup(
    name="AllocationOfPrimaryCareCenters",
    version="0.1",
    packages=find_packages(),
    package_data={'AllocationOfPrimaryCareCenters': ['data/libraries.csv']},
    include_package_data=True,
    python_requires='>=3.12',
    install_requires=[
        'pandas==2.2.1',
        'numpy==1.26.4',
        'geopandas==0.14.3',
        'networkx==3.2.1',
        'plotly==5.20.0',
        'typing_extensions==4.11.0',
        'gerrychain==0.3.1',
        'shapely==2.0.3',
        'matplotlib==3.8.3',
        'Pympler==1.0.1',
        'jsonpointer==2.4'
    ],
    author="Alaittin Kirtisoglu",
    author_email="akirtisoglu@hawk.iit.edu",
    description="will be explained soon",
    url="https://github.com/kirtisoglu/Allocation-of-Primary-Care-Centers-in-Chicago"
)
