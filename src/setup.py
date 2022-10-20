from distutils.core import setup

requirements = [
    'requests',
    'tqdm',
    'pandas'
]

setup(
    name='graph2gnn',
    version='0.0.1',
    author='Rob Rossmiller',
    author_email='opensource@optum.com',
    description='A library to get from a graph DB to GNN training data in less time',
    long_description=open('../README.md').read(),
    # long_description_content_type='text/markdown',
    url='https://github.com/Optum/graph2gnn',
    project_urls={
        'Bug Tracker': 'https://github.com/Optum/graph2gnn/issues',
    },
    license='Apache 2.0',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: POSIX'
    ],
    packages=['graph2gnn'],
    install_requires=requirements,
    python_requires='>=3.8',
)