from setuptools import setup

setup(name='paican',
      version='0.1',
      description='Bayesian Robust Attributed Graph Clustering: '
                  'Joint Learning of Partial Anomalies and Group Structure',
      author='Aleksandar Bojchevski',
      author_email='bojchevs@in.tum.de',
      packages=['paican'],
      install_requires=['tensorflow>=1.0', 'numpy'],
      zip_safe=False)
