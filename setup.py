from setuptools import setup

setup(name='automotive_behavior_inference',
      version='0.1',
      description='inferring automotive driving behavior',
      author='Blake Wulfe',
      author_email='blake.w.wulfe@gmail.com',
      license='MIT',
      packages=['abi'],
      zip_safe=False,
      install_requires=[
        'numpy',
        'tensorflow',
      ])