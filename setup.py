from setuptools import setup, find_packages

setup(
    name='blog',
    version='1.0.0',
    author='Anil Panda',
    author_email='akp.mooc@gmail.com',
    description='Code accompanying the blog  post',
    packages=find_packages(),    
    install_requires=['numpy >= 1.11.1', 'matplotlib >= 1.5.1'],
)