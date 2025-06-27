from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path:str) -> List[str]:
    """
    This function will return the list of requirements
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
        # If -e . is present, then remove it
        if '-e .' in requirements:
            requirements.remove('-e .')

    return requirements

setup(
    name='MachineLearningProject',
    version='0.0.1',
    author='Himanshu',
    author_email='himanshurawat4139@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)