from setuptools import find_packages, setup
from typing import List 

HYPHEN_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    Docstring for get_requirements
    
    :param file_path: Description
    :type file_path: str
    :return: Description
    :rtype: List[str]
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements]
        #requirements = [req.replace("\n", "") for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(
name = 'mlproject',
version = '0.0.1',
author = 'tariq',
author_email = 'tariq.mansaray@gmail.com',
packages = find_packages(),
install_requires = get_requirements('requirements.txt')



)