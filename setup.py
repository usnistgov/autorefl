from setuptools import setup

setup(
    name='autorefl',
    version='0.1.0',
    packages=['autorefl'],
    url='https://github.com/hoogerheide/autorefl',
    license='MIT License',
    author='David Hoogerheide',
    author_email='david.hoogerheide@nist.gov',
    description='Autonomous Neutron Reflectometry',
    install_requires=["numpy", "scipy", "matplotlib", "reductus==0.9.0", "bumps==0.9.1",
                      "refl1d==0.8.15"],
    include_dirs=True,
    include_package_data=True
)
