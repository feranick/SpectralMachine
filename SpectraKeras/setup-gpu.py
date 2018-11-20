from setuptools import setup, find_packages

setup(
    name='SpectraKeras',
    packages=find_packages(),
    install_requires=['numpy', 'keras', 'h5py', 'tensorflow-gpu'],
    entry_points={'console_scripts' : ['SpectraKeras_MLP=SpectraKeras_MLP:SpectraKeras_MLP',
        'SpectraKeras_CNN=SpectraKeras_CNN:SpectraKeras_CNN']},
    py_modules=['SpectraKeras_MLP','SpectraKeras_CNN','libSpectraKeras'],
    version='20181120a',
    description='Machine learning for spectral data',
    long_description= """ Machine learning for spectral data """,
    author='Nicola Ferralis',
    author_email='ferralis@mit.edu',
    url='https://github.com/feranick/SpectralMachine',
    download_url='https://github.com/feranick/SpectralMachine',
    keywords=['Machine learning', 'physics', 'spectroscopy', 'diffraction'],
    license='GPLv2',
    platforms='any',
    classifiers=[
     'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
     'Development Status :: 5 - Production/Stable',
     'Programming Language :: Python',
     'Programming Language :: Python :: 3',
     'Programming Language :: Python :: 3.5',
     'Programming Language :: Python :: 3.6',
     'Intended Audience :: Science/Research',
     'Topic :: Scientific/Engineering :: Chemistry',
     'Topic :: Scientific/Engineering :: Physics',
     ],
)
