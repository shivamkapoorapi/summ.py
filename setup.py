import setuptools

setuptools.setup(
    name='streamlit-app',
    version='1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='Streamlit app for document summarization',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        'streamlit',
        'langchain',
        'torch',
        'transformers',
        'base64',
    ],
)
