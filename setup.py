from setuptools import setup, find_packages

setup(
    name='torp',
    version='0.1.0',
    description='Torch Helper',
    author='Sathyaprakash Narayanan',
    packages=find_packages(),
    install_requires=[certifi==2023.5.7,charset-normalizer==3.2.0,cmake==3.26.4,contourpy==1.1.0,cycler==0.11.0,filelock==3.12.2,fonttools==4.41.0,idna==3.4,Jinja2==3.1.2,kiwisolver==1.4.4,lit==16.0.6,MarkupSafe==2.1.3,matplotlib==3.7.2,mpmath==1.3.0,networkx==3.1,numpy==1.25.1,nvidia-cublas-cu11==11.10.3.66,nvidia-cuda-cupti-cu11==11.7.101,nvidia-cuda-nvrtc-cu11==11.7.99,nvidia-cuda-runtime-cu11==11.7.99,nvidia-cudnn-cu11==8.5.0.96,nvidia-cufft-cu11==10.9.0.58,nvidia-curand-cu11==10.2.10.91,nvidia-cusolver-cu11==11.4.0.1,nvidia-cusparse-cu11==11.7.4.91,nvidia-nccl-cu11==2.14.3,nvidia-nvtx-cu11==11.7.91,packaging==23.1,Pillow==10.0.0,pyparsing==3.0.9,python-dateutil==2.8.2,requests==2.31.0,six==1.16.0,sympy==1.12,torch==2.0.1,torchprofile==0.0.4,torchvision==0.15.2,tqdm==4.65.0,triton==2.0.0,typing_extensions==4.7.1,urllib3==2.0.3]  # List any dependencies your package requires
)
