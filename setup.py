from setuptools import setup

setup(name='simple_elmo_pytorch',
      version='0.2',
      description='Simple_elmo pytorch is a Python library to work with pre-trained ELMo embeddings in PyTorch',
      packages=['simple_elmo_pytorch', 'simple_elmo_pytorch.nn','simple_elmo_pytorch.data','simple_elmo_pytorch.models'],
      author_email='joefox@inbox.ru',
      zip_safe=False)
