import os
from setuptools import setup, find_packages


setup(
    name="LLMInstruct",
    version="1.0.0",
    author="NVIDIA",
    description='This package provides a framework to develop llm-generated instruction data.',
    packages=find_packages("."),
    python_requires=">=3.8",
    platforms=["any"],
    install_requires=[
        "openai==1.11.1",
        "langchain==0.1.5",
        "tiktoken==0.5.2",
        "numpy==1.26.2",
        "pandas==2.1.3",
        "astunparse==1.6.3",
        "datasets==2.16.1",
        "tqdm==4.66.1",
        "gradio==4.8.0",
        "datasketch==1.6.4",
        "nltk==3.8.1",
        "sympy",
        "vcdvcd"
    ],
)
