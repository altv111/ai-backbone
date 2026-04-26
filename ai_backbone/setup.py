from setuptools import find_packages, setup


setup(
    name="ai-backbone",
    version="0.1.0",
    description="Stateless AI backbone foundation layer",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.111.0",
        "uvicorn>=0.30.0",
        "pydantic>=2.7.0",
        "pydantic-settings>=2.2.1",
        "httpx>=0.27.0",
        "pytest>=8.2.0",
        "pytest-asyncio>=0.23.0",
    ],
)
