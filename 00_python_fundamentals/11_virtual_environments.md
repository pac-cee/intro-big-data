# Virtual Environments and Package Management

## Table of Contents
1. [Introduction to Virtual Environments](#introduction-to-virtual-environments)
2. [Creating and Managing Virtual Environments](#creating-and-managing-virtual-environments)
3. [Package Management with pip](#package-management-with-pip)
4. [requirements.txt and pip freeze](#requirementstxt-and-pip-freeze)
5. [setup.py and pyproject.toml](#setuppy-and-pyprojecttoml)
6. [Dependency Management Tools](#dependency-management-tools)
7. [Best Practices](#best-practices)
8. [Practice Exercises](#practice-exercises)

## Introduction to Virtual Environments

### What is a Virtual Environment?
A virtual environment is an isolated Python environment that allows you to manage dependencies for different projects separately.

### Why Use Virtual Environments?
- Isolate project dependencies
- Avoid version conflicts between projects
- Reproduce environments easily
- Keep your global Python installation clean

## Creating and Managing Virtual Environments

### Using `venv` (Built-in)
```bash
# Create a virtual environment
python -m venv myenv

# Activate the environment
# On Windows:
myenv\Scripts\activate
# On macOS/Linux:
source myenv/bin/activate

# Deactivate the environment
deactivate
```

### Using `virtualenv`
```bash
# Install virtualenv
pip install virtualenv

# Create a virtual environment
virtualenv myenv

# Activate (same as venv)
source myenv/bin/activate  # macOS/Linux
myenv\Scripts\activate     # Windows
```

### Using `conda`
```bash
# Create a conda environment
conda create --name myenv python=3.9

# Activate the environment
conda activate myenv

# Deactivate
conda deactivate

# List all environments
conda env list

# Remove an environment
conda env remove --name myenv
```

## Package Management with pip

### Basic Commands
```bash
# Install a package
pip install package_name

# Install specific version
pip install package_name==1.2.3

# Install from requirements.txt
pip install -r requirements.txt

# Upgrade a package
pip install --upgrade package_name

# Uninstall a package
pip uninstall package_name

# List installed packages
pip list

# Show package information
pip show package_name

# Search for packages
pip search "query"
```

### Installing from Different Sources
```bash
# Install from PyPI (default)
pip install package_name

# Install from a local directory
pip install /path/to/directory

# Install from a git repository
pip install git+https://github.com/user/repo.git@branch

# Install from a wheel file
pip install package_name.whl

# Install in development/editable mode
pip install -e .
```

## requirements.txt and pip freeze

### Generating requirements.txt
```bash
# Save all installed packages to requirements.txt
pip freeze > requirements.txt

# Install from requirements.txt
pip install -r requirements.txt
```

### Example requirements.txt
```
# Exact version
Django==3.2.9

# Version range
requests>=2.25.0,<3.0.0

# Git repository
git+https://github.com/user/repo.git@branch#egg=package_name

# Local package
./local_package

# With environment markers
pytest; python_version > '3.5'
```

### pip-tools for Better Dependency Management
```bash
# Install pip-tools
pip install pip-tools

# Create requirements.in with your direct dependencies
# Then compile to requirements.txt
pip-compile requirements.in

# Update all dependencies
pip-compile --upgrade

# Update specific package
pip-compile --upgrade-package package_name
```

## setup.py and pyproject.toml

### setup.py (Legacy)
```python
from setuptools import setup, find_packages

setup(
    name="mypackage",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'requests>=2.25.0',
        'numpy>=1.19.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'black>=21.0',
        ],
    },
)
```

### pyproject.toml (Modern)
```toml
[build-system]
requires = ["setuptools>=42", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "mypackage"
version = "0.1.0"
description = "My awesome package"
requires-python = ">=3.7"
dependencies = [
    "requests>=2.25.0",
    "numpy>=1.19.0",
]

[project.optional-dependencies]
dev = ["pytest>=6.0", "black>=21.0"]

[tool.black]
line-length = 88
target-version = ['py37']
```

## Dependency Management Tools

### pipenv
```bash
# Install pipenv
pip install pipenv

# Create a new project
pipenv --python 3.9

# Install a package
pipenv install package_name

# Install dev dependency
pipenv install --dev pytest

# Run commands in the virtual environment
pipenv run python script.py

# Activate the virtual environment
pipenv shell

# Generate requirements.txt
pipenv lock -r > requirements.txt
```

### Poetry
```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -


# Create a new project
poetry new myproject
cd myproject

# Add dependencies
poetry add requests
poetry add --dev pytest

# Install dependencies
poetry install

# Run commands
poetry run python script.py

# Activate the virtual environment
poetry shell

# Export to requirements.txt
poetry export -f requirements.txt --output requirements.txt
```

### conda (Alternative to pip)
```bash
# Install a package
conda install package_name

# Install from a specific channel
conda install -c conda-forge package_name

# Create environment from environment.yml
conda env create -f environment.yml

# Export environment
conda env export > environment.yml
```

## Best Practices

### Virtual Environment Best Practices
1. Always use a virtual environment for each project
2. Never install packages globally unless absolutely necessary
3. Include a `.gitignore` file to exclude virtual environment directories
4. Document how to set up the development environment in README.md

### Dependency Management Best Practices
1. Pin exact versions in production (use `==`)
2. Use `requirements.in` and `requirements.txt` for pip-tools
3. Separate development and production dependencies
4. Use environment markers when needed
5. Consider using `pipenv` or `poetry` for better dependency management

### Security Best Practices
1. Regularly update dependencies to fix security vulnerabilities
2. Use `pip-audit` to check for known vulnerabilities
3. Never commit sensitive information in requirements files
4. Use private package indexes when needed

## Practice Exercises

1. **Create and Activate a Virtual Environment**
   - Create a new virtual environment using `venv`
   - Activate it and install a few packages
   - Deactivate and reactivate it

2. **Manage Dependencies**
   - Create a `requirements.txt` file with some dependencies
   - Install them in a new virtual environment
   - Add a new package and update the requirements file

3. **Convert to pyproject.toml**
   - Take an existing project with `setup.py` and convert it to use `pyproject.toml`
   - Build and install the package in development mode

4. **Use pip-tools**
   - Create a `requirements.in` file with some dependencies
   - Use `pip-compile` to generate a `requirements.txt`
   - Add a new dependency and update the files

5. **Set Up a Development Environment**
   - Choose between `pipenv` or `poetry`
   - Set up a new project with development dependencies
   - Add a script to run tests

6. **Environment Variables**
   - Create a `.env` file with some variables
   - Use `python-dotenv` to load them in your Python script
   - Add `.env` to `.gitignore`

7. **Docker Integration**
   - Create a `Dockerfile` for a Python application
   - Use multi-stage builds to keep the image small
   - Set up a `docker-compose.yml` for development

8. **Dependency Auditing**
   - Use `safety` or `pip-audit` to check for vulnerabilities
   - Update any vulnerable dependencies
   - Document the process

9. **Continuous Integration**
   - Set up GitHub Actions or GitLab CI to test your project
   - Add steps to install dependencies and run tests
   - Configure caching for faster builds

10. **Documentation**
    - Document the setup process in your README.md
    - Include all necessary commands
    - Add a "Troubleshooting" section with common issues

---
Next: [Advanced Python Features](./12_advanced_features.md)
