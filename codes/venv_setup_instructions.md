# Setting Up a Python Virtual Environment (venv)

## 1. Create a Virtual Environment

Open your terminal in this folder and run:

```
python3 -m venv .venv
```

## 2. Activate the Virtual Environment

On macOS/Linux:
```
source .venv/bin/activate
```
On Windows:
```
.venv\Scripts\activate
```

## 3. Install Modules

While the venv is active, install modules using pip:
```
pip install <module-name>
```

## 4. Using a requirements.txt File

To install all modules listed in a `requirements.txt` file:
```
pip install -r requirements.txt
```

To generate a `requirements.txt` file from your current environment:
```
pip freeze > requirements.txt
```

## 5. Using Jupyter with Your venv

Install Jupyter and the IPython kernel in your venv:
```
pip install jupyter ipykernel
python -m ipykernel install --user --name=venv
```

Now you can select the `venv` kernel in Jupyter Notebook.
