## create the virtual environment

1. create a virtual environment using your favourite tool:

```
python -m venv PPE-proj
```

2. Activate the virtual environment:
   in Windows git-bash

```
$source PPE-proj/Scripts/activate
```

in MacOS/Linux

```
$source PPE-proj/bin/activate
```

3. Install the following packages:

```
pip install roboflow
pip install opencv-python
```

## Run the app

Note that a webcam is required to run the app.
In the project directory:

```
python app.py
```
