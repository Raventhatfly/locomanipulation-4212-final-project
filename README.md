# CulinaBot

This repository contains our final project for MIT 6.4212 Robotic Manipulation. We implemented **CulinaBot**, a mobile manipulator that can execute user commands in a custom kitchen environment using Large Language Models (GPT4). Here is a demo of our robot executing the command "Put the apple beside the banana":

https://github.com/user-attachments/assets/1352b97e-5ff6-4e8f-9033-09292d4c8b13

## Environment Setup
```
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
pip install -e .
```

## how to render the env inside the meshcat 
```
python3 setup_env.py
```


