# VCBot

The Virtual Cartesian Robot application for human-robot interaction. 

# Introduction

The *virtual Cartesian robot* (VCBot) is a program designed for interacting with an artificial neural agent through the computer mouse. VCBot is implemented in the Python programming language running on the top of the *neural robotics library* (see the [NRL project](https://github.com/oist-cnru/NRL)), which is an open-source tool implemented in the C++ programming language for prototyping human-robot interaction experiments, based on artificial neural cognitive control from recurrent neural network models.

## Project content

In this repository the implementation of VCBot is provided in the Python programming language version 3.7. The graphical user interface (GUI) relies on the tkinter toolkit, so the program can run in Linux, Microsoft Windows, and Mac OS X platforms. A user guide is provided [here](src/data/document/guide_print_version.pdf) with details on how to operate the GUI. The tutorial video below presents the project and explains how to run VCBot .

<a href="https://youtu.be/86mBuHwQWKg" rel="">![](src/images/tutorial2.png?raw=true)</a>

The instructions to install Python and the tkinter toolkit in multiple platforms are provided next.

## Requirements

Please download the [NRL project](https://github.com/henferch/NRL.git), follow the instruction provided for compiling the library in your host platform, and place the library file in the folder *src/lib*.

VCBot was developed in Python version 3.7, It requires the following packages: *numpy*, *sklearn*, *pandas*, *parse*, *matplotlib*, *IPython*, and *ttkthemes*

## Installation

It is recommended to create a virtual environment for running the client application, in order to preserve previous configurations in your system.

### Ubuntu 16.04

- **Install the tkinter toolkit**
    ```
    sudo apt-get install python3.7-tk
    ```
- **Create a virtual environment (recommended)**
  Install the virtual environment
  ```
  sudo pip3 install virtualenv
  virtualenv --python=/usr/bin/python3.7 ~/Workspace/virtual3_7
  ```
  Activate the virtual environment
  ```
  source DESIRED_PATH/virtual3_7/bin/activate
    ```
 
### Windows 8.1-10.x

-  **Install the tkinter toolkit**
 
 Download Python from https://www.python.org/downloads/windows/
 The version *Python 3.7.3 - March 25, 2019* was successfully tested.
 In advanced options select install tkinter and add python to the PATH environment variable .

- **Create a virtual environment (recommended)**

    Install the virtual environment
  ```
  pip install virtualenv
  virtualenv DESIRED_PATH\virtual3_7
  ```
  Activate the virtual environment
  ```
  DESIRED_PATH/virtual3_7/Scripts/activate.bat
  ```
### MAC OS X

- **install homebrew**

- **Install the tkinter toolkit**
  ```
  brew install tcl-tk
  ```
- **Create a virtual environment (optional but recommended)**
  Install the virtual environment
  ```
  virtualenv --python=/usr/bin/python3.7 DESIRED_PATH/virtual3_7
  ```
  Activate the virtual environment
  ```
  source ~/Workspace/virtual3_7/bin/activate
  ```
### All platforms

- **Install the Python required packages**
  ```
  pip install --upgrade setuptools
  pip3 install numpy
  pip3 install sklearn
  pip3 install pandas
  pip3 install parse
  pip3 install matplotlib
  pip3 install IPython
  pip3 install ttkthemes
  ```
 
- **Run VCBot**
  ```
  python main.py
  ```

## Work reference

H. F. Chame, A. Ahmadi, J. Jun (2020) *Towards hybrid primary intersubjectivity: a neural robotics library for human science*


## Contact

This program was implemented by: Hendry F. Chame

**Lab** Cognitive Neurorobotics Research Unit (CNRU)

**Institution** Okinawa Institute of Science and Technology Graduate University (OIST)

**Address** 1919-1, Tancha, Onna, Kunigami District, Okinawa 904-0495, Japan

**E-mail** hendryfchame@gmail.com



