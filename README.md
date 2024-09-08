# Virtual Radar: Real-Time Millimeter-Wave Radar Sensor Simulation for Perception-driven Robotics

**[ViRa](https://aau.at/vira) is a real-time FMCW radar simulation frameworks aimed for robotic applications.**

<img src="https://github.com/chstetco/virtualradar/blob/main/readme_images/ICRA2021_snip01.gif" width="416" height="234" /> <img src="https://github.com/chstetco/virtualradar/blob/main/readme_images/ICRA2021_snip02.gif" width="416" height="234" />

The framework allows simulation of FMCW radar sensors with different configuration parameters in different scenarios embedded in the Unity3D game engine environment. 

## Main features

* Generation of radar **raw data** in real-time
* Fully customizable radar parameters
* Modelling of multi-antenna systems
* Modelling of wave penetration effects of non-conductive objects
* Modelling of antenna radiation patterns for beamforming

A [paper](https://ieeexplore.ieee.org/document/9387149) describing ViRa has been accepted for publication in the IEEE Robotics and Automation Letters (RA-L). 
A [video](https://www.youtube.com/watch?v=R3ZSykLs5iA) showcasing the frameworks capabilities in different scenarios can be found on our Youtube channel.

**NOTE:** We are continuosly working to improve and expand the ViRa framework. For most recent news, please refer to our [news section](https://github.com/chstetco/virtualradar/blob/main/docs/news.md).

<img src="https://www.aau.at/wp-content/uploads/2021/02/efrelogo.png" width="350" height="75" />        

![alt text](https://www.aau.at/wp-content/uploads/2021/02/KWF_FE_pos-aubergine-120x120.png)


## Installation and Usage

For a detailed instruction on how to install and use Vira on your platform, please refer to the links below.

* [Installation Guide](https://virtualradar.readthedocs.io/en/latest/_site/project/installation.html)
* [Framework Documentation](https://virtualradar.readthedocs.io)

## News

- 2021-23-07
  + We added some sample code for processing the output data of ViRa. 
  + We will soon release further implementations for multi-antenna systems. Stay tuned!

- 2021-02-03
  + ViRa's first release is online! V0.1.0 is now available for download.

- 2021-20-02
  + We are online! Creating documentation and installation guides.

- 2021-13-02
  + ViRa has been accepted for publication in the RA-L journal.

## Use in Synthetic Data Collection
1. Download [Unity 2019.4.8](https://unity.com/cn/releases/editor/whats-new/2019.4.8#installs) and [Unity Hub](https://unity.com/cn/download).
2. In Unity Hub, go to the "Install" page to "Locate" your Unity installation.
3. Clone this repo and enter the directory: `git clone https://github.com/Shimmer93/virtualradar.git && cd virtualradar`.
4. In Unity Hub, go to the "Projects" page to "Add" this repo as a Unity project.
5. Enter the project and select a scene.
6. To configure the mmwave radar, click the camera icon in the scene and adjust the parameters under "Screen Space Radar Control Plot (Script)".
7. To start the data collection:
   1. In the terminal, run `python main.py <args>` to start the TCP/IP server to receive and process data. Please make the arguments consistent with parameters set in Unity.
   2. Then in Unity, press `Ctrl+P` or click the "Play" button in the top center to start simulation and the TCP/IP client for data transmission.
   3. Currently the received data are not saved. To be implemented.
8. To stop the data collection:
   1. In Unity, press `Ctrl+P`. There may be lagging within the Unity program.
   2. The python script will quit automatically after some time due to timeout.


## Reference
Please cite our RA-L paper if you use this repository in your publications:
```
@ARTICLE{9387149,  
author={C. {Schöffmann} and B. {Ubezio} and C. {Böhm} and S. {Mühlbacher-Karrer} and H. {Zangl}},  
journal={IEEE Robotics and Automation Letters},   
title={Virtual Radar: Real-Time Millimeter-Wave Radar Sensor Simulation for Perception-Driven Robotics},   
year={2021},  
volume={6},  
number={3},  
pages={4704-4711},  
doi={10.1109/LRA.2021.3068916}
}
```
