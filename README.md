# CarND-Capstone
Capstone project of Udacity's Self Driving Car Engineer Nanodegree

## Overview

This is the Capstone project of Udacity's Self Driving Car Engineer Nanodegree course. In this project we 
build various components to drive an autonomous car around a simulated and real-site test tracks.

## Installation Instructions

Please follow the instructions [here](https://github.com/udacity/CarND-Capstone) in order to get setup with:

* Robot Operating System (ROS) : Installed in either a Docker container or on an Ubuntu (recommended 16.04)
OS which could be in a virtual-machine or a dedicated OS
* Udacity simulator : Installed and run in the Ubuntu OS

## Component Details

We implement three major components to drive Carla (Udacity's self-driving car) around the test tracks. 

* Waypoint Updater
* Drive-by-wire Controller
* Traffic Light Detector

The overall architecture of the system with the above components is described in the [project-walkthrough doc](./walkthrough.md).

## Attribution

* [Alex Lechner's](https://github.com/alex-lechner) post on [traffic-light classification](https://github.com/alex-lechner/Traffic-Light-Classification)
* [Vatsal Srivatsava's](https://github.com/coldKnight) dataset containing traffic-light images from
Udacity simulator and site, and the [transfer-learning instructions](https://github.com/coldKnight/TrafficLight_Detection-TensorFlowAPI#get-the-dataset)
