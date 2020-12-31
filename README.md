# Prediction of graduation destination based on graph convolutional network embedded social relations
This is our implementation for the paper:\
Yang Guangze, Sun Sisi, Chen Lingyu, Ouyang Yong and Ye zhiwei. Prediction of graduation destination by the model of Social Graph Embedding-based Self-Attention Neural Network

Environment Settings
=
We use the framework pytorch.\
· pytorch version：'1.5.1'\
· python version: '3.6'\

student.py
=
The model is composed of a graph convolutional network, an attentional mechanism and multi-layer projection layer\

Dataset
=
we have collected nearly 28,429 academic records from 22 departments over spanning almost six years. These academic records include elective courses and compulsory courses.\
At the same time, we collected the development route of the graduates for 4 years(2014-2017) with different distribution to guide this task.
