
Copyright     	All rights are reserved, this code/project is not Open Source or Free
Bug           	None Documented     
Author        	Nathaniel Crossman (U00828694)
Email		 	crossman.4@wright.edu

Professor     	Meilin Liu
Course_Number 	CS 4370/6370-90
Date			12 05, 2019

Project Name:  CrossmanNathaniel_Project_Bonus

Project description:
	•	Task 1 - Basic CUDA Program using global memory (Works!)
		o	Every Part of this project works Completely
	•	Task 2 – CUDA program that takes advantage of shared memory (Works!)
		o	Every Part of this project works Completely


CUDA ENVIRONMENT:
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2018 NVIDIA Corporation
Built on Tue_Jun_12_23:07:04_CDT_2018
Cuda compilation tools, release 9.2, V9.2.148

RUN PROJECT:
To execute Reduce Program Type the Following:
To Run my code, you must use this configuration setting: nvcc -arch compute_50 -rdc=tru. Below is my full execution commands.

•	singularity exec --nv /home/containers/cuda92.sif nvcc -arch compute_50 -rdc=true CrossmanNathaniel_Project4_Bonus.cu -o Bonus
•	singularity exec --nv /home/containers/cuda92.sif /home/w072nxc/CS4370/project4/ Bonus

	NOTE: You must enter your Wright state User ID in order to get this to work.. /home/YOUR_W_ID/CS4370/Bonus





