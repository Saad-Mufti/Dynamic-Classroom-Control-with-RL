# **Dynamic-Classroom-Control-with-RL**

## **Problem**
Students in classrooms often face difficulty focusing during lectures or other class activities, preventing them from learning efficiently. Thus, it is the primary focus of teachers to make the classroom a comfortable environment for learning to happen. 

## How it works
Classrooms have various environment variables that can be modified, such as temperature, air flow, and lighting. We propose a RL model that takes in some summary statistics of students in the classroom, and adjusts the environment to maximize student's productivity. 

![Diagram of our proposed model](/img/model_diagram.png)

The RL agent takes in three "statistics": an attention/engagement score, a thermal comfort score, and a facial emotion score. These scores are calculated for each student, and then aggregated for each cluster of students (using Bayesian clustering).


## **Installation and Running**

1. Clone the repository: `git clone https://github.com/Saad-Mufti/Dynamic-Classroom-Control-with-RL.git`
2. Install necessary dependencies: `pip install -r requirements.txt`. Note that `tensorflow` is only required for running `fer_model_custom.py`. Otherwise, it is fine to use the [`fer`](https://github.com/justinshenk/fer) package by [Justin Shenk](https://github.com/justinshenk). Also note that `pytorch` is installed to use CUDA support.
3. Install `OpenFace` *from source* as described [here](https://github.com/AnshulSood11/Engagement-Level-Prediction#geting-started).
4. Run the `room_agent_notebook.ipynb` notebook to train the RL model.