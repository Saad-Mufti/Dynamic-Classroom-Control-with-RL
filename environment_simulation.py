# %%
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import random
import os


# %%
def Profile_Def():

    Profile_Info1 = {}
    Profile_Info2 = {}

    Profile_Info1['1'] = [ 21,  64, 174, 1]                                               # TC_info 1 Include the Age, Weight, Height Gender
    Profile_Info2['1'] = [ 46.49, 1.27, 0.47, 76, 33]                                          # TC_info 2 include Metabolic Rate, Clothing Level, Heart Rate and Skin Temperature

    # Prof
    Profile_Info1['2'] = [18, 58, 183, 1]; Profile_Info2['2'] = [42.87, 1.45, 0.54, 79, 29]

    Profile_Info1['3'] = [19, 46, 150, 2]; Profile_Info2['3'] = [45.61, 1.19, 0.53, 62, 30]

    Profile_Info1['4'] = [21, 56, 164, 2]; Profile_Info2['4'] = [41.69, 1.15, 0.53, 82, 31]

    Profile_Info1['5'] = [32, 63, 174, 1]; Profile_Info2['5'] = [59.34, 1.29, 0.54, 70, 33]

    Profile_Info1['6'] = [21, 50, 164, 1]; Profile_Info2['6'] = [43.12, 1.24, 0.53, 87, 29]

    Profile_Info1['7'] = [32, 63, 174, 2]; Profile_Info2['7'] = [64.26, 1.26, 0.54, 71, 34]

    Profile_Info1['8'] = [31, 56, 164, 2]; Profile_Info2['8'] = [42.57, 1.15, 0.53, 98, 30]

    Profile_Info1['9'] = [24, 55, 170, 1]; Profile_Info2['9'] = [47.03, 1.20, 0.46, 69, 34]

    Profile_Info1['10'] = [21, 61, 171, 2]; Profile_Info2['10'] = [50, 1.11, 0.54, 63, 30]

    return Profile_Info1, Profile_Info2

# %%
def Profile_sampling(profile, n):
  test_list = [-1, 1]
  # Adding Noise to the metabolic Rate

  mean_met = profile[f'{n}'][0]; var_met = 0.14
  mu_met = np.random.normal(0, var_met)
  random_num = random.choice(test_list)
  mu_met *= random_num
  profile[f'{n}'][0] += mu_met
  

  # Adding Noise to the clothing level

  mean_cl = profile[f'{n}'][1]; var_cl = 0.3
  mu_cl = np.random.normal(0, var_cl)
  random_num = random.choice(test_list)
  mu_cl *= random_num
  profile[f'{n}'][1] += mu_cl
  # Adding Noise to the Heart Rate
  
  mean_hr = profile[f'{n}'][2]; var_hr = 4.8
  mu_hr = np.random.normal(0, var_hr)
  random_num = random.choice(test_list)
  mu_hr *= random_num
  profile[f'{n}'][2] += mu_hr

  # Adding Noise to the Skin Temperature
  
  mean_st = profile[f'{n}'][3]; var_st = 1.2
  mu_st = np.random.normal(0, var_st)
  random_num = random.choice(test_list)
  mu_st *= random_num
  profile[f'{n}'][3] += mu_st

# %%
X_row_1 = []; Y_row_1 = []; X1 = 1; Y1 = 0.5 
X_row_2 = []; Y_row_2 = []; X2 = 1; Y2 = 1
X_row_3 = []; Y_row_3 = []; X3 = 1; Y3 = 1.5

for i in range(0,10):
    X_row_1.append(X1); Y_row_1.append(Y1)
    X_row_2.append(X2); Y_row_2.append(Y2)
    X_row_2.append(X3); Y_row_2.append(Y3)
    X1 = X1 + 0.5; X2 = X2 + 0.5; X3 = X3 + 0.5

X_row = X_row_1 + X_row_2 + X_row_3
Y_row = Y_row_1 + Y_row_2 + Y_row_3

variability_Seats = np.random.normal(0, 0.2)

for i in range(0, len(X_row)):
    test_list = [-1,1]
    variability_Seats_X = np.random.normal(0, 0.1)
    variability_Seats_Y = np.random.normal(0.2, 0.12)
    random_num = random.choice(test_list)
    X_row[i] = (X_row[i] +  variability_Seats_X)
    Y_row[i] = (Y_row[i] +  variability_Seats_Y)

plt.scatter(X_row, Y_row)

X_row = np.array(X_row)
X_row = X_row.reshape(-1)

Y_row = np.array(Y_row)
Y_row = Y_row.reshape(-1)

pipe_df = pd.DataFrame([X_row, Y_row])
pipe_df = pipe_df.T
pipe_df.columns = ['x', 'y']

# Thermal Comfort Information Which consists of two info lists

def pipe_df_Init(Init_Temp):

    Temperature = [[Init_Temp]]*len(pipe_df)
    files_dir = os.listdir('Profiles')
    Class_df = []
    Classes = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'suprise']
    test_list = range(0,6)
    
    Profiles1_df = []
    Profiles2_df = []
    Profile_name = []
    Profile_sex = []
    emotion_profile = []
    emotion_path = []
    video_path = []
    image_profile = []
    
    Profile_Info1, Profile_Info2 = Profile_Def()

    for j in range(0,len(pipe_df)):

        test_list = range(1,10)
        random_num = random.choice(test_list)
        Profile_sampling(Profile_Info2, random_num)
        Profiles1_df.append(Profile_Info1[f'{random_num}'])
        Profiles2_df.append(Profile_Info2[f'{random_num}'])
        Profile_name.append(random_num)
        
        Profile_sex.append(Profile_Info1[f'{random_num}'][3])
    
    pipe_df[f'Profile Number'] =  Profile_name
    pipe_df[f'Profile Sex'] = Profile_sex
    pipe_df[f'Profile Info 1'] =  Profiles1_df
    pipe_df[f'Profile Info 2'] =  Profiles2_df
    pipe_df['Temperature'] = Temperature
    

    # print(pipe_df.head())
    
    random_num = np.random.choice(6, size = len(pipe_df))

    for i in range(0,len(pipe_df)):
        
        Classes = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'suprise']
        random_num = np.random.choice(6)
        emotion_class = random_num       # Get the value of the emotion Class
        Class = Classes[emotion_class]      # The Class Corresponding String Value
        sex = pipe_df[f'Profile Sex'][i]    # Get the sex of the corresponding profile

        if sex == 1: s = 'm'
        if sex == 2: s = 'f'

        test_list = [1,2]                               # Choose from one of the profiles
        random_n = random.choice(test_list)             
        directory = f'Profiles/{s}_{random_n}/{Class}.jpg'    # Get the directory from the selection
        image = cv2.imread(directory)                   # read the corresponding image
        emotion_profile.append(image)                   # append the image to the list 
        emotion_path.append(directory)                  # append the image path to the directory
        image_profile.append(random_n)


    for k in range(0,30):
        random_n = np.random.choice(10)
        attention_class = random_n
        path = f'{attention_class}_o_10.mp4'
        video_path.append(path)

    pipe_df['img_profile'] = image_profile
    pipe_df['img_path'] = emotion_path
    pipe_df['video_path'] = video_path

    # pipe_df['vid_path'] = video_path

def df_Update(pipe_df, Temperature, Emotion, Emotion_value, Attention, Attention_Value):

    Classes = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'suprise']
    img_path = []
    video_path = []

    Temperature_val = [[Temperature]]*len(pipe_df)

    Attention_Score  = Attention + Attention_Value
    Emotion_Score = Emotion + Emotion_value

    for i in range(0,len(pipe_df)):

        emotion_class = np.argmax(Emotion_Score)
        sex = pipe_df['Profile Sex'][i]
        profile_img = pipe_df['img_profile'][i]

        Class = Classes[emotion_class]
        directory = f'Profiles/{sex}_{profile_img}/{Class}.jpg'

        img_path.append(directory)

        attention_class = Attention_Score
        path = f'{attention_class}_o_10.mp4'
        video_path.append(path)
    

    pipe_df['img_path'] = img_path
    pipe_df['video_path'] = video_path
    pipe_df['Temperature'] = Temperature_val


    return pipe_df
        
    

    pipe_df['img_path'] = img_path
    pipe_df['video_path'] = video_path
    pipe_df['Temperature'] = Temperature_val


    return pipe_df

# %%
def Profile_Change(Light, Air_Mvmt, Profile_Indx):

    if Profile_Indx == 1:
        if Light < 2700:
            if Air_Mvmt < 1: TC_var == -0.3; prob = [0, 0, 0.05, -0.1, -0.1, 0.05, 0]
            if Air_Mvmt > 1.2: TC_var == -0.4; prob = [0.01, 0, 0.04, -0.15, -0.1, 0.05, 0]
        
        if Light > 5000:
            if Air_Mvmt < 1: TC_var = -0.05;  prob = [0, 0, 0.03, -0.04, -0.02, 0.03, 0]
            if Air_Mvmt > 1: TC_var = 0.2;  prob = [0, 0, -0.03, 0.06, 0.02, -0.05, 0]
            
    
    if Profile_Indx == 2:
        if Light < 2700:
            if Air_Mvmt < 1: TC_var == -0.1;  prob = [0, 0, 0.02, -0.025, -0.025, 0.03, 0]
            if Air_Mvmt > 1.5: TC_var == -0.3; prob = [0, 0, 0.03, -0.04, -0.03, 0.04, 0]
        
        if Light > 5000:
            if Air_Mvmt < 1: TC_var = 0.15; prob = [0, 0, -0.02, 0.02, 0.01, -0.01, 0]
            if Air_Mvmt > 2: TC_var = -0.2; prob = [0, 0, 0.02, -0.025, -0.025, 0.03, 0]

    if Profile_Indx == 3:
        if Light < 2700:
            if Air_Mvmt < 0.5: TC_var == -0.25; prob = [0, 0, 0.02, 0.015, -0.02, 0.015, 0]
            if Air_Mvmt > 1.5: TC_var == -0.15; prob = [0, 0, -0.015, 0.015, 0.01, -0.01, 0]
        
        if Light > 5000:
            if Air_Mvmt < 1: TC_var = -0.2; prob = [0, 0, 0.02, -0.02, -0.01, 0.01, 0]
            if Air_Mvmt > 1: TC_var = -0.3; prob = [0, 0, 0.02, -0.03, -0.01, 0.02, 0]

    if Profile_Indx == 4:
        if Light < 2700:
            if Air_Mvmt < 1: TC_var == -0.3; prob = [0, 0, 0.02, -0.02, -0.01, 0.01, 0]
            if Air_Mvmt > 2: TC_var == -0.4; prob = [0, 0, 0.03, -0.03, -0.02, 0.02, 0]
        
        if Light > 5000:
            if Air_Mvmt < 1: TC_var = 0.1;  prob = [0, 0, -0.02, 0.02, 0.01, -0.01, 0]
            if Air_Mvmt > 2: TC_var = 0.2; prob = [0, 0, -0.02, 0.03, 0.01, -0.01, 0]

    if Profile_Indx == 5:
        if Light < 2700:
            if Air_Mvmt < 1: TC_var == 0.1; prob = [0, 0, -0.02, 0.02, 0.01, -0.01, 0]
            if Air_Mvmt > 1.2: TC_var == 0.2; prob = [0, 0, -0.02, 0.02, 0.01, -0.01, 0]
        
        if Light > 5000:
            if Air_Mvmt < 1: TC_var = -0.1; prob = [0, 0, 0.02, -0.015, -0.015, 0.01, 0]
            if Air_Mvmt > 1: TC_var = -0.15; prob = [0, 0, 0.025, -0.02, -0.015, 0.01, 0]

    if Profile_Indx == 6:
        if Light < 2700:
            if Air_Mvmt < 1: TC_var == -0.1; prob = [0, 0, 0.01, -0.01, -0.01, 0.01, 0]
            if Air_Mvmt > 1.2: TC_var == -0.15; prob = [0, 0, 0.02, -0.015, -0.015, 0.01, 0]
        
        if Light > 5000:
            if Air_Mvmt < 0.7: TC_var = 0.25; prob = [0, 0, -0.025, 0.03, 0.015, -0.02, 0]
            if Air_Mvmt > 1.5: TC_var = 0.15; prob = [0, 0, -0.02, 0.02, 0.01, -0.01, 0]

    if Profile_Indx == 7:
        if Light < 2700:
            if Air_Mvmt < 1: TC_var == 0.3; prob = [0, 0, -0.025, 0.035, 0.015, -0.025, 0]
            if Air_Mvmt > 1.2: TC_var == 0.25; prob = [0, 0, -0.025, 0.03, 0.015, -0.02, 0]
        
        if Light > 5000:
            if Air_Mvmt < 0.7: TC_var = -0.1;  prob = [0, 0, 0.01, -0.01, -0.01, 0.01, 0]
            if Air_Mvmt > 2: TC_var = -0.2;  prob = [0, 0, 0.02, -0.03, -0.01, 0.01, 0]


    if Profile_Indx == 8:
        if Light < 2700:
            if Air_Mvmt < 1: TC_var == -0.05; prob = [0, 0, 0.01, -0.01, -0.005, 0.005, 0]
            if Air_Mvmt > 1.2: TC_var == -0.15; prob = [0, 0, 0.015, -0.015, -0.01, 0.01, 0]
        
        if Light > 5000:
            if Air_Mvmt < 1: TC_var = 0.25; prob = [0, 0, -0.025, 0.035, 0.015, -0.025, 0]
            if Air_Mvmt > 1.5: TC_var = 0.2;  prob = [0, 0, -0.02, 0.02, 0.01, -0.01, 0]

    if Profile_Indx == 9:
        if Light < 2700:
            if Air_Mvmt < 1: TC_var == -0.05; prob = [0, 0, 0.01, -0.01, -0.01, 0.01, 0]
            if Air_Mvmt > 1.2: TC_var == -0.1;  prob = [0, 0, 0.02, -0.02, -0.01, 0.01, 0]
        
        if Light > 5000:
            if Air_Mvmt < 1: TC_var = 0.1; prob = [0, 0, -0.01, 0.01, 0.01, -0.01, 0]
            if Air_Mvmt > 1: TC_var = 0.3; prob = [0, 0, -0.03, 0.035, 0.01, -0.015, 0]

    if Profile_Indx == 10:
        if Light < 2700:
            if Air_Mvmt < 1: TC_var == -0.3; prob = [0, 0, 0.03, -0.03, -0.01, 0.01, 0]
            if Air_Mvmt > 1.2: TC_var == -0.4; prob = [0, 0, -0.03, 0.035, 0.01, -0.015, 0]
        
        if Light > 5000:
            if Air_Mvmt < 1: TC_var = -0.25; prob = [0, 0, 0.02, -0.03, -0.01, 0.01, 0]
            if Air_Mvmt > 1: TC_var = -0.2; prob = [0, 0, 0.01, -0.01, -0.01, 0.01, 0]

    

    return TC_var, prob

# %%
def Attention_Variablity(TC_Value, Emotion_var):

    Emotion_Good_Score = (Emotion_var[3] + Emotion_var[4]) / 2
    Emotion_Bad_Score = (Emotion_var[0] + Emotion_var[1] + Emotion_var[2]) / 3
    Emotion_Diff_Score = Emotion_Good_Score - Emotion_Bad_Score

    Emotion_Low = 0
    Emotion_high = 10

    # Decrease in Attention Case of Optimal Thermal Comfort

    if TC_Value < -2:
        if Emotion_Diff_Score < 0:
            drop = [2,3]                                # Choose from one of the profiles for the drop
            n = random.choice(test_list)
            Attention_drop = n                          # Drop of attention by 2 scales than the previous state
            Attention_increase = 0                      # No increase of attention in this case

        if Emotion_Diff_Score > 0:
            drop = [1,2]                              
            n = random.choice(test_list)
            Attention_drop = n
            Attention_increase = 0
    
    if (TC_Value < -1) & (TC_Value > -2):
        if Emotion_Diff_Score < 0:
            drop = [1,2]                              
            n = random.choice(test_list)
            Attention_drop = n
            Attention_increase = 0
        
        if Emotion_Diff_Score > 0:
            Attention_drop = 1
            Attention_increase = 0
    
    if (TC_Value >= 2):
        drop = [2, 3]                              
        n = random.choice(test_list)
        Attention_drop = n              
        Attention_increase  = 0

    # Increase in Attention Case of Optimal Thermal Comfort

    if (TC_Value < 2) & (TC_Value >= 0):
        if Emotion_Diff_Score < 0:
            Attention_drop = 0
            Attention_increase = 2 
        
        if Emotion_Diff_Score > 0:
            Attention_drop = 0
            Attention_increase = 3

    if (TC_Value <= 1) & (TC_Value >= 0):
        if Emotion_Diff_Score < 0:
            Attention_drop = 0
            Attention_increase = 1
        
        if Emotion_Diff_Score > 0:
            Attention_drop = 0
            Attention_increase = 2
  

    if (TC_Value >= -1) & (TC_Value <= 0):
        if Emotion_Diff_Score < 0:
            Attention_drop = 0
            Attention_increase = 1
        
        if Emotion_Diff_Score > 0:
            Attention_drop = 0
            Attention_increase = 2
    

    Attention_Var = [Attention_drop, Attention_increase]
    
    return Attention_Var



# %%
def State_Score(TC, TC_Var, Emotion, prob, Attention, Attention_Var ):
    
    TC_Value = TC + TC_Var
    Emotion_Value = Emotion + prob

    Attention_Value = Attention + Attention_Var[1] - Attention_Var[0]

    return TC_Value, Emotion_Value, Attention_Value

# %%
def Value_score(TC, TC_Var, Emotion, prob, Attention):

    # Thermal Comfort Score

    TC_Value = TC + TC_Var
    if (TC_Value < -1) or (TC_Value >2):
        TC_score = -1
    
    if (TC_Value >= -1) and (TC_Value <= 2):
        TC_score = 1

    
    #Emotional State
    
    Emotion_var = Emotion + prob
    Emotion_Good_Score = (Emotion_var[3] + Emotion_var[4]) / 2
    Emotion_Bad_Score = (Emotion_var[0] + Emotion_var[1] + Emotion_var[2]) / 3
    Emotion_Diff_Score = Emotion_Good_Score - Emotion_Bad_Score

    if Emotion_Diff_Score > 0:
        Emotion_score = 0.5
    if Emotion_Diff_Score < 0:
        Emotion_score = -0.5
    
    Attention_score = 2 * Attention - 1


    return TC_score, Emotion_score, Attention_score



