import torch
from sklearn.mixture import BayesianGaussianMixture
import pandas as pd
import subprocess
import fer
from engagement_detection import eng_predict
import os
import cv2
import joblib
import numpy as np

OPENFACE_ROOT_DIR = r"D:/Downloads/OpenFace-OpenFace_2.2.0/OpenFace-OpenFace_2.2.0/x64/Release/"
VIDEO_DIR = r"C:/Users/Saad Mufti/OneDrive - Worcester Polytechnic Institute (wpi.edu)/Computer Science/CS 539 Project/sample_vids"

# print(fer.FER().detect_emotions("C:/Users/Saad Mufti/Downloads/young-focused-architect-working-on-260nw-271150427.webp"))
# Input: Dataframe containing:
# Thermal comfort profiles (2 columns)
# Img path (for emotion recongition)
# Video path (for attention recogntion)
# x coordinate
# y coordinate
class ModelPipeline:
    def __init__(self, df_in, room_img=None) -> None:
        self.detect_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.n_s = len(df_in) if df_in is not None else 0
        # self.cluster_ids = []
        self.room_img = room_img
        self.centers = []
        self.df_in = df_in
        self.individual_res = pd.DataFrame()
        self.cluster_res = pd.DataFrame()
        self.tc_model = joblib.load("Thermal_Comfort_Model1.joblib")
        # self.e_model = fer.FER()

        self.init_eng_preds()

    def init_eng_preds(self):
        self.eng_preds = {}
        files = os.listdir(VIDEO_DIR)
        preds = eng_predict.predict()
        for i, f in enumerate(files):
            self.eng_preds[f] = preds[i]

    def person_detect(self):
        res = self.detect_model(self.room_img)
        df_bb = res.pandas().xyxy[0]
        df_bb = df_bb[df_bb["class"] == 0]
        df_bb["center_x"] = df_bb.apply(lambda r: (r.xmin + r.xmax) / 2, axis=1)
        df_bb["center_y"] = df_bb.apply(lambda r: (r.ymin + r.ymax) / 2, axis=1)

        self.centers = df_bb[["center_x", "center_y"]].to_numpy()
        self.individual_res[["center_x", "center_y"]] = df_bb[["center_x", "center_y"]] 
        self.n_s = len(df_bb)

        img_cv = cv2.imread(self.room_img)
        xmin, xmax = df_bb["xmin"].to_numpy(), df_bb["xmax"].to_numpy()
        ymin, ymax = df_bb["ymin"].to_numpy(), df_bb["ymax"].to_numpy()
        img_crops = [img_cv[int(ymin[i]):int(ymax[i]), int(xmin[i]):int(xmax[i])] for i in range(len(df_bb))]

        if not os.path.exists("./crops/"):
            os.mkdir("./crops/")
        for loc, r in enumerate(img_crops):
            cv2.imwrite(fr"./crops/{loc}.jpg", r)

    def apply_cluster(self):
        bayes_cluster_model = BayesianGaussianMixture(n_components=self.n_s)
        locs = self.df_in[["x", "y"]]
        bayes_cluster_model.fit(locs)
        # self.cluster_ids = bayes_cluster_model.predict(self.centers)
        self.individual_res["cluster_id"] = bayes_cluster_model.predict(locs)

    def emotion_classify(self):
        imgs = [cv2.imread(path) for path in self.df_in["img_path"]]
        fer.FER().detect_emotions("C:/Users/Saad Mufti/Downloads/young-focused-architect-working-on-260nw-271150427.webp")
        emotions_raw = [fer.FER().detect_emotions(img)[0]["emotions"] for img in imgs]
        score_calc = lambda a, d, n, sad, h, f, su: a * 0.25 + d * 0.05 + n * 0.6 + f * 0.05 + sad * 0.08 + su * 0.01 + h * 0.05
        scores = [score_calc(e["angry"], e["disgust"], e["neutral"], e["sad"], e["happy"], e["fear"], e["surprise"])
                        for e in emotions_raw]
        self.individual_res["emotions_raw"] = [i.values() for i in emotions_raw]
        self.individual_res["e_score"] = scores

    def engagement_classify(self):
        # exe = os.path.join(OPENFACE_ROOT_DIR, "/FeatureExtraction.exe")
        # for v in self.df_in["vid_name"].to_numpy():
            # path = os.path.join(VIDEO_DIR, v)
            # subprocess.run([exe, "-f", f'"{path}"', "-wild", "-pose", "-gaze", "-2Dfp", "-3Dfp"])
        vids = self.df_in["video_path"].to_numpy()
        self.individual_res["eng_score"] = np.array(self.eng_preds[v][0][0] for v in vids)

    def tc_classify(self):
        p = pd.DataFrame(self.df_in[f'Profile Info 1'].to_list(), columns = ['age', 'weight', 'height', 'gender'])
        t = pd.DataFrame(self.df_in[f'Temperature'].to_list(), columns = ['temperature'])
        d =  pd.DataFrame(self.df_in[f'Profile Info 2'].to_list(), columns = ['rh', 'met', 'cl', 'hr','st'])
        df = pd.concat([p,t,d], axis = 1, ignore_index = True)

        preds = self.tc_model.predict(df)
        self.individual_res["t_score"] = preds


    def run(self):
        # self.person_detect()
        self.apply_cluster()
        self.emotion_classify()
        self.engagement_classify()
        self.tc_classify()

        input_layer = np.zeros((400,))
        # Input layer to reinforcement model will be 400 neurons, 4 neurons per student
        # 1st = cluster id, 2nd = emotional score, 3rd = engagement score, 4th = thermal comfort score 
        for i in range(self.n_s):
            data = self.individual_res.iloc[i]
            # data["eng_score"] = np.zeros((len(self.data),))
            # print("?", data["cluster_id"])
            input_layer[i + 0] = data["cluster_id"]
            input_layer[i + 1] = data["e_score"]
            input_layer[i + 2] = data["eng_score"]
            input_layer[i + 3] = data["t_score"]
        
        # assert data["cluster_id"].unique()
        print(self.individual_res)
        print(self.individual_res.drop(["emotions_raw"], axis=1).columns)
        df_means = self.individual_res.drop(["emotions_raw"], axis=1).groupby("cluster_id").mean()
        print(df_means.columns)
        input_layer_compact = df_means.to_numpy().flatten()
        input_layer_compact = input_layer_compact[:45]
        return input_layer, input_layer_compact
        
        

    def summary(self):
        print("Pipeline Summary:")
        # print(f"Number of students detected: {self.n_s}")
        print(f"Number of clusters: {len(self.individual_res['cluster_id'].unique())}")



import Environment_Simulation
model = ModelPipeline(Environment_Simulation.pipe_df)
inp, inp_compact = model.run()