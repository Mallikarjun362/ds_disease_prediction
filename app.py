
# from pywebio import start_server
# from pywebio.input import *
# from pywebio.output import *


from pywebio.platform.flask import webio_view
from pywebio import STATIC_PATH
from flask import Flask, send_from_directory
from pywebio.input import *
from pywebio.output import *
import argparse
from pywebio import start_server
import pandas as pd
import pickle
import numpy as np


symptoms = ['None','abdominal_pain',
 'abnormal_menstruation',
 'acidity',
 'acute_liver_failure',
 'altered_sensorium',
 'anxiety',
 'back_pain',
 'belly_pain',
 'blackheads',
 'bladder_discomfort',
 'blister',
 'blood_in_sputum',
 'bloody_stool',
 'blurred_and_distorted_vision',
 'breathlessness',
 'brittle_nails',
 'bruising',
 'burning_micturition',
 'chest_pain',
 'chills',
 'cold_hands_and_feets',
 'coma',
 'congestion',
 'constipation',
 'continuous_feel_of_urine',
 'continuous_sneezing',
 'cough',
 'cramps',
 'dark_urine',
 'dehydration',
 'depression',
 'diarrhoea',
 'dischromic_patches',
 'distention_of_abdomen',
 'dizziness',
 'drying_and_tingling_lips',
 'enlarged_thyroid',
 'excessive_hunger',
 'extra_marital_contacts',
 'family_history',
 'fast_heart_rate',
 'fatigue',
 'fluid_overload',
 'foul_smell_ofurine',
 'headache',
 'high_fever',
 'hip_joint_pain',
 'history_of_alcohol_consumption',
 'increased_appetite',
 'indigestion',
 'inflammatory_nails',
 'internal_itching',
 'irregular_sugar_level',
 'irritability',
 'irritation_in_anus',
 'itching',
 'joint_pain',
 'knee_pain',
 'lack_of_concentration',
 'lethargy',
 'loss_of_appetite',
 'loss_of_balance',
 'loss_of_smell',
 'malaise',
 'mild_fever',
 'mood_swings',
 'movement_stiffness',
 'mucoid_sputum',
 'muscle_pain',
 'muscle_wasting',
 'muscle_weakness',
 'nausea',
 'neck_pain',
 'nodal_skin_eruptions',
 'obesity',
 'pain_behind_the_eyes',
 'pain_during_bowel_movements',
 'pain_in_anal_region',
 'painful_walking',
 'palpitations',
 'passage_of_gases',
 'patches_in_throat',
 'phlegm',
 'polyuria',
 'prognosis',
 'prominent_veins_on_calf',
 'puffy_face_and_eyes',
 'pus_filled_pimples',
 'receiving_blood_transfusion',
 'receiving_unsterile_injections',
 'red_sore_around_nose',
 'red_spots_over_body',
 'redness_of_eyes',
 'restlessness',
 'runny_nose',
 'rusty_sputum',
 'scurring',
 'shivering',
 'silver_like_dusting',
 'sinus_pressure',
 'skin_peeling',
 'skin_rash',
 'slurred_speech',
 'small_dents_in_nails',
 'spinning_movements',
 'spotting_urination',
 'stiff_neck',
 'stomach_bleeding',
 'stomach_pain',
 'sunken_eyes',
 'sweating',
 'swelled_lymph_nodes',
 'swelling_joints',
 'swelling_of_stomach',
 'swollen_blood_vessels',
 'swollen_extremeties',
 'swollen_legs',
 'throat_irritation',
 'toxic_look_(typhos)',
 'ulcers_on_tongue',
 'unsteadiness',
 'visual_disturbances',
 'vomiting',
 'watering_from_eyes',
 'weakness_in_limbs',
 'weakness_of_one_body_side',
 'weight_gain',
 'weight_loss',
 'yellow_crust_ooze',
 'yellow_urine',
 'yellowing_of_eyes',
 'yellowish_skin',]

import pickle
import pandas as pd

def toarr(ls):
    d = []
    for i in symptoms:
        if i == 'None':
            continue
        if i in ls:
            d.append(1)
        else:
            d.append(0)
    return d
#trained prediction model loading from pickel file
model = None
with open("./model_disease_prediction.pkl","rb") as f:
    model = pickle.load(f)
disc = pd.read_csv("./dataset/symptom_Description.csv") # Discription of the disease
pre = pd.read_csv("./dataset/symptom_precaution.csv") # prections of the Disease


css_style = """
<style>
@import url('https://fonts.googleapis.com/css?family=Roboto');

@import url('https://fonts.googleapis.com/css?family=Source+Sans+Pro:200,200i,300,300i,400,400i,600,600i,700,700i,900,900i');

body, html{
  padding: 0;
  margin: 0;
  width: 100%;
}

hr{
  border-top: 1px solid #cccccc;
}

.example-container{
  width: 50%;
  margin: 30px auto;
  font-family: "Source Sans Pro";
}

.chip-container{
  border: 1px solid #cccccc;
  width: 100%;
  padding: 10px;
  background: #ffffff;
  font-family: "Roboto";
}

.chip-container .basic-chip a{
  color: inherit;
}

.chip-title{
  padding: 20px 10px;
  font-size: 14px;
}

.basic-chip{
  padding: 5px 10px;
  border-radius: 50px;
  display: inline-flex;
  margin: 5px;
}

.click-chip, .click-chip-hover{
  padding: 5px 10px;
  border-radius: 50px;
  display: inline-flex;
  margin: 5px;
  cursor: pointer;
}

.click-chip-hover:hover{
  filter: brightness(85%);
}

.outline{
  border: 1.5px solid #cccccc;
  color: #cccccc;
  outline: none;
}

.outline-green{
  border: 1.5px solid #3a913f;
  color: #3a913f;
  outline: none;
}

.outline-blue{
  border: 1.5px solid #0074c3;
  color: #0074c3;
  outline: none;
}

.background-grey{
  background: #dddddd;
  color: #616161;
}

.background-green{
  background: #3a913f;
  color: #ffffff;
}

.background-blue{
  background: #0074c3;
  color: #ffffff;
}

.icon{
  /*background: #bdbdbd;*/
  color: #616161;
  margin: 0px 5px 0px -5px;
  background: #bbbbbb;
  padding: 2px;
  border-radius: 50px;
}

.fa-times, .fa-times-circle{
  margin: 0px -5px 0px 5px;
}
</style>
"""



def html_list_to_chips(ls):
    html_c = []
    for i  in ls:
        html_c.append(f'<div class="basic-chip background-grey">{i.replace("_"," ")}</div>')
    #adding style
    #adding html
    content = ""
    for i in html_c:
        content += i
    content = css_style + content
    h = f"<div>{content}</div>"
    return h



## DISPLAYING BACK ON THE SCREEN
#displaying back to the screen


def main():
    # while True:
        ## TAKING INPUT part
        #code for taking inputs through the web interface
        put_html("""
        <center style="color:#0B94E2">
        <h1>
        Disease Prediction
        <span>
        <img src="https://www.creativefabrica.com/wp-content/uploads/2020/02/16/Medical-Logo-Graphics-1-29-580x386.jpg" style="height:80px;width:120px"/>
        </span>
        </h1>
        <center>
        """)
        inpts = []
        for i in range(1,18):
            inpts.append(select(f"symptom {i}",symptoms,name=f"s{i}"))




        data = input_group("Symptoms",inpts) #and extracting all the symptoms from the "data" and  storing in "symptoms_list" 

        input_symptoms_list = list(data.values())

        x = toarr(input_symptoms_list) #converting the "input_symptoms_list" to an array of 0s and 1s 

        ## PREDICTING part
        #storing the predicted result in "p"
        output_disease = model.predict([x])[0]
        print(output_disease)
        output_discription = disc[disc["Disease"]==output_disease]["Description"].values
        if len(output_discription)==0:
            output_discription = ""
        else:
            output_discription = output_discription[0]
        print(output_discription)
        pre1 = pre.loc[pre["Disease"]==output_disease].iloc[0]
        precation_list = []

        for i in pre1[["Precaution_1","Precaution_2","Precaution_3","Precaution_4"]]:
            if str(i) == "nan":
                continue
            precation_list.append(str(i))
        print(precation_list)

        input_symptoms_list = set([i for i in input_symptoms_list if str(i) != "None"])

        put_html(f"<h1>{output_disease}</h1>")
        put_html("<h4>Entered Symptoms :</h4>")
        put_html(html_list_to_chips(input_symptoms_list))
        put_html("<br/>"+output_discription)
        put_html("<h4>Precations :</h4>")
        for i,pr in enumerate(precation_list):
            put_html(f"<p style='left-padding:15px'>{i+1} . <big style='color:grey'>{pr}</big></p>")
        put_html("<center><a href='/' style='background-color: #F34423DD;color: white;padding: 1em 1.5em;text-decoration: none;text-transform: uppercase;'><big>Home<big></a></center>")
# if __name__ == "__main__":
#     start_server(main, port=8080, debug=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=8080)
    args = parser.parse_args()
    start_server(main, port=args.port)

#   background-color: red;color: white;padding: 1em 1.5em;text-decoration: none;text-transform: uppercase;
