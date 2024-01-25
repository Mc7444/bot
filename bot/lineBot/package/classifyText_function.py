import numpy as np
from package.sql_connector import * 
from package.nlp_function import *
import joblib

# LOAD SEGMENTATION MODEL
# 0:Negative 1:Positive 2:Question
segment_type = "model/segment_classify/check_segment_type.sav"
segment_vectorizer = "model/segment_classify/segment_vectorizer_word.sav"
segment_model = joblib.load(open(segment_type, "rb"))
vectorizer_segment = joblib.load(open(segment_vectorizer, "rb"))

# LOAD QUESTION MODEL
# 0:ค่าเทอม 1:Other
question_type = "model/question_classify/check_question_type.sav"
question_vectorizer = "model/question_classify/question_vectorizer_word.sav"
question_model = joblib.load(open(question_type, "rb"))
vectorizer_question = joblib.load(open(question_vectorizer, "rb"))

reply = True

def split_text(recived_txt):
    print("#######Get in spilt_text######")
    split_text = split_word(recived_txt)
    print(f"text:{split_text}")    
    return split_text

def split_text_ai(recived_txt):    
    print("#######Get in spilt_text_ai######")
    split_text_ai = str(split_word(recived_txt))
    split_text_ai = text_process_save_comma(split_text_ai)
    print(f"ai:{split_text_ai}")
    return split_text_ai

# FUNCTION SEGMENTATION MODEL
def classifySegment(text):
    print(f"\n\nMODEL TYPE _____{type(segment_model)} " )
    print(f"\n\nvec TYPE _____{type(vectorizer_segment)} ")
 
    if "สวัสดี" in text:
        feedback = "โรงเรียนชลประทานสวัสดีค่ะ ต้องการติดต่อด้านไหนคะ"
        return (feedback,"Greeting",True)
    
    text_list = vectorizer_segment.transform([text]).reshape(1,-1).todense()  # นำข้อความที่ถูกแบ่งคำแล้ว (text) มาใช้ vectorizer (vectorizer_question) ในการแปลงเป็น vector โดยใช้ transform และ reshape เพื่อเตรียมข้อมูลและแปลงเป็น dense matrix ด้วย todense().
    predictions = segment_model.predict(np.asarray(text_list))                 # ทำนายประเภทของคำถาม (question type) โดยใช้โมเดลที่โหลดมา (question_model) และข้อมูลที่เตรียมไว้ (text_list) โดยใช้ predict 
    print(f"THIS IS predictions::{predictions}")

    # 0:Negative 1:Positive 2:Question
    if(predictions[0]==0): 
        print('Negative Group')
        feedback= 'ขอบคุณสำหรับคำแนะนำค่ะ ทางโรงเรียนชลประทานวิทยาจะดำเนินการแจ้งให้กับหน่วยงานที่เกี่ยวข้องได้รับทราบค่ะ' 
        return (feedback,"negative",True)
    elif (predictions[0]==1):
        print('Positive Group')
        feedback='โรงเรียนชลประทานวิทยา ขอขอบคุณค่ะ'
        return (feedback,"positive",True)
    else:
        print('Question Group')
        return classifyQuestion(text)


# FUNCTION QUESTION MODEL
def classifyQuestion(text):
    print(f"\n\nMODEL TYPE _____{type(question_model)} ")
    print(f"\n\nvec TYPE _____{type(question_vectorizer)} ")

    text_list = vectorizer_question.transform([text]).reshape(1, -1).todense()
    question_predictions = question_model.predict(np.asarray(text_list))
    question_type = question_predictions[0]
    print(f"Prediction: {question_type}")

    menu = " "
    if question_type == 1:      ### type 1 == ฝ่ายวิชาการ / โครงการพิเศษ / EP ###
        menu = "คุณต้องการติดต่อบริการด้านอะไร\n 1.ตารางเรียน\n 2.ตารางสอบ\n 3.หลักสูตร\n 4.ฟอร์มแบบคำร้องขอหลักฐานการศึกษา\n 5.อื่นๆ ติดต่อสอบถามเพิ่มเติม"
    elif question_type == 2:    ### type 2 == ฝ่ายโภชนาการ / พัสดุ / สหกรณ์ ### 
        menu = "คุณต้องการติดต่อบริการด้านอะไร\n 1.เมนูอาหารกลางวันประจำสัปดาห์\n 2.สหกรณ์โรงเรียนชลประทานวิทยา\n 3.อื่นๆ ติดต่อสอบถามเพิ่มเติม"
    elif question_type == 3:    ### type 3 == ฝ่ายปกครอง / กิจการนักเรียน /  กิจกรรมต่างๆ โครงการภายใน-นอก ##
        menu = "คุณต้องการติดต่อบริการด้านอะไร\n 1.ระเบียบการแต่งกายของนักเรียน \n 2.กิจกรรมนักเรียน\n 3.นักศึกษาวิชาทหาร\n 4.อื่นๆ ติดต่อสอบถามเพิ่มเติม"
    elif question_type == 4:    ### type 4 == ฝ่ายธุรการ / ฝ่ายบัญชี / บริหาร ###
        menu = "คุณต้องการติดต่อบริการด้านอะไร\n 1.สมัครเรียน\n 2.สมัครบุคลากร\n 3.ค่าธรรมเนียมการศึกษา\n 4.อื่นๆ ติดต่อสอบถามเพิ่มเติม"
    elif question_type == 5:    ### type 5 == ศูนย์พัฒนาโรงเรียนดิจิทัล ###
        menu = "คุณต้องการติดต่อบริการด้านอะไร\n 1.ศูนย์ฝึกว่ายน้ำ\n  2.ระบบชำระเงิน\n 3.ประกาศผลการเรียน\n 4.ระบบรถรับ-ส่ง\n 5.ระบบโรงอาหารอื่นๆ\n 6.ติดต่อสอบถามเพิ่มเติม"
    else:                       ### type 0 == ไม่เกี่ยวข้อง ###
        menu = "รอสักครู่ ฉันจะรีบติดต่อกลับไปให้เร็วที่สุดนะคะ / I’ll get back to you as soon as possible"

    return (menu,"question",True)