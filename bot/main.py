import joblib
# import sql_connector
import numpy as np
from lineBot.function import *
from lineBot.package.package.sql_connector import *
from package.nlp_function import *
#from bag_of_word import bow_dict,scores,temp_category


# question = ['ตาราง', 'เรียน', 'จะ', 'ออก', 'ช่วง', 'ไหน', 'ครับ']
# response_check = True  เอาออกเปลี่ยนเป็น default 0ที่sql แทน

# LOAD SEGMENTATION MODEL
# 0:Negative 1:Positive 2:Question
segment_type = "model/segment_classify/check_segment_type.sav"
segment_vectorizer = "model/segment_classify/segment_vectorizer_word.sav"
segment_model = joblib.load(open(segment_type,"rb"))
vectorizer_segment = joblib.load(open(segment_vectorizer,"rb"))

# LOAD QEUSTION MODEL
# 0:ค่าเทอม 1:Other
question_type = "model/question_classify/check_question_type.sav"
question_vectorizer = "model/question_classify/question_vectorizer_word.sav"
question_model = joblib.load(open(question_type,"rb"))
vectorizer_question = joblib.load(open(question_vectorizer,"rb"))


# รับข้อความจากการแชท และแปลงเป็น ARR
recived_txt="เมนูอาหารอาทิตย์นี้มายังคะ"  
text = [recived_txt]
split_text = split_word(recived_txt)                   # แยกคำจากข้อความใช้ฟังก์ชัน split_word เอาไปใช้ต่อใน word matching
split_text_ai = str(split_word(recived_txt))           # ทำการแปลงผลที่ได้จาก (split_word(recived_txt)) ให้อยู่ในรูปของข้อความ(string) โดยใช้ฟังก์ชัน str()
split_text_ai=text_process_save_comma(split_text_ai)   # ประมวลผลข้อความใช้ text_process_save_comma

# VECTORIZER AND PREDICT SEGMENTATION
# ใช้ (vectorizer_segment) ในการแปลง text เป็น vector โดยใช้ Bag-of-Words TF-IDF และแปลงเป็นอาเรย์ (array)
# ใช้โมเดล (segment_model) ที่ได้ทำการทำนายผล (text_list) โดยใช้ฟังก์ชัน predict เพื่อให้ได้ผลลัพธ์การทำนายของโมเดล
text_list = vectorizer_segment.transform([split_text_ai]).reshape(1,-1).todense()  # นำข้อความที่ถูกแบ่งคำแล้ว (split_text_ai) มาใช้ vectorizer (vectorizer_question) ในการแปลงเป็น vector โดยใช้ transform และ reshape เพื่อเตรียมข้อมูลและแปลงเป็น dense matrix ด้วย todense().
segment_predictions = segment_model.predict(np.asarray(text_list))                # ทำนายประเภทของคำถาม (question type) โดยใช้โมเดลที่โหลดมา (question_model) และข้อมูลที่เตรียมไว้ (text_list) โดยใช้ predict 
predictions = segment_model.predict(np.asarray(text_list)) 

# 0:Negative 1:Positive 2:Question
if(predictions[0]==0): 
    print('Negative Group')
    text_type = "negative"
    reply_text('ขอบคุณสำหรับคำแนะนำค่ะ ทางโรงเรียนชลประทานวิทยาจะดำเนินการแจ้งให้กับหน่วยงานที่เกี่ยวข้องได้รับทราบค่ะ')
elif (predictions[0]==1):
    print('Positive Group')
    text_type = "positive"
    reply_text('โรงเรียนชลประทานวิทยา ขอขอบคุณค่ะ')
else:
    print('Question Group')
    text_type = "question"
    text_list = vectorizer_question.transform([split_text_ai]).reshape(1,-1).todense()  # นำข้อความที่ถูกแบ่งคำแล้ว (split_text_ai) มาใช้ vectorizer (vectorizer_question) ในการแปลงเป็น vector โดยใช้ transform และ reshape เพื่อเตรียมข้อมูลและแปลงเป็น dense matrix ด้วย todense().
    question_predictions = question_model.predict(np.asarray(text_list))                # ทำนายประเภทของคำถาม (question type) โดยใช้โมเดลที่โหลดมา (question_model) และข้อมูลที่เตรียมไว้ (text_list) โดยใช้ predict
    question_type = question_predictions[0]   
    print(f"Prediction: {question_type} ")
    if(question_type==1):
        print("ฝ่ายวิชาการ / โครงการพิเศษ / EP")
    elif(question_type==2):
        print("ฝ่ายโภชนาการ / พัสดุ / สหกรณ์")
    elif(question_type==3):
        print("ฝ่ายปกครอง / กิจการนักเรียน / บริการ ")
    elif(question_type==4):
        print("ฝ่ายธุรการ / ฝ่ายบัญชี / บริหาร")
    elif(question_type==5):
        print("สารสนเทศ")
    else:
        print("ไม่เกี่ยวข้อง")
    

#### ส่งพารามิเตอร์เข้าไปในฟังก์ชัน sql_connector.val #####
# sql_connector.insert_text_and_type(recived_txt,text_type)
