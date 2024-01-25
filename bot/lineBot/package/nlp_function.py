import re
import deepcut

# Reply text function
def reply_text(txt):  
    print(txt)        

# Text process
def word_split(text):   
    words = re.split(r",",text) 
    return words                 

# Save, ใช้ในการประมวลผลข้อความที่รับเข้ามา โดยลบอักขระที่กำหนด และตัวเลขออกจากข้อความ          
def text_process_save_comma(text):  
    text = re.sub("\[|\]|'|"," ",text).replace(" ", "")   
    text = re.sub(r'[0-9]+'," ",text).replace(" ", "")    
    return text

# Text process, ใช้ deepcut.tokenize() เพื่อแบ่งคำใน(text) เป็น(tokens) โดยใช้ตัวตัดคำจาก deepcut
def split_word(text):
    tokens = deepcut.tokenize(text) 
    return tokens


