# import mysql.connector #เป็นโมดูลที่ใช้สำหรับเชื่อมต่อและจัดการกับฐานข้อมูล MySQL

# # เชื่อมต่อกับ MySQL Database
# connection = mysql.connector.connect(
#   host="localhost",
#   user="root",
#   password="",
#   database="cpw_sql"
# )

# สร้าง global ที่มีตัวแปร x1, x2, x3 กำหนดค่าเริ่มต้นของตัวแปร x1, x2, x3 ให้เป็น None
# global x1 
# global x2
# global x3
# x1 = None 
# x2 = None
# x3 = None



# def val2(x, y, n):      # รับอาร์กิวเมนต์ x, y, n 
#    global x1, x2, x3    # กำหนดให้ x1, x2, x3 เป็นตัวแปร global 
#    x1 = x               # กำหนดค่าของตัวแปร x1 ให้มีค่าเป็น x, x2 ให้มีค่าเป็น y, x3 ให้มีค่าเป็น n
#    x2 = y  
#    x3 = n 
#    print(x3)
 
   



#def insert_text_and_type(recived_txt,text_type):
    # global x1, x2, x3
    # if(m == True):
    #     m = 1
    # else:
    #     m = 0

    # สร้างตัวแปร db_cursor , connection เป็นตัวแปรที่เก็บการเชื่อมต่อกับฐานข้อมูล 
    # cursor() เป็นเมท็อดที่เรียกใช้บน connection ,ใช้ในการ execute คำสั่ง SQL และดึงผลลัพธ์จากการสั่ง SQL นั้นๆ
    #db_cursor = connection.cursor()

    # คำสั่ง SQL สำหรับการเพิ่มข้อมูลใหม่ลงในตาราง "category" , เพิ่มข้อมูลในคอลัมน์ "recive_text" และ "text_type"
    # ใช้ %s ใน SQL command และใส่ค่าที่ต้องการเพิ่มในฟังก์ชัน execute ของ cursor 

    ##ใช้ if else ในการกำหนดว่าsql command จะส่งแบบใด
    #if(text_type=="question"):  #ถ้าประเภทที่รับมาเป็นคำถาม
    #    sql_command = "INSERT INTO category (recive_text,text_type) VALUES (%s, %s)"
    #else:  #ถ้าประเภทที่รับมาไม่เป็นคำถาม(เป็น คำชมหรือตำหนิ)
    #    sql_command = "INSERT INTO category (recive_text,text_type,response_check) VALUES (%s, %s,1)"


    # val2(z, k, m)      # กำหนด x1, x2, x3 ให้มีค่าเท่ากับ z, k, m โดยใช้ฟังก์ชัน val2
    #db_cursor.execute(sql_command,(recived_txt,text_type)) # Execute คำสั่ง SQL โดยใช้ค่า x1 และ x2 ที่กำหนดในตาราง "category" ผ่าน cursor.
    #connection.commit()


