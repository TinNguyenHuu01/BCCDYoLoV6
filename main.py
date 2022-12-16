import cv2
from flask import Flask, render_template, request
from random import random
from my_yolov6 import my_yolov6
import os


yolov6_model = my_yolov6("best_model.pt","cpu","mydataset.yaml", 640, True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static"

@app.route('/', methods=['GET', 'POST'])

def index():
   if request.method == "POST":
        try:
            # Lấy file gửi lên
            image = request.files['file']
            if image:
                # Lưu file
                path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
                print("Save = ", path_to_save)
                image.save(path_to_save)

                # Convert image to dest size tensor
                frame = cv2.imread(path_to_save)

                frame, ndet = yolov6_model.infer(frame, conf_thres=0.6, iou_thres=0.45)

                if ndet!=0:
                    cv2.imwrite(path_to_save, frame)
                    # Trả về kết quả
                    return render_template("index.html", user_image = image.filename , rand = str(random()),
                                           msg="Tải file lên thành công", ndet = ndet)
                else:
                    return render_template('index.html',user_image = image.filename , rand = str(random()) ,msg='Không nhận diện được vùng tế bào ung thư', ndet = ndet)
            else:
                # Nếu không có file thì yêu cầu tải file
                return render_template('index.html', msg='Hãy chọn file để tải lên', ndet = ndet)

        except Exception as ex:
            # Nếu lỗi thì thông báo
            print(ex)
            return render_template('index.html', msg='Không nhận diện được vùng tế bào ung thư', ndet = ndet)

   else:
        # Nếu là GET thì hiển thị giao diện upload
      return render_template('index.html')
if __name__ == '__main__':
   app.run(host='127.0.0.1', port=8080, debug=True)
