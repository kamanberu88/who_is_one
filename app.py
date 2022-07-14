# 必要なモジュールを読み込む
# Flask関連
from flask import Flask, render_template, request, redirect, url_for, abort

# PyTorch関連
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import cv2
# Pillow(PIL)、datetime
from PIL import Image, ImageOps
from datetime import datetime

def label2name(member_label):
    if member_label==0:
        name='cr7'

    elif member_label==1:
        name='messi'

    elif member_label==2:
        name='neymar'

    elif member_label==3:
        name='yoona'

    return name



device = torch.device("cpu")
model = 0

model = torchvision.models.resnet18(num_classes=4).to(device)
# 学習モデルをロードする

model.load_state_dict(torch.load("./model_train_net2.pth"))

model = model.eval()
cascade_path = "./haarcascades/haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(cascade_path)
color = (0, 255, 0)  # 検出した顔を囲む四角形の色
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "GET":
        return render_template("index.html")
    if request.method == "POST":
        # アプロードされたファイルをいったん保存する
        f = request.files["file"]
        filepath = "./static/" + datetime.now().strftime("%Y%m%d%H%M%S") + ".png"
        f.save(filepath)
        # 画像ファイルを読み込む

        image1 = cv2.imread(filepath)
        rect = cascade.detectMultiScale(image1, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        if len(rect) > 0:
            for x, y, w, h in rect:
                trim = image1[y:y + h, x:x + w]
                cv2.imwrite('./Face_detection.jpg', trim)
                break

        else:
            return render_template("redirect.html")   #fadce is not recognized

                # PyTorchで扱えるように変換(リサイズ、白黒反転、正規化、次元追加)
            #image = ImageOps.invert(trim)
        transform = transforms.Compose(
                 [transforms.ToTensor(),
                 transforms.Resize(128),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                 ])
        image = transform(trim).unsqueeze(0)
        # 予測を実施
        output = model(image)
        #y = F.softmax(output, dim=1)
        #y=y.item()
        #print(y)
        member_label = output.argmax(dim=1, keepdim=True)  # model出力の中で要素の値が最大となる要素番号がメンバーのラベルとなる
        result = label2name(member_label)
        return render_template("keka.html", filepath1=filepath,filepath2='./Face_detection.jpg', result=result,
                               )




    else:
        return redirect(url_for('index'))


if __name__ == "__main__":
    app.run(debug=True)
