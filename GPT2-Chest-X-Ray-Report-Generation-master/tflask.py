# temporary flask application to store images in oimages folder

from flask import Flask, render_template, request
app=Flask("__name__")


@app.route("/",methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/",methods=["POST"])
def predict():
    imagefile=request.files["imagefile"]
    # print(imagefile)
    print(imagefile.filename)
    image_path="./oimages/"+imagefile.filename
    imagefile.save(image_path)
    return render_template('index.html',prediction="temporary")

if __name__=="__main__":
    app.run(port=3000,debug=True)