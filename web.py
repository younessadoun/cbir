import os
#from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import pickle
import search

class userr:
    def __init__(self, name, pas):
        self.n = name
        self.p = pas

class index:
    n = None
    c = None
    c2= None
    s = None
    t = None
    p = None
    d = None

quer=None
region=None

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def upload_form():
    return render_template('login.html')

@app.route('/logged', methods=['POST'])
def logged():
    valid=False
    uname = request.form['uname']
    psw = request.form['psw']
    #print(uname)
    #print(psw)
    with (open("users.dat", "rb")) as openfile:
        while True:
            try:
                tem=pickle.load(openfile)
                print(tem.n)
                print(tem.p)
                if uname==tem.n and psw==tem.p:
                    valid=True
                    break
            except EOFError:
                break
    if valid==True:
        return render_template('upload.html')
    else:
        return render_template('login.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        #flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        global region
        region = request.form['region']
        # print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        return render_template('upload.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    global quer
    quer = filename
    #print(region)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/my-link')
def my_link():
    print('search started')
    res0 = search.search(UPLOAD_FOLDER + quer,region)
    print(res0)
    res=[]
    i=0
    for im in res0:
        res.append('static/'+res0[i])
        i=i+1
    #print(res)
    search.emptyF('static/uploads/*')
    return render_template('result.html', links=res0)

if __name__ == "__main__":
    app.run()