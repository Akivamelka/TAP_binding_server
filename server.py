import os
from flask import Flask, request, flash, redirect, render_template, send_from_directory
from werkzeug.utils import secure_filename
from model.FFNN_test import compute_output, compute_single_output

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = os.path.abspath("uploader_folder")
model_path = os.path.abspath("model/model_tap.pth")

@app.route('/', methods=['GET', 'POST'])
def home():
    error_message = False
    results = False
    if request.method == 'POST':
        try:
            if len(request.files) == 0:
                flash('No file part')
                return redirect(request.url)
            # get first file and/or peptide
            f = request.files['the_file']
            peptide_seq = request.form['peptide_seq']
            # if user does not select file, browser also submit an empty part without filename
            new_peptide_seq = ''
            if peptide_seq:
                proba, concentration = compute_single_output(peptide_seq, model_path)
                new_peptide_seq = f"Concentration = {concentration} , Probability = {proba}"
            if f.filename == '':
                return render_template('home.html', error_message=error_message, results=results,
                                       peptide_seq=peptide_seq, new_peptide_seq=new_peptide_seq)
            if f:
                filename = secure_filename(f.filename)
                user_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                f.save(user_file)
                output_file = os.path.join(app.config['UPLOAD_FOLDER'], 'output_for_user.csv')
                compute_output(user_file, output_file, model_path)
                results = True
            return render_template('home.html', error_message=error_message, results=results, peptide_seq=peptide_seq, new_peptide_seq=new_peptide_seq)
        except:
            error_message = True
            results = False
            return render_template('home.html', error_message=error_message)
    return render_template('home.html', error_message=error_message, results=results)


@app.route("/help")
def help():
    return render_template("help.html")


@app.route("/example")
def example():
    return render_template("example.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/download_output")
def download_output():
    return send_from_directory(app.config['UPLOAD_FOLDER'], 'output_for_user.csv', as_attachment=True)

@app.route("/download_example")
def download_example():
    return send_from_directory(directory="static", filename=f"example_file.csv", as_attachment=True)

# clear the cache
@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

if __name__ == "__main__":
    app.run(debug=True)
    # app.run(debug=True, host='0.0.0.0')
