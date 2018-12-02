from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/view_erd')
def view_erd():
    return render_template('view_erd.html')

if __name__ == '__main__':
    app.run(debug=True)
