from flask import Flask
app = Flask(__name__)

from flask import render_template

@app.route('/')
def index():
    return render_template('index.html', similarities=[])

if __name__ == '__main__':
    app.run()
