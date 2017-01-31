from flask import Flask
app = Flask(__name__)

from flask import render_template
from lib.interval import Interval

ALERT_TYPE = {
        Interval(0, 33): 'danger',
        Interval(33, 66): 'warning',
        Interval(66, 100): 'success',
}

@app.route('/')
def index():
    return render_template('index.html', similarities=[])

if __name__ == '__main__':
    app.run()
