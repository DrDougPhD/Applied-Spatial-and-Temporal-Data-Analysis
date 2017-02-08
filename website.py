from flask import Flask
app = Flask(__name__)

from flask import render_template
from lib.interval import Interval
import cnn

ALERT_TYPE = {
        Interval(0, 33): 'danger',
        Interval(33, 66): 'warning',
        Interval(66, 100): 'success',
}

@app.route('/')
def index():
    n = 5
    data = cnn.process(n)
    # sort the data from greatest scores to least
    for k in data:
        data[k].sort(key=lambda c: c.score, reverse=True)

    if 'euclidean' in data:
        max_euclidean_distance = float('-inf')
        for comparison in data['euclidean']:
            max_euclidean_distance = max(max_euclidean_distance, 
                                         comparison.score)
    else:
        max_euclidean_distance = None

    return render_template('index.html', similarities=data,
                           euclidean_max=max_euclidean_distance)


if __name__ == '__main__':
    app.run()
