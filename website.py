from flask import Flask
app = Flask(__name__)

from flask import render_template
import cnn


@app.route('/')
def matrix_choices():
	return render_template('choices.html')


@app.route('/<matrix_type>/<int:n>', defaults={'matrix_type': 'tf', 'n': 10})
def similarities(matrix_type, n):
    data = cnn.process(n, method=matrix_type, randomize=True)
    # sort the data from greatest scores to least
    for k in data:
        if k == 'euclidean':
            data[k].sort(key=lambda c: c.score)
        else:
            data[k].sort(key=lambda c: c.score, reverse=True)

    if 'euclidean' in data:
        max_euclidean_distance = float('-inf')
        for comparison in data['euclidean']:
            max_euclidean_distance = max(max_euclidean_distance, 
                                         comparison.score)
    else:
        max_euclidean_distance = None

    return render_template('similarities.html', similarities=data,
                           euclidean_max=max_euclidean_distance)


if __name__ == '__main__':
    app.run()
