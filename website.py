from flask import Flask
app = Flask(__name__)

from flask import render_template
import cnn
import utils


@app.route('/')
def matrix_choices():
	return render_template('choices.html')


@app.route('/<matrix_type>/<int:n>', defaults={'matrix_type': 'tf', 'n': 10})
def similarities(matrix_type, n):
    data = cnn.from_pickle(n)
    if data is None:
      data = cnn.process(n, method=matrix_type, randomize=True)
    else:
      print('Pickle loaded')

    return render_template('similarities.html', similarities=data)


if __name__ == '__main__':
    utils.setup_logger(debug=False, appname='cnn')
    # check if pickle is available. if not, we need to create it from here.
    n = 100
    matrix_type = 'tf'
    data = cnn.from_pickle(n)
    if data is None:
      cnn.process(n=n, method=matrix_type, randomize=True)
    app.run()
