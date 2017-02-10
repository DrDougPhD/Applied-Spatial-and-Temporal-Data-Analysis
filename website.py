from flask import Flask
app = Flask(__name__, static_url_path='')

from flask import render_template
from flask import send_from_directory

data = None

@app.route('/')
def matrix_choices():
	return render_template('choices.html')


@app.route('/<matrix_type>/<int:n>', defaults={'matrix_type': 'tf', 'n': 10})
def similarities(matrix_type, n):
    global data
    return render_template('similarities.html', similarities=data)


@app.route('/get/<filename>')
def load_article(filename):
    return send_from_directory('results/articles', filename)


def run(data):
    global data = data
    app.run()


if __name__ == '__main__':
    print('='*80)
    print('You should use $ python3 cnn.py to run the website')
    print('='*80)
    run()
