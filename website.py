def run(data, args):
    from flask import Flask
    app = Flask(__name__, static_url_path='')

    from flask import render_template
    from flask import send_from_directory

    @app.route('/')
    def matrix_choices():
        return render_template('choices.html', num_articles=args.num_to_select)

    @app.route('/<matrix_type>/<int:n>',
               defaults={'matrix_type': 'tf', 'n': 10})
    def similarities(matrix_type, n):
        return render_template('similarities.html', similarities=data)

    @app.route('/get/<filename>')
    def load_article(filename):
        return send_from_directory('results/articles', filename)

    app.run()