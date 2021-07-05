import os
import tempfile

import flask
import flask_cors

from preprocessing.feature_extraction import Features

app = flask.Flask(__name__)
flask_cors.CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 256 * 1024 * 1024

cache_dir = ".cache"
os.makedirs(cache_dir, exist_ok=True)


@app.route('/testAlive', methods=['GET', 'POST'])
def test_alive():
    return flask.jsonify({
        "status": "ok"
    })


@app.route('/flicker/create', methods=['POST'])
def func():
    file = flask.request.files['video']
    save_path = tempfile.NamedTemporaryFile()
    file.save(save_path.name)

    video_features = Features(
        save_path.name, False, cache_dir)
    video_features.feature_extraction()

    return flask.jsonify({
        "status": "ok",
        "similarities2": video_features.similarities[0].tolist(),
        "similarities6": video_features.similarities[2].tolist(),
        "suspects": video_features.suspects.tolist(),
        "horizontal_displacements": video_features.horizontal_displacements.tolist(),
        "vertical_displacements": video_features.vertical_displacements.tolist()
    })


app.run("0.0.0.0", port=9696, debug=True)
