import os
import tempfile

import flask
import flask_cors

from preprocessing.embedding.facenet import Facenet
from preprocessing.feature_extraction import Features
from core.flicker import Flicker

app = flask.Flask(__name__)
flask_cors.CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 256 * 1024 * 1024

cache_dir = ".cache"
os.makedirs(cache_dir, exist_ok=True)

facenet = Facenet()


@app.route('/testAlive', methods=['GET', 'POST'])
def test_alive():
    return flask.jsonify({
        "status": "ok"
    })


@app.route('/flicker_detection', methods=['POST'])
def func():
    file = flask.request.files['video']
    with tempfile.NamedTemporaryFile() as save_path:
        file.save(save_path.name)
        video_features = Features(facenet, save_path.name, False, cache_dir)
        video_features.feature_extraction()

    flicker = Flicker(video_features.fps, video_features.similarities, video_features.suspects,
                      video_features.horizontal_displacements, video_features.vertical_displacements)

    return flask.jsonify(
        flicker.flicker_detection(output_path="", output=False)
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

