import os
import tempfile

import flask
import flask_cors

from core.flicker import feature_extraction

app = flask.Flask(__name__)
flask_cors.CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 256 * 1024 * 1024

cache_dir = ".cache"
os.makedirs(cache_dir, exist_ok=True)


@app.route('/flicker/create', methods=['POST'])
def func():
    file = flask.request.files['video']
    save_path = tempfile.NamedTemporaryFile()
    file.save(save_path.name)
    similarities, suspects, horizontal_displacements, vertical_displacements = feature_extraction(
        save_path.name, False, cache_dir)
    return flask.jsonify({
        "status": "ok",
        "similarities2": similarities[0].tolist(),
        "similarities6": similarities[2].tolist(),
        "suspects": suspects.tolist(),
        "horizontal_displacements": horizontal_displacements.tolist(),
        "vertical_displacements": vertical_displacements.tolist()
    })


app.run("0.0.0.0", port=9696, debug=True)
