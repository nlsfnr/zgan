import flask
import os
import sys
from pathlib import Path
import logging

sys.path.append('..')

from dirk.inference import Inference  # noqa: E402
from dirk.utils import AttrDict  # noqa: E402


def build_app() -> flask.Flask:
    fmt = '[%(asctime)s|%(name)s|%(levelname)s] %(message)s'
    logging.basicConfig(format=fmt, level=logging.INFO)

    app = flask.Flask(__name__)
    config_path = os.getenv('ZGAN_CONFIG', None)
    checkpoint_path = os.getenv('ZGAN_CHECKPOINT', None)
    if config_path is None:
        raise FileNotFoundError('Env var not set: ZGAN_CONFIG')
    if checkpoint_path is None:
        raise FileNotFoundError('Env var not set: ZGAN_CHECKPOINT')

    cfg = AttrDict.from_yaml(Path(config_path))
    inf = Inference(cfg, Path(checkpoint_path))

    @app.route('/best', methods=['GET'])
    def best() -> flask.Response:
        n = int(flask.request.args.get('n', 32))
        pop = int(flask.request.args.get('pop', 1000))
        imgs = inf.best(n, pop)
        return flask.Response(''.join(inf.to_html(img) for img in imgs))

    @app.route('/threshold', methods=['GET'])
    def threshold() -> flask.Response:
        n = int(flask.request.args.get('n', 32))
        threshold = float(flask.request.args.get('threshold', 0.2))
        assert 0 <= threshold < 1.
        imgs = inf.threshold(n, threshold)
        return flask.Response(''.join(inf.to_html(img) for img in imgs))

    return app
