#!/usr/bin/env python3


from pathlib import Path
from flask import Flask, request, send_from_directory

root_path = Path(__file__).absolute().parents[2]
dist_folder = root_path / 'dist/client'

print(dist_folder)

app = Flask(__name__, static_url_path=str(dist_folder))


# @app.route('/')
# def root():
#     return app.send_static_file('index.html')


@app.route('/<path:path>')
def send_js(path):
    print(f'path: {path}')
    return send_from_directory(dist_folder, path)


if __name__ == "__main__":
    app.run()
