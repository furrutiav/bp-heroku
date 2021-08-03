from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS
import base64

from time import sleep
from models import solve

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
api = Api(app)


def base64_to_img(my_string):
    my_img = base64.b64decode(my_string)
    filename = 'some_image.jpg'
    with open(filename, 'wb') as f:
        f.write(my_img)


class Base(Resource):
    def get(self):
        return {"about": "hello world"}

    def post(self):
        some_json = request.get_json()
        my_string = some_json["test"]
        base64_to_img(my_string)
        performance = solve()
        return {"TEST": performance}, 201


class Multi(Resource):
    def get(self, num):
        return {"result": num*10}


api.add_resource(Base, "/")
api.add_resource(Multi, "/multi/<int:num>")

if __name__ == "__main__":
    import json
    file_tag_names = open("tags_name.json")
    tag_names = json.load(file_tag_names)
    app.run(debug=True, port=5000)
