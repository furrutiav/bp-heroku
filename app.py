from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors import CORS
import base64

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


api.add_resource(Base, "/")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
