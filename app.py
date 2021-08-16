from flask import Flask, request
# from flask_restful import Resource, Api
from flask_cors import CORS
import base64

from bongard_problem import BongardProblem
from solver import BPSolver

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
# api = Api(app)


def get_extension(my_img):
    if b"GIF" in my_img:
        return "GIF"


def checkBase64(my_string):
    try:
        my_img = base64.b64decode(my_string)
        check = base64.b64encode(my_img).decode("utf-8") == my_string
        extension = get_extension(my_img)
        return my_img, check, extension
    except Exception:
        return False, None, None


def base64_to_img(my_string):
    my_img, check, extension = checkBase64(my_string)
    if check:
        filename = 'someimage.jpg'
        with open(filename, 'wb') as f:
            f.write(my_img)
        return True
    else:
        return False


# class Base(Resource):
#     def get(self):
#         return {"about": "hello world"}
#
#     def post(self):
#         some_json = request.get_json()
#         my_string = some_json["test"]
#         base64_to_img(my_string)
#         performance = solve()
#         return {"TEST": performance}, 201


# api.add_resource(Base, "/")

@app.route('/', methods=['GET'])
def get():
    return {"About": "https://github.com/furrutiav/bp-heroku"}


@app.route('/', methods=['POST'])
def post():
    some_json = request.get_json()
    my_string = some_json["image"]
    selection = some_json["selection"]
    n_attributes = some_json["n_att"]
    check = base64_to_img(my_string)
    if check:
        filename = "someimage.jpg"
        bp = BongardProblem(filename)
        if selection == 0:
            s = BPSolver(bp, n_select=n_attributes, alpha_lasso=0, n_lasso=0)
        else:
            s = BPSolver(bp, n_select=0, alpha_lasso=0.1, n_lasso=n_attributes)
        s.default_solve()
        score = s.solved_pd().to_dict()["Score"]
        keys = [key.split("(")[0] for key in score.keys()]
        values = list(score.values())
        output = {keys[i]: v for i, v in enumerate(values)}
        output["solution"] = list(s.columns)
        return {"solved": output}, 201
    else:
        return {}, 404


if __name__ == "__main__":
    app.run(debug=True)
