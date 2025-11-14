import your_markov  # your algorithm code
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_url_path="", static_folder="static")


# Serve the main HTML file
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# Endpoint that runs your Markov chain
@app.route("/run_markov", methods=["POST"])
def run_markov():
    data = request.json
    steps = data.get("steps", 1000)

    # Run your algorithm (replace with your real function)
    result = your_markov.run_chain(steps)

    # Must return JSON-safe data (lists, dicts, numbers)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
