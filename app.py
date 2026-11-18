# Web version - runs the SAME python model inside the browser using PyScript
import random
from pyscript import document
import airbnb_price as ab

random.seed(42)

data = ab.make_data(600)
means, stds = ab.find_scaling([d[0] for d in data])
X = [ab.scale(d[0], means, stds) for d in data]
y = [d[1] for d in data]
weights, bias = ab.train(X, y, epochs=300, lr=0.2)
mae, r2 = ab.test_model(X, y, weights, bias)


def value(id):
    return document.getElementById(id).value


def checked(id):
    return 1 if document.getElementById(id).checked else 0


def predict_price(event):
    row = [int(value("bedrooms")), int(value("bathrooms")), int(value("guests")),
           float(value("distance")), float(value("rating")), int(value("city"))]
    for name in ab.AMENITIES:
        row.append(checked(name))
    price = max(ab.predict(ab.scale(row, means, stds), weights, bias), 20)
    document.getElementById("result").innerText = "Predicted price: $%.2f per night" % price


def show_importance():
    ranked = sorted(zip(ab.FEATURES, weights), key=lambda p: abs(p[1]), reverse=True)
    html = ""
    for name, w in ranked:
        arrow = "▲ raises" if w > 0 else "▼ lowers"
        html += "<tr><td>%s</td><td>%.2f</td><td>%s</td></tr>" % (name.replace("_", " "), abs(w), arrow)
    document.getElementById("importance").innerHTML = html


show_importance()
document.getElementById("status").innerText = "Model ready! Average error: $%.2f | R2: %.3f (fake data)" % (mae, r2)
document.getElementById("predict-btn").disabled = False
