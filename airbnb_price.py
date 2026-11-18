# Airbnb Price Determiner and Predictor
# School AI project - only uses plain python (no libraries to install)
#
# How it works:
# 1. We make some fake Airbnb listings (since we don't have real data)
# 2. We train a simple linear regression model using gradient descent
# 3. We use the model to predict the price of a new listing
# 4. We also show which features matter most for the price

import random

random.seed(42)

# names of the things (features) that decide the price
FEATURES = ["bedrooms", "bathrooms", "max_guests", "distance_km", "rating", "has_pool", "city_tier"]


# ---------- STEP 1: make fake data ----------
def make_data(n=500):
    data = []
    for i in range(n):
        bedrooms = random.randint(1, 5)
        bathrooms = random.randint(1, 3)
        max_guests = bedrooms * 2 + random.randint(0, 2)
        distance_km = round(random.uniform(0.5, 20), 1)
        rating = round(random.uniform(3.0, 5.0), 1)
        has_pool = random.randint(0, 1)
        city_tier = random.randint(1, 3)  # 1 = small town, 3 = big city

        # secret "real" formula + some random noise
        price = (30
                 + bedrooms * 35
                 + bathrooms * 15
                 + max_guests * 5
                 - distance_km * 2.5
                 + (rating - 3) * 20
                 + has_pool * 40
                 + city_tier * 30
                 + random.gauss(0, 10))
        price = max(price, 20)
        row = [bedrooms, bathrooms, max_guests, distance_km, rating, has_pool, city_tier]
        data.append((row, price))
    return data


# ---------- STEP 2: scaling ----------
# we scale every feature so they are all around the same size
# (otherwise gradient descent goes crazy)
def find_scaling(rows):
    means = []
    stds = []
    for j in range(len(rows[0])):
        col = [r[j] for r in rows]
        m = sum(col) / len(col)
        var = sum((x - m) ** 2 for x in col) / len(col)
        means.append(m)
        stds.append(var ** 0.5 if var > 0 else 1)
    return means, stds


def scale(row, means, stds):
    return [(row[j] - means[j]) / stds[j] for j in range(len(row))]


# ---------- STEP 3: train the model ----------
def train(X, y, epochs=1000, lr=0.05):
    n = len(X)
    k = len(X[0])
    weights = [0.0] * k
    bias = 0.0

    for epoch in range(epochs):
        grad_w = [0.0] * k
        grad_b = 0.0
        for i in range(n):
            pred = bias + sum(weights[j] * X[i][j] for j in range(k))
            error = pred - y[i]
            for j in range(k):
                grad_w[j] += error * X[i][j]
            grad_b += error
        # update the weights
        for j in range(k):
            weights[j] -= lr * grad_w[j] / n
        bias -= lr * grad_b / n
    return weights, bias


def predict(row_scaled, weights, bias):
    return bias + sum(w * x for w, x in zip(weights, row_scaled))


# ---------- STEP 4: check how good it is ----------
def test_model(X, y, weights, bias):
    preds = [predict(x, weights, bias) for x in X]
    mae = sum(abs(p - a) for p, a in zip(preds, y)) / len(y)
    mean_y = sum(y) / len(y)
    ss_total = sum((a - mean_y) ** 2 for a in y)
    ss_res = sum((a - p) ** 2 for p, a in zip(preds, y))
    r2 = 1 - ss_res / ss_total
    return mae, r2


# ---------- user input helper ----------
def ask_number(question, low, high, is_int=True):
    while True:
        try:
            value = float(input(question))
            if value < low or value > high:
                print("  please enter a value between", low, "and", high)
                continue
            return int(value) if is_int else value
        except ValueError:
            print("  that is not a number, try again")


def main():
    print("=== Airbnb Price Determiner & Predictor ===")
    print("Training the model, please wait...")

    data = make_data(500)
    random.shuffle(data)
    split = int(len(data) * 0.8)
    train_data = data[:split]
    test_data = data[split:]

    means, stds = find_scaling([d[0] for d in train_data])
    X_train = [scale(d[0], means, stds) for d in train_data]
    y_train = [d[1] for d in train_data]
    X_test = [scale(d[0], means, stds) for d in test_data]
    y_test = [d[1] for d in test_data]

    weights, bias = train(X_train, y_train)
    mae, r2 = test_model(X_test, y_test, weights, bias)
    print("Done! Average error: $%.2f | R2 score: %.3f" % (mae, r2))

    while True:
        print("\n1. Predict price of a listing")
        print("2. See what affects price the most")
        print("3. Exit")
        choice = input("Choose (1/2/3): ").strip()

        if choice == "1":
            bedrooms = ask_number("Bedrooms (1-5): ", 1, 5)
            bathrooms = ask_number("Bathrooms (1-3): ", 1, 3)
            guests = ask_number("Max guests (1-12): ", 1, 12)
            distance = ask_number("Distance from city center in km (0-20): ", 0, 20, False)
            rating = ask_number("Rating (3.0-5.0): ", 3, 5, False)
            pool = ask_number("Has pool? (1 = yes, 0 = no): ", 0, 1)
            tier = ask_number("City size (1 = small town, 2 = medium, 3 = big city): ", 1, 3)

            row = [bedrooms, bathrooms, guests, distance, rating, pool, tier]
            price = predict(scale(row, means, stds), weights, bias)
            price = max(price, 20)
            print("\n>>> Predicted price: $%.2f per night" % price)

        elif choice == "2":
            print("\nFeature importance (bigger number = bigger effect on price):")
            ranked = sorted(zip(FEATURES, weights), key=lambda p: abs(p[1]), reverse=True)
            for name, w in ranked:
                direction = "raises" if w > 0 else "lowers"
                print("  %-12s %6.2f  (%s price)" % (name, abs(w), direction))

        elif choice == "3":
            print("Bye!")
            break
        else:
            print("Please type 1, 2 or 3")


if __name__ == "__main__":
    main()
