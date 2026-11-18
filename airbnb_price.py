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

# amenities (yes/no things) and how much each one adds to the price in our fake data
AMENITIES = {
    "pool": 40,
    "wifi": 10,
    "air_conditioning": 20,
    "kitchen": 15,
    "parking": 12,
    "tv": 6,
    "washing_machine": 8,
    "heater": 7,
}

NUMBER_FEATURES = ["bedrooms", "bathrooms", "max_guests", "distance_km", "rating", "city_tier"]
FEATURES = NUMBER_FEATURES + list(AMENITIES.keys())


# ---------- STEP 1: make fake data ----------
def make_data(n=600):
    data = []
    for i in range(n):
        bedrooms = random.randint(1, 5)
        bathrooms = random.randint(1, 3)
        max_guests = bedrooms * 2 + random.randint(0, 2)
        distance_km = round(random.uniform(0.5, 20), 1)
        rating = round(random.uniform(3.0, 5.0), 1)
        city_tier = random.randint(1, 3)  # 1 = small town, 3 = big city

        amenity_values = [random.randint(0, 1) for a in AMENITIES]

        # secret "real" formula + some random noise
        price = (30
                 + bedrooms * 35
                 + bathrooms * 15
                 + max_guests * 5
                 - distance_km * 2.5
                 + (rating - 3) * 20
                 + city_tier * 30
                 + random.gauss(0, 10))
        for value, bonus in zip(amenity_values, AMENITIES.values()):
            price += value * bonus
        price = max(price, 20)

        row = [bedrooms, bathrooms, max_guests, distance_km, rating, city_tier] + amenity_values
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


# ---------- user input helpers ----------
def ask_choice(question, options):
    """Show a numbered list of options and return the one the user picks."""
    print("\n" + question)
    for i, opt in enumerate(options, 1):
        print("  %d. %s" % (i, opt))
    while True:
        try:
            pick = int(input("Your choice: "))
            if 1 <= pick <= len(options):
                return pick - 1
        except ValueError:
            pass
        print("  please type a number from 1 to", len(options))


def ask_yes_no(question):
    while True:
        answer = input(question + " (y/n): ").strip().lower()
        if answer in ("y", "yes"):
            return 1
        if answer in ("n", "no"):
            return 0
        print("  please type y or n")


def ask_number(question, low, high):
    while True:
        try:
            value = float(input(question))
            if low <= value <= high:
                return value
            print("  please enter a value between", low, "and", high)
        except ValueError:
            print("  that is not a number, try again")


def get_listing_from_user():
    bedrooms = ask_choice("How many bedrooms?",
                          ["1 bedroom", "2 bedrooms", "3 bedrooms", "4 bedrooms", "5 bedrooms"]) + 1
    bathrooms = ask_choice("How many bathrooms?", ["1 bathroom", "2 bathrooms", "3 bathrooms"]) + 1
    guests = ask_choice("How many guests can stay?",
                        ["2", "4", "6", "8", "10", "12"])
    guests = (guests + 1) * 2
    city = ask_choice("What kind of place is it in?",
                      ["Small town", "Medium city", "Big city"]) + 1
    distance = ask_number("\nDistance from city center in km (0-20): ", 0, 20)
    rating = ask_number("Guest rating (3.0-5.0): ", 3, 5)

    print("\nNow some amenities:")
    amenity_answers = []
    for name in AMENITIES:
        nice_name = name.replace("_", " ")
        amenity_answers.append(ask_yes_no("Does it have " + nice_name + "?"))

    return [bedrooms, bathrooms, guests, distance, rating, city] + amenity_answers


def main():
    print("=== Airbnb Price Determiner & Predictor ===")
    print("Training the model, please wait...")

    data = make_data(600)
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
            row = get_listing_from_user()
            price = predict(scale(row, means, stds), weights, bias)
            price = max(price, 20)
            print("\n>>> Predicted price: $%.2f per night" % price)

        elif choice == "2":
            print("\nFeature importance (bigger number = bigger effect on price):")
            ranked = sorted(zip(FEATURES, weights), key=lambda p: abs(p[1]), reverse=True)
            for name, w in ranked:
                direction = "raises" if w > 0 else "lowers"
                print("  %-17s %6.2f  (%s price)" % (name, abs(w), direction))

        elif choice == "3":
            print("Bye!")
            break
        else:
            print("Please type 1, 2 or 3")


if __name__ == "__main__":
    main()
