# Airbnb Price Determiner & Predictor

A simple school AI project that predicts the nightly price of an Airbnb listing.
Written in plain Python - no libraries needed.

## How to run
```bash
python3 airbnb_price.py
```

## What it does
- Makes 600 fake Airbnb listings (no real dataset)
- Trains a linear regression model with gradient descent
- Asks you to pick bedrooms, bathrooms, guests and city size, then asks yes/no for amenities (pool, wifi, AC, kitchen, parking, TV, washing machine, heater)
- Shows which features affect price the most

## Limitations
The data is fake, so the model only learns the formula we made up. A real
version would use a real dataset (e.g. Inside Airbnb) and more features.
