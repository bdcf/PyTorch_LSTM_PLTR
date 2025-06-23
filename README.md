# LSTM tracking PLTR stock 2022-06-21 to 2025-06-20 in Pytorch

This is an LSTM model made in PyTorch to track and predict stock prices of PLTR from 06/21/22 to 06/21/25. It takes 95% of the data
as training data and 5% as testing data then learns the parameters of the stock value based on the close cost everyday.

## How to Run (Visual Studio Code)

To get this project to work follow these steps:
1. Open in Visual Studio Code.
2. Click the search bar at the top of the project.
3. Navigate to "Show and run Commands" and select.
4. Navigate to "Python: Create Enviorment".
5. Create a venv enviroment.
6. pip install any libraries as needed.
7. Run the project with the python command through terminal.

## About this Project

I made this project as I've always been interested in the stock market and I wanted to get familiar, somewhat, with how AI works in
the industry. To do this I made an LSTM model that tracks the Palantir stock. The reason I choose the Palantir stock is as it is the
stock I've made the most money on in the past. The LSTM I have created seems to fit the line well using random seeds but I still
wouldn't recommend to base your trades off of it as it only tracks close price and does not go into the guts of the company, recent
news etc. 
