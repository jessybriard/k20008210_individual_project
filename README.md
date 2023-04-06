Predicting Foreign Exchange prices using Commodity prices and AI - Jessy BRIARD

- Individual Project for MSci Computer Science Year 3, King's College London
- Academic Year 2022-23
- Student: Jessy BRIARD (Student number: 20008210)
- Supervisor: Dr. Peter McBurney


INSTRUCTIONS TO BUILD AND RUN THE SOFTWARE:

Installation:
- Download and install Python version 3.8.10 64-bit (Make sure to include pip in the Python installation):
    - Windows or MacOS: \
        Download from https://www.python.org/downloads/release/python-3810/
    - Unix: \
        Run inside a Command-Line Interface: sudo apt-get install python3.8
- Open a Command-Line Interface
- Inside the Command-Line Interface:
    - Navigate to the root of the project (using the 'cd' command)
    - Run one of the following commands to create a Python virtual environment:
        - python -m venv venv
        - python3 -m venv venv
    - Run the following command to activate the virtual environment:
        - Windows: \
            venv\Scripts\activate.bat
        - MacOS or Unix: \
            source venv/bin/activate
    - Run the following command to install the necessary libraries in the Python virtual environment:
        - Windows: \
            .\venv\Scripts\python.exe -m pip install -r requirements.txt
        - MacOS or Unix: \
            venv/bin/python -m pip install -r requirements.txt

The following procedures can be followed once the installation is completed.

Run unit tests and coverage:
- Open a Command-Line Interface
- Inside the Command-Line Interface:
    - Navigate to the root of the project (using the 'cd' command)
    - Run the following command to activate the virtual environment:
        - Windows: \
            venv\Scripts\activate.bat
        - MacOS or Unix: \
            source venv/bin/activate
    - Run the following command to run unit tests and coverage: \
        coverage run --omit="tests*" -m unittest discover && coverage report

Run the Foreign Exchange - Commodities Correlation analysis:
- Open a Command-Line Interface
- Inside the Command-Line Interface:
    - Navigate to the root of the project (using the 'cd' command)
    - Run the following command to activate the virtual environment:
        - Windows: \
            venv\Scripts\activate.bat
        - MacOS or Unix: \
            source venv/bin/activate
    - Run the following command to export the project's root directory to the PYTHONPATH:
        - Windows: \
            set PYTHONPATH=%PYTHONPATH%;.
        - MacOS or Unix: \
            export PYTHONPATH="${PYTHONPATH}:."
    - Run the following command to run the Foreign Exchange - Commodities Correlation analysis:
        - Windows: \
            .\venv\Scripts\python.exe src/commodity_forex_correlation_analysis.py
        - MacOS or Unix: \
            venv/bin/python src/commodity_forex_correlation_analysis.py

Run Performance Evaluation and Hypothesis Testing:
- Open a Command-Line Interface
- Inside the Command-Line Interface:
    - Navigate to the root of the project (using the 'cd' command)
    - Run the following command to activate the virtual environment:
        - Windows: \
            venv\Scripts\activate.bat
        - MacOS or Unix: \
            source venv/bin/activate
    - Run the following command to open a Python shell in the project root:
        - Windows: \
            .\venv\Scripts\python.exe
        - MacOS or Unix: \
            venv/bin/python
    - Inside the Python shell, to run Classification experiments:
        - Import the desired method, run: \
            from src.performance_evaluation_and_comparison import evaluate_and_compare_classification
        - Import and instantiate a scikit-learn Classification model, for example: \
            from sklearn import linear_model \
            model = linear_model.LogisticRegression(solver="liblinear", class_weight="balanced")
        - Pick values for parameters forex_ticker, comdty_tickers, use_close_high_low, and nb_samples, for example: \
            forex_ticker = "AUDUSD=X" \
            comdty_tickers = ["GC=F", "SI=F"] \
            use_close_high_low = True \
            nb_samples = 100
        - Refer to the method's documentation in 'Appendix A: User Guide' to understand the method's parameters and their possible values
        - Run the Performance Evaluation and Comparison: \
            evaluate_and_compare_classification(forex_ticker=forex_ticker, comdty_tickers=comdty_tickers, model=model, use_close_high_low=use_close_high_low, nb_samples=nb_samples)
    - Inside the Python shell, to run Regression experiments:
        - Import the desired method, run: \
            from src.performance_evaluation_and_comparison import evaluate_and_compare_regression
        - Import the PriceAttribute Enum Class: \
            from src.tools.constants import PriceAttribute
        - Import and instantiate a scikit-learn Regression model, for example: \
            from sklearn import linear_model \
            model = linear_model.LinearRegression()
        - Pick values for parameters attribute, forex_ticker, comdty_tickers, use_close_high_low, and nb_samples, for example: \
            attribute = PriceAttribute.CLOSE \
            forex_ticker = "EURCAD=X" \
            comdty_tickers = ["HO=F", "BZ=F", "NG=F", "CL=F", "RB=F"] \
            use_close_high_low = True \
            nb_samples = 100
        - Refer to the method's documentation in 'Appendix A: User Guide' to understand the method's parameters and their possible values
        - Run the Performance Evaluation and Comparison: \
            evaluate_and_compare_regression(attribute=attribute, forex_ticker=forex_ticker, comdty_tickers=comdty_tickers, model=model, use_close_high_low=use_close_high_low, nb_samples=nb_samples)