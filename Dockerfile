RUN pip install pickle

Pip install joblib

RUN pip install -r requirements.txt

COPY sp500_companies.csv ./sp500_companies.csv
COPY sp500_index.csv ./sp500_index.csv
COPY sp500_stocks.csv ./sp500_stocks.csv


COPY Data_Preprocessing.py ./Data_Preprocessing.py
COPY Modeling.py ./Modeling.py
COPY Shap_Values.py ./Shap_Values.py

RUN python3 Data_Preprocessing.py
RUN python3 Modeling.py
RUN python3 Shap_Values.py

