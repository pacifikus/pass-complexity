import time

import pandas as pd
import requests


if __name__ == "__main__":

    while True:
        try:
            time.sleep(5)
            response = requests.get("http://127.0.0.1:6789/lstm_model/predictor")
            df = pd.DataFrame().from_dict(response)
            df.to_csv('data/answer.csv', mode='a', header=False, index=False)
        except KeyboardInterrupt:
            print('Receiving is done...')
            break
