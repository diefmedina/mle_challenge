from model import DelayModel
import pandas as pd
import os

# Open data to train the model
data = pd.read_csv(os.environ.get("FLIGHT_DELAY_DATA"))

delay_model = DelayModel()

# preprocess
features, target = delay_model.preprocess(data)

# fit
delay_model.fit(features, target)

# save model
delay_model.save_model(os.environ.get('FLIGHT_DELAY_MODEL'))