import pandas as pd
from keras.models import Sequential
import keras
from keras.layers import Dense
from pathlib import Path
from sklearn.model_selection import train_test_split

path = Path(__file__).parent.absolute()

x = pd.read_csv(
    f'{path}/inputs.csv'
)

y = pd.read_csv(
    f'{path}/outputs.csv'
)

trainer_x, test_x, trainer_y, test_y = train_test_split(
    x, y, test_size=0.25
)

model = Sequential()

model.add(
    Dense(
        input_dim=30,
        units=16,
        activation='relu',
        kernel_initializer='random_uniform'
    )
)

model.add(
    Dense(
        units=1,
        activation='sigmoid'
    )
)

optimizer = keras.optimizers.Adam(
    lr=0.001,
    decay=0.0001,
    clipvalue=0.5
)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)

model.fit(
    trainer_x,
    trainer_y,
    batch_size=10,
    epochs=100
)

predicts = model.predict(test_x)
predicts = predicts > 0.5


result = model.evaluate(test_x, test_y)

print(result)
