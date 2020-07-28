

from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from bru_net import bru_regressor as br

    
x, y = load_boston(return_X_y=True)
scaler = StandardScaler()

x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.1, random_state=5, shuffle=True)
x_tr = scaler.fit_transform(x_tr)
x_te = scaler.transform(x_te)

nn = br.BRURegressor(input_shape=x_tr.shape, n_hidden_layers=1, n_units=6,
                     learning_rate=0.01, reg_lambda=0.5, beta=0.9)
nn.fit(x_tr, y_tr, n_iter=1000)
loss = nn.costs
pred = nn.predict(x_te)

mae = mean_absolute_error(y_te, nn.predict(x_te))


plt.plot(loss)
plt.title('Training loss')
plt.show()
plt.close()

# Test "online learning"
new_examples = 16

nn.fit(x_te[0:new_examples, :], y_te[0:new_examples], n_iter=1) 
print('Before sample fit: ', mae)
print('After sample fit: ', mean_absolute_error(y_te[new_examples:], nn.predict(x_te[new_examples:, :])))

