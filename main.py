import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from  sklearn import metrics


data= pd.read_excel('./base/dataset.xlsx')
#print(data)

# verificando object types 
# print(data.dtypes)

# Criando dict passando coluna Portatil com chave
alterando = {'Portatil': {'Smartphone': 1 , 'Tablet': 2}}

# Inserir as alterações no data
data.replace(alterando,inplace=True)
#print(data)

y = data.Portatil 
x = data.drop(columns=['Portatil'])

#treinando
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.4)     #60% para ele treinar
# O computador vai aprender
clf = DecisionTreeClassifier()
clf.fit(x_train,y_train)

#prediter (o computador será testado)
resp_pc = clf.predict(x_test)
gabarito = y_test

print(f'Resultados obtidos: \n{resp_pc}')
print(f'Gabarito----------: \n{gabarito.values}')
print(f'Precisão: {str(metrics.precision_score(gabarito,resp_pc))}')


