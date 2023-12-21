import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Carregando os dados
df = pd.read_excel('student_data.xlsx')

# Identificação e apresentação de colunas com valores missing
missing_values = df.isnull().sum()
print("Valores missing por coluna:\n", missing_values)
print('\n')

# Tratamento de valores missing

# Análise das variáveis numéricas
numeric_data = df.select_dtypes(include=[np.number])
numeric_desc = numeric_data.describe()
print("Descrição das variáveis numéricas:\n", numeric_desc)
print('\n')

# Calculando o desvio padrão
std_dev = numeric_data.std()
print("Desvio padrão das variáveis numéricas:\n", std_dev)
print('\n')

# Calculando o coeficiente de variação
cv = (std_dev / numeric_data.mean()) * 100
print("Coeficiente de variação das variáveis numéricas:\n", cv)
print('\n')

# Análise das variáveis de texto
categorical_data = df.select_dtypes(include=[object])
categorical_desc = categorical_data.describe()
print("Descrição das variáveis categóricas:\n", categorical_desc)
print('\n')

# Frequências absolutas e relativas
for col in categorical_data.columns:
    freq_abs = categorical_data[col].value_counts()
    freq_rel = categorical_data[col].value_counts(normalize=True) * 100
    print(f"Frequências absolutas para a coluna {col}:\n", freq_abs)
    print(f"Frequências relativas para a coluna {col}:\n", freq_rel)
    print('\n')

# Agrupando dados com menos de 5% da amostra
for col in categorical_data.columns:
    mask = (categorical_data[col].value_counts(normalize=True) * 100 < 5)
    categorical_data.loc[categorical_data[col].isin(mask.index[mask]), col] = 'Outros'
    print(f"Coluna {col} após agrupamento:\n", categorical_data[col].value_counts())
    print('\n')

# Médias/medianas em relação com a G3
mean_G3 = df.groupby('G3')[numeric_data.columns].mean()
median_G3 = df.groupby('G3')[numeric_data.columns].median()
print("Médias das variáveis em relação à G3:\n", mean_G3)
print("Medianas das variáveis em relação à G3:\n", median_G3)
print('\n')

# Correlação
# Seleciona apenas as colunas numéricas
numeric_df = df.select_dtypes(include=[np.number])

# Calcula a matriz de correlação
correlation_matrix = numeric_df.corr()
print("Matriz de correlação:\n", correlation_matrix)
print('\n')

# Seleção de variáveis
# Aqui você deve selecionar as variáveis de acordo com a sua análise
# Vamos supor que você selecionou todas as variáveis numéricas
selected_variables = numeric_df.columns.tolist()

# Histograma e boxplot das variáveis numéricas selecionadas
for var in selected_variables:
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    df[var].hist(bins=30)
    plt.title('Histograma')
    plt.savefig(f'{var}_histogram.png')

    plt.subplot(1, 2, 2)
    df.boxplot([var])
    plt.title('Boxplot')
    plt.savefig(f'{var}_boxplot.png')
    plt.show()

# Gráfico de Dispersão multivariado (não salvo devido a restrições do formato de saída)

# Regressão linear múltipla
X = df[selected_variables]
y = df['G3']

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print("Equação da regressão: ", model.coef_)
print("RMSE: ", rmse)
print("RMSE / média da variável resposta: ", rmse / y.mean())
print('\n')

# Transformando variáveis de texto em variáveis binárias
binary_data = pd.get_dummies(categorical_data)
print("Dados após transformação em variáveis binárias:\n", binary_data.head())
binary_data.to_csv('student_data_binary.csv', index=False)

# Calculando a matriz de correlação para as variáveis binárias
correlation_matrix_binary = binary_data.corr()
print("Matriz de correlação para as variáveis binárias:\n", correlation_matrix_binary)

# Selecionando 5 variáveis explicativas que não tenham correlação entre si
correlation_matrix_abs = correlation_matrix_binary.abs()
selected_variables = [col for col in correlation_matrix_abs.columns if (correlation_matrix_abs[col] < 0.1).all()][:5]
print("Variáveis selecionadas: ", selected_variables)

# Suponha que 'var1' é a variável numérica selecionada e 'var2' é a terceira variável
var1, var2 = selected_variables[0], selected_variables[2]

# Cria o gráfico de dispersão
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x=var1, y='G3', hue=var2)
plt.title('Gráfico de dispersão multivariado')
plt.xlabel(var1)
plt.ylabel('G3')
plt.show()

# Salvando o dataframe modificado em um novo arquivo csv
df.to_csv('student_data_modified.csv', index=False)
