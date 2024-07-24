import pandas as pd
import matplotlib.pyplot as plt

# Carregar o arquivo .pkl
file_path = 'cartpole.pkl'
data = pd.read_pickle(file_path)

# Exibir as primeiras linhas do dataframe para entender a estrutura

# Verificar se as colunas 'episodio' e 'reforco' existem
if 'episodio' in data.columns and 'reforco' in data.columns:
    # Criar o gráfico de episódio x reforço
    plt.figure(figsize=(10, 5))
    plt.plot(data['episodio'], data['reforco'], marker='o', linestyle='-', color='b')
    plt.title('Episódio vs Reforço')
    plt.xlabel('Episódio')
    plt.ylabel('Reforço')
    plt.grid(True)
    plt.show()
else:
    print("As colunas 'episodio' e 'reforco' não foram encontradas no arquivo.")

