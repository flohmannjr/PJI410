import numpy as np
import pandas as pd

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns

import datetime, os, random

from sklearn.metrics import classification_report, ConfusionMatrixDisplay, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

PALETA = 'summer_r'
CORES  = ['#007F66', '#339966', '#66B266', '#99CC66', '#CCE566']

LINHA_ESPESSURA = 1

CAMADAS = 4

CAMADA_UNIDADES      = 64
CAMADA_INICIALIZADOR = 'he_uniform'
CAMADA_ATIVACAO      = 'relu'

SAIDA_UNIDADES = 1
SAIDA_ATIVACAO = 'linear'

PERDA       = 'mae'
OTIMIZADOR  = Adam
APRENDIZADO = 0.001
METRICAS    = ['mae', 'mse']

ITERACOES = 500

# https://stackoverflow.com/a/66343730

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def set_global_determinism(seed):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

set_global_determinism(seed=SEMENTE)

def avaliar_previsoes(alvos, previsoes):

    print(classification_report(alvos, previsoes))

    ConfusionMatrixDisplay.from_predictions(alvos, previsoes, values_format='d', cmap=PALETA)
    plt.grid(False)

    relatorio = classification_report(alvos, previsoes, output_dict=True)

    return {'acuracia': relatorio['accuracy'],
            'precisao': relatorio['weighted avg']['precision'],
            'revocacao': relatorio['weighted avg']['recall'],
            'pontuacao-f1': relatorio['weighted avg']['f1-score']}

def obter_conjunto(dados_treino, dados_validacao, q_modelos=10):

    conjunto = []

    for m in range(q_modelos):

        print(f'Modelo {m} de {q_modelos}.')

        modelo = Sequential()

        [modelo.add(Dense(units=CAMADA_UNIDADES, kernel_initializer=CAMADA_INICIALIZADOR, activation=CAMADA_ATIVACAO)) for _ in range(CAMADAS)]
        modelo.add(Dense(units=SAIDA_UNIDADES, activation=SAIDA_ATIVACAO))

        modelo.compile(loss=PERDA,
                       optimizer=OTIMIZADOR(learning_rate=APRENDIZADO),
                       metrics=METRICAS)

        modelo.fit(dados_treino,
                   epochs=ITERACOES,
                   validation_data=dados_validacao,
                   verbose=0)
        
        conjunto.append(modelo)

    return conjunto

def obter_previsoes(conjunto, dados_validacao):
    
    previsoes = []

    for modelo in conjunto:
        previsoes.append(modelo.predict(dados_validacao, verbose=0))
    
    return tf.constant(tf.squeeze(previsoes))

def notificacoes_semanais(dados, local):

    ax = sns.lineplot(data=dados, x='Segunda-feira', y='Quantidade', linewidth=LINHA_ESPESSURA, color=CORES[0])

    plt.title(f'Notificações semanais {local}')
    plt.xlabel('')
    plt.ylabel('Quantidade de Notificações')

    plt.xlim([datetime.date(2020, 12, 25), datetime.date(2023, 3, 1)])

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))

    for tick in ax.get_xticklabels(which='both'):
        tick.set_rotation(90)

    plt.grid(visible=True, which='both', axis='both')
    plt.show()

def criar_janelas(dados, janela_tamanho, horizonte_tamanho, premios=[]):

    # Array 2D de 0 a janela_tamanho + horizonte_tamanho.
    janela_primaria = np.expand_dims(np.arange(janela_tamanho + horizonte_tamanho), axis=0)

    # Array 2D com todas as janelas completas com os índices dos dados.
    indices = janela_primaria + np.expand_dims(np.arange(len(dados) - (janela_tamanho + horizonte_tamanho - 1)), axis=0).T

    # Dados em formato de janelas com horizontes.
    janelas_horizontes = dados[indices]

    # Separa os dados em janelas, horizonte.
    if len(premios) == 0:
        janelas = janelas_horizontes[:, :-horizonte_tamanho]
    else:
        janelas = np.column_stack((janelas_horizontes[:, :-horizonte_tamanho], premios[indices[:, -(horizonte_tamanho + 1)]]))

    horizontes = janelas_horizontes[:, -horizonte_tamanho:]

    return janelas, horizontes

def separar_janelas_treino_teste(janelas, horizontes, tamanho_teste=0.2):

    q_teste = int(len(janelas) * (1 - tamanho_teste))

    janelas_treino    = janelas[:q_teste]
    janelas_teste     = janelas[q_teste:]
    horizontes_treino = horizontes[:q_teste]
    horizontes_teste  = horizontes[q_teste:]

    return janelas_treino, janelas_teste, horizontes_treino, horizontes_teste

def grafico_series(X_treino=[], y_treino=[],
                   X_teste=[], y_teste=[],
                   X_previsao=[], y_previsao=[],
                   inicio=0, fim=None, local=''):

    base_legenda = 1
    semanas = []

    if len(X_treino) > 0:
        sns.lineplot(x=X_treino[inicio:fim], y=y_treino[inicio:fim], color=CORES[4], linewidth=LINHA_ESPESSURA, label='Treino')
        semanas.extend(X_treino[inicio:fim])
        base_legenda -= 0.06

    if len(X_teste) > 0:
        sns.lineplot(x=X_teste[inicio:fim], y=y_teste[inicio:fim], color=CORES[0], linewidth=LINHA_ESPESSURA, label='Teste')
        semanas.extend(X_teste[inicio:fim])
        base_legenda -= 0.06

    if len(X_previsao) > 0:
        sns.lineplot(x=X_previsao[inicio:fim], y=y_previsao[inicio:fim], color=CORES[2], linewidth=LINHA_ESPESSURA, label='Previsão')
        semanas.extend(X_previsao[inicio:fim])
        base_legenda -= 0.06

    semanas = np.unique(np.sort(semanas))

    plt.title(f'Notificações semanais{local}')
    plt.xlabel('')
    plt.ylabel('Quantidade de Notificações')

    plt.xticks(ticks=semanas, labels=pd.to_datetime(semanas).strftime('%d/%m'), rotation=90)

    plt.legend(loc=(1.02, base_legenda), frameon=True, facecolor='white')

    plt.show()

def metricas_modelo(y_teste, y_previsao):

    mae = mean_absolute_error(y_teste, y_previsao)
    rmse = np.sqrt(mean_squared_error(y_teste, y_previsao))
    mape = mean_absolute_percentage_error(y_teste, y_previsao)

    return {'Mean Absolute Error': mae,
            'Root Mean Squared Error': rmse,
            'Mean Absolute Percentage Error': mape}

def criar_marco_modelo(modelo_nome, caminho='marcos'):

    return ModelCheckpoint(filepath=os.path.join(caminho, modelo_nome),
                           monitor='val_loss',
                           save_best_only=True,
                           verbose=0)

def gerar_previsoes_futuro(modelo, dados, quantidade_intervalos, janela_tamanho):

    previsoes = []
    janela    = tf.squeeze(dados[-janela_tamanho:])

    for _ in range(quantidade_intervalos):

        previsao = max(1, int(modelo.predict(tf.expand_dims(janela, axis=0), verbose=0)))

        previsoes.append(previsao)

        janela = np.append(janela, previsao)[-janela_tamanho:]

    return previsoes
