import streamlit as st
import pandas as pd
import numpy as nppip
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import plotly.express as px
import altair as alt
import joblib 
import pickle 

# Configurar o layout da página
st.set_page_config(layout="wide")
col1, col2, col3 = st.columns([1,8,1])
col4, col5, col6 = st.columns([2,6,2])
col7, col8 = st.columns(2) 
col9, col10, col11 = st.columns(3) 
col12, col13, col14 = st.columns(3) 


#upload de csv
dados = pd.read_csv('ipea.csv')
agrupado= pd.read_csv('agrupado.csv')
completo = pd.read_csv('df_completo.csv')



with col1:
    st.write("")
with col2:
    st.title(":orange[Prevendo Preços do Petróleo Brent-FOB]")  
with col3:
    st.write("")


with col4:
    st.write("")
with col5:
    st.image('oil-pumps-work-rhythmically-against-backdrop-dusky-sky.jpg', width=500,use_column_width=2, caption='Extração de petróleo- Image by rorozoa on Freepik')
with col6:
    st.write("")


#informações
st.write('''Brent é um tipo de petróleo cru extraído do Mar do Norte, negociado na Bolsa de Valores de Londres( London Stock Exchange-LSE ). A cotação Brent é a referência para os mercados europeu e asiático.''')
st.write('''O Petróleo Brent é extraído do Mar do Norte em países como Noruega, Reino Unido e Dinamarca. Entidades como a  Organização de Petróleo e Gás( OGA ) são responsáveis por supervisionar a extração de petróleo e garantir e eficiência de maneira segura, com responsabilidade ambiental e social. Como outro objetivo, a Organização dos Países Exportadores de Petróle( OPEP ), agrega países que estão unidos para a extração de petróleo em países como: Argélia, Angola, Equador, Gabão, Guiné Equatorial, Irã, Iraque, Kuwait, Líbia, Nigéria, República do Congo (Congo-Brazzaville), Arábia Saudita, Emirados Árabes Unidos e Venezuela. ''')
st.write('''As crises geopolíticas entre países produtores de petróleo  podem afetar a distribuição e abastecimento de clientes, influenciando as bolsas de valores do mundo. Os países produtores petrolíferos do Mar do Norte são aliados para garantir a regulamentação e segurança ambiental e financeira para garantir suas transações de forma eficiente. Alguns países com alto nível de produção podem afetar o preço do barril Brent, é o caso da Arábia Saudita que registra de 12 a 15 por cento da produção mundial de petróleo. ''')
st.write('''Nos dias atuais, demanda global por este combustível fóssil é enorme e insaciável, no entanto alguns países apresentam um mercado com maior dependência do petróleo. Os Estados Unidos, China e países da União Europeia são os maiores consumidores do produto. A produção e separação do petróleo não é sustentável ambientalmente causando o aquecimento global e até o derretimento de geleiras nos polos, trazendo como consequência o aumento do nível do mar. Como foco no futuro, existem organizações e empresas com o propósito de criar meios de energia sustentável e que não agrida o meio ambiente. Essas inovações podem afetar a produção, extração de petróleo, além disso trazer uma redução no preço do barril de petróleo.''')


#Plot de histórico
st.subheader("", divider="orange")
st.subheader("Histórico de Preços do Petróleo Brent-FOB (US$)📈", divider="orange")
with st.container():
    st.write("Período: 2000 à 2024")
    st.line_chart(dados, x='Data', y='Preco_Petroleo_Brent_FOB')
st.write("☞ Analisando a série histórica de preços do petróleo brent no período de 2000 a 2024, observamos inicialmente uma crescente contínua de 2002 a meados de julho 2008, e a partir do pico máximo do período, houve uma queda vertiginosa até outubro de 2009, quando se inicia uma retomada de crescente dos preços. Em 2015, verificamos uma nova queda de preços e os principais motivos são o aumento na produção dos Estados Unidos e uma menor demanda que a esperada nos mercados europeu e asiático. Em 2016, o preço caiu em consequência das preocupações com o crescimento da China, aumento dos estoques americanos e a crise diplomática entre Irã e Arábia Saudita. No ano de 2020, o mundo foi afetado pela pandemia de COVID-19. As bolsas de valores cairam tanto que tiveram negociações suspensas temporariamente em alguns países, além disso, os preços foram afetados pela disputa entre Rússia e Arábia Saudita.")

#Visualizando os dados
st.subheader("", divider="orange")
st.subheader("Análise dos dados 📊", divider="orange")
with st.container():
    col7, col8 = st.columns(2)
    with col7:
        #Subindo e tratando os dados
        st.markdown('Histórico de Preços (US$)')
        st.dataframe(dados.set_index('Data'), use_container_width=True)
    with col8:
        # Exibindo a base de dados
        st.markdown('Médias de Preços por Ano (US$)')
        st.bar_chart(agrupado.set_index('ano'),use_container_width=True)
paragrafo ="""☞ Durante os período de 2011 a 2014, verificamos os maiores picos na média de preços do petróleo mundial, na serie analisada entre 2000 a 2024.
•Em 2011, a alta foi em face da retomada da economica global pós crise financeira de 2008-2009, que surgiu com a bolha imobiliaria dos Estados 
Unidos e teve como conseguência um efeito domino em larga escala em todos os setores no mundo, principalmente no preço do petróleo.
•Em 2012, houve uma sanção NDAA - Lei de autorização de defesa Nacional que proibiu transações e a venda do petróleo do Irá, mantendo o preço 
médio do petroleo global no mesmo patamar de 2011.
•Já em 2013, houve um avanço tecnologico nos Estados Unidos que impulsionou a venda e distribuição do produto, chamado de 'boom do shale oil'.
•Em 2014, houve uma queda de -8.3 percentuais em relação a 2013 devido a produção e extração de petróleo xisto (boom do shale oil) pelos 
Estados Unidos e pelas sanções de retomada de vendas pela OPEP, a fim de equalizar os preços para garantir uma melhor distrinbuição do 
produto colocando os países do Oriente médio em jogo."""
st.text(paragrafo)


#Visualizando as Estatísticas
st.subheader("", divider="orange")
st.subheader("Estatística dos Dados📊", divider="orange")
with st.container():
    col9, col10, col11 = st.columns(3)
    with col9:
        st.metric(label='Preço Mínimo (US$)', value=dados['Preco_Petroleo_Brent_FOB'].min()) 
        menor = dados['Preco_Petroleo_Brent_FOB'].min()
        min_data = dados[['Data','Preco_Petroleo_Brent_FOB']].loc[dados['Preco_Petroleo_Brent_FOB'] == menor].set_index('Data')
        st.write('Data de registro:',min_data)    
        st.write("☞ A queda do preço em abril de 2020 é consequência do colapso da demanda após a crise da Covid-19 e as preocupações do mercado com os impactos do coronavírus e paralização da economia global.")   
    with col10:
        media = dados['Preco_Petroleo_Brent_FOB'].mean()
        formatted_string = "{:.2f}".format(media)
        st.metric(label='Preço Médio (US$)', value=formatted_string)
    with col11:
        st.metric(label='Preço Máximo (US$)', value=dados['Preco_Petroleo_Brent_FOB'].max())
        maior= dados['Preco_Petroleo_Brent_FOB'].max()
        max_data = dados[['Data','Preco_Petroleo_Brent_FOB']].loc[dados['Preco_Petroleo_Brent_FOB'] == maior].set_index('Data')
        st.write('Data de registro:',max_data)  
        st.write("☞ O maior preço foi verificado em julho de 2008. Os riscos crescentes de desaceleração da economia norte-americana e dos indícios de “diminuição de demanda” por combustíveis nos países desenvolvidos (em particular, nos EUA) elevaram os preços.") 
st.write('Diferença Percentual de Preços Entre os Dias Subsequentes(%)')
fig = plt.figure(figsize=(15, 6))
values =completo[ 'Diferenca(%)'].iloc[-180:] 
datas = completo[ 'Data'].iloc[-180:] 
sns.barplot(data=completo.sort_values('Data', ascending=False),x=datas, y=values, color='dodgerblue')
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
plt.gcf().autofmt_xdate()
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.grid(color='lightgrey')
st.pyplot(fig)



#Predições
st.subheader("", divider="orange")
st.subheader("Previsão de  Preço 📈", divider="orange")
st.write('##### Você quer prever os preços do Petróleo Brent?')
num = st.slider("Dias a prever: ", 1, 10,5)

dados = pd.read_csv('ipea.csv')
dados['Data'] = pd.to_datetime(dados['Data'])
for lag in range(1, 3):  # Criou atraso de 1 dia nesse lag
    dados[f'Preco_lag_{lag}'] = dados['Preco_Petroleo_Brent_FOB'].shift(lag)
dados = dados.dropna()

X = dados[['Preco_lag_1','Preco_lag_2']].values
y = dados['Preco_Petroleo_Brent_FOB'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False, random_state=42)
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.2, max_depth=6, random_state=42,loss='squared_error')
model.fit(X_train, y_train)
previsao =  model.predict(X_test)

mse = mean_squared_error(y_test, previsao)
mae = mean_absolute_error(y_test, previsao)
r2 = r2_score(y_test, previsao)

#prevendo novas datas
ultima_data = X[-1].reshape(1, -1)
pred_futuro = []

for _ in range(num):  # para cada dia da próxima semana
    pred_dia_futuro = model.predict(ultima_data)[0]
    pred_futuro.append(pred_dia_futuro)
    ultima_data = np.roll(ultima_data, -1)
    ultima_data[0, -1] = pred_dia_futuro

# As datas correspondentes à próxima semana
prox_data = pd.date_range(dados['Data'].iloc[-1], periods=(num+1), freq='B')[1:]
  
# Selecionar os dados da semana atual (últimos 7 dias do dataset)
datas_sem_atual = dados['Data'].iloc[-num:]
preco_sem_atual = dados['Preco_Petroleo_Brent_FOB'].iloc[-num:]

# Plotar os preços reais da semana atual e as previsões para a próxima semana
st.caption(f'### ☞ Preços Previstos para {num} dias em US$')
df = pd.DataFrame()
df['Data'] = prox_data.values
df['Preco_Previsto'] = pred_futuro
st.dataframe(df)


fig = plt.figure(figsize=(10, 5))
plt.plot(datas_sem_atual, preco_sem_atual, color='royalblue',marker='o',linestyle='-',label='Preços Atuais')
plt.plot(prox_data, pred_futuro, 'r--o', color='tomato',marker='o',linestyle='dashed',label='Previsões para a Próxima Semana')
# Formatar o eixo x para exibir datas
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.gcf().autofmt_xdate()  # Ajustar formato das datas para evitar sobreposição
plt.xlabel('Data', fontsize=7)
plt.ylabel('Preço (US$)', fontsize=7)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.title('Preços Reais e Previsões', fontsize=9)
plt.legend(fontsize=7)
plt.grid(True)
st.pyplot(fig)
 

#Visualizando as Estatísticas
st.subheader("", divider="orange")
col112, col13,col14 = st.columns(3)    
with st.container():
    col12, col13,col14 = st.columns([5,2,3])   
    with col12:
        st.markdown("##### Fontes: Instituto de Pesquisa Econômica Aplicada-IPEA e YFinance") 
    with col13:
        st.link_button("Site do IPEA", "http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view",type="primary")
    with col14:
        st.write("")

 



    