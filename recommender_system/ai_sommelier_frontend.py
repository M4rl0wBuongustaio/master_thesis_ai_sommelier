import ai_sommelier_backend
from datetime import datetime
import streamlit as st
import json
import os

interview_queries_path = \
    '/Users/leonbecker/DataspellProjects/master_thesis_ai_sommelier/recommender_system/interview_queries'


def save_json(data: json):
    with open(interview_queries_path, 'a') as f:
        json.dump(data, f)
        f.write(os.linesep)


st.set_page_config(
    page_title='AI Sommelier',
    page_icon='🍷',
    layout='wide'
)

sommelier = ai_sommelier_backend

st.title('KI Sommelier')

save_results = st.checkbox(label='Resultate speichern', value=False)

with st.form('query_form'):
    num_recommendations = st.selectbox(
        'Wie viele Empfehlungen möchtest Du?',
        (1, 2, 3, 4, 5))
    query = st.text_input(
        label='Bitte beschreibe uns was für einen Wein Du möchtest.',
        max_chars=256,
        placeholder='red full bodied wine with balanced tannin'
    )
    submitted = st.form_submit_button('Get Recommendations')
    if submitted and not query:
        st.warning('Deine Beschreibung muss mindestens aus einem Wort bestehen', icon="⚠️")

cols: st.columns = st.columns(num_recommendations, gap='large')

if submitted and query:
    with st.spinner('Bitte warte kurz, während wir Deine Empfehlungen mit KI erstellen!'):
        recommendations: list = sommelier.get_recommendations(query=query, n=num_recommendations)

    for i in range(len(cols)):
        column = cols[i]
        column.subheader(str(recommendations[i]['Name'] +
                             ' (' +
                             recommendations[i]['Price'] + ')'
                             ))
        column.image(recommendations[i]['Image path'], width=50)
        column.markdown(
            str('* **Weintyp**: ' + recommendations[i]['Wine Type'] + '\n' +
                '* **Land**: ' + recommendations[i]['Country'] + '\n' +
                '* **Region**: ' + recommendations[i]['Region'] + '\n' +
                '* **Winzer**: ' + recommendations[i]['Winery'] + '\n' +
                '* **Rebsorten**: ' + recommendations[i]['Main grapes'] + '\n' +
                '* **Übereinstimmung**: ' + recommendations[i]['Probability'] + '\n' +
                '* *Id*: ' + recommendations[i]['Id'] + '\n' +
                '* **URL**: ' + '[Wein auf Vivino](%s)' % (recommendations[i]['URL'])
                )
        )
    if save_results:
        for i in range(len(cols)):
            recommendations[i]['query'] = query
            recommendations[i]['Zeitstempel'] = datetime.now().isoformat(sep=" ", timespec="seconds")
            save_json(recommendations[i])
