import ai_sommelier_backend
import streamlit as st

st.set_page_config(
    page_title='AI Sommelier',
    page_icon='🍷',
    layout='wide'
)

sommelier = ai_sommelier_backend

st.title('KI Sommelier')

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
            str(' * **Weintyp**: ' + recommendations[i]['Wine Type'] + '\n' +
                ' * **Land**: ' + recommendations[i]['Country'] + '\n' +
                ' * **Winzer**: ' + recommendations[i]['Winery'] + '\n' +
                '* **Rebsorten**: ' + recommendations[i]['Main grapes'] + '\n' +
                ' * **Übereinstimmung**: ' + "{0:.0%}".format(recommendations[i]['Probability']) + '\n' +
                ' * *Id*: ' + str(recommendations[i]['Id'])
                )
        )
