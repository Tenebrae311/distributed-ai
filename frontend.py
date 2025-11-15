import streamlit as st

import pandas as pd
import numpy as np
from datetime import datetime
from numpy.random import default_rng as rng
from civiclink.pipeline.main import load_data_train_test, train, get_shap_fig


columns_to_display = [
    'Name',
    'Datum',
    'Betrag',
    'Betrugswahrscheinlichkeit',
]
X_train, X_test, y_train, y_test = load_data_train_test()
model = train(X_train, y_train)
rows_to_predict = X_test.drop(columns=['Name', 'Datum'])
scores = model.predict_proba(rows_to_predict)[:, 1] * 100
X_pred = X_test.copy()
X_pred['Betrugswahrscheinlichkeit'] = scores
X_pred = X_pred.sort_values(by=['Betrugswahrscheinlichkeit'],
ascending=False)
X_pred['Betrag'] = X_pred['Erstattungsbetrag_Erwartet'].map(lambda x: f'{x}€')
X_pred['Betrugswahrscheinlichkeit'] = X_pred['Betrugswahrscheinlichkeit'].map(lambda x: f'{x:.2f}%')
X_pred['Datum'] = X_pred['Datum'].map(lambda x: datetime.strptime(x, '%Y-%m-%d').strftime('%d.%m.%Y'))


def show_details():
    st.page_link(dashboard_page, icon=':material/arrow_back:')

    st.markdown('# Details')
    selected = st.session_state['selected_details']
    row = X_pred.iloc[selected]
    details = row[columns_to_display]
    score = float(details['Betrugswahrscheinlichkeit'][:-2])

    st.markdown(f'''
        ### Name: {details['Name']}
        ### Abgabedatum: {details['Datum']}
        ### Erstattungsbetrag: {details['Betrag']}€
    ''')
    st.download_button(
        'Steuererklärung',
        data='placeholder',
        file_name='Steuererklärung.txt',
        icon=':material/download:',
        mime='application/text'
    )

    st.markdown(f'### Betrugswahrscheinlichkeit: {score}%')
    if score >= 0.6 and score < 0.8:
        st.warning(
            'Mittleres Risiko für eine betrügerische Steuererklärung',
            icon="⚠️",
        )
    if score >= 0.8:
        st.error(
            'Hohes Risiko für eine betrügerische Steuererklärung',
            icon="⚠️",
        )

    st.markdown('''
        ---
        ### Begründung
    ''')
    fig = get_shap_fig(X_test.iloc[selected:selected+1], model)
    st.pyplot(fig)

    st.markdown('''
        ---
        ### Daten
    ''')
    pairs = {}
    for col in columns_to_display:
        pairs[col] = details[col]
    st.table(pairs, border='horizontal')


def dashboard():
    st.markdown('# Dashboard')
    selected = st.dataframe(
        X_pred[columns_to_display],
        selection_mode='single-row',
        on_select='rerun',
    )['selection']['rows']
    if selected:
        st.session_state['selected_details'] = selected[0]
        st.switch_page(details_page)



def form():
    with st.form('tax-form'):
        st.text('Example Tax Form')

        numeric_inputs_data = (
            (
                'Summe_Einkuenfte_Brutto',
                'Summe der Einkünfte (Brutto)',
                85017.15,
            ),
            ('Summe_Werbungskosten', 'Summe Werbungskosten', 6037.81),
            ('Summe_Sonderausgaben', 'Summe Sonderausgaben', 8917.49),
            (
                'Summe_Ausserg_Belastungen',
                'Summe außergewöhnliche Belastungen',
                0,
            ),
            (
                'Erstattungsbetrag_Erwartet',
                'Erwarteter Erstattungsbetrag',
                3395.46,
            ),
            (
                'Anzahl der Homeoffice Tage',
                'Anzahl_Tage_Homeoffice',
                120,
            ),
            (
                'Entfernung zwischen der Wohnung und der Arbeit',
                'Entfernung_Wohnung_Arbeit',
                0,
            ),
            ('Kosten der Arbeitsmittel', 'Kosten_Arbeitsmittel', 1214.67),
            ('Kosten der Bewirtung', 'Kosten_Bewirtung', 2219.01),
            ('Kosten der Geschäftsreisen', 'Kosten_Geschaeftsreisen', 6566.63),
            ('Alter', 'Alter', 60),
            ('Anzahl der Kinder', 'Anzahl_Kinder', 0),
        )
        numeric_inputs = [
            st.number_input(n, key=k, value=v)
            for n,k,v in numeric_inputs_data
        ]

        categoric_inputs_data = (
            (
                'Familienstand',
                'Familienstand',
                ('ledig', 'verheiratet', 'geschieden'),
                2,
            ),
            ('Steuerklasse', 'Steuerklasse', (0, 1, 2, 3, 4), 2),
            (
                'Bundesland',
                'Bundesland',
                (
                    'Baden-Württemberg',
                    'Bayern',
                    'Berlin',
                    'Brandenburg',
                    'Bremen',
                    'Hamburg',
                    'Hessen',
                    'Mecklenburg-Vorpommern',
                    'Niedersachsen',
                    'Nordrhein-Westfalen',
                    'Rheinland-Pfalz',
                    'Saarland',
                    'Sachsen',
                    'Sachsen-Anhalt',
                    'Schleswig-Holstein',
                    'Thüringen',
                ),
                9,
            ),
            (
                'Religionszugehörigkeit',
                'Religionszugehörigkeit',
                (
                    'römisch-katholisch',
                    'evangelisch',
                    'konfessionslos',
                    'muslimisch',
                    'andere',
                    'keine Angabe',
                ),
                3,
            ),
            (
                'Einkunftsart',
                'Einkunftsart',
                (
                    'Nichtselbständige Arbeit',
                    'Selbständige Arbeit',
                    'Gewerbebetrieb',
                    'Kapitalvermögen',
                    'Vermietung und Verpachtung',
                    'Sonstige Einkünfte (z.B. Rente)',
                ),
                2,
            ),
            (
                'Branche (Selbstständig)',
                'Branche_Selbststaendig',
                (
                    'IT-Dienstleistungen',
                    'Handwerk',
                    'Beratung',
                    'Einzelhandel',
                    'Gastronomie',
                    'Gesundheitswesen',
                    'Produktion',
                    'Sonstige',
                ),
                6,
            ),
        )
        categorical_inputs = [
            st.selectbox(n, d, key=k, index=i)
            for n,k,d,i in categoric_inputs_data
        ]

        st.pills(
            'Anlagen',
            ('N', 'V', 'KAP', 'Kind', 'G'),
            selection_mode='multi',
            default=['G'],
            key='anlagen',
        )

        st.divider()
        submitted = st.form_submit_button(
            "Submit",
            use_container_width=True,
        )
        if submitted:
            st.html('''
            <div class="stAlert" data-testid="stAlert"><div role="alert" data-baseweb="notification" data-testid="stAlertContainer" class="stAlertContainer st-ae st-af st-ag st-ah st-ai st-aj st-ak st-ed st-am st-an st-ao st-ap st-aq st-ar st-as st-ee st-au st-av st-aw st-ax st-ay st-ba st-b0 st-b1 st-b2 st-b3 st-b4 st-b5 st-b6 st-b7"><div class="st-b8 st-b9"><div data-testid="stAlertContentWarning" class="st-emotion-cache-14i9r8l en1c6do0" style="
    text-align: center;
"><div class="st-emotion-cache-1xmp9w2 e9q2xfh0"><span class="st-emotion-cache-6jwljf e1t4gh342"><span data-testid="stAlertDynamicIcon" aria-hidden="true" class="st-emotion-cache-8hkptd e1t4gh344">⚠️</span></span><div data-testid="stMarkdownContainer" class="st-emotion-cache-1nux2oy et2rgd20" style="width: calc(100% - 1.75rem);"><p>Fraud detected!</p></div></div></div></div></div></div>
            ''')


details_page = st.Page(show_details, title='Details')
dashboard_page = st.Page(dashboard, title='Dashboard')
# navigation = st.navigation([st.Page(form), dashboard_page, details_page])
navigation = st.navigation([dashboard, show_details])
navigation.run()
