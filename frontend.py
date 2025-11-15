import streamlit as st

import pandas as pd
import numpy as np
from numpy.random import default_rng as rng


test_data = pd.DataFrame(
    {
        "col1": list(range(20)) * 3,
        "col2": rng(0).standard_normal(60),
        "col3": ["a"] * 20 + ["b"] * 20 + ["c"] * 20,
    }
)


def show_details():
    pass


def dashboard():
    st.markdown('# Dashboard')
    selected = st.dataframe(
        test_data,
        selection_mode='single-row',
        on_select=show_details,
    )['selection']['rows']
    if selected:
        selected = selected[0]



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
                <div class="stMarkdown" data-testid="stMarkdown" style="display: flex; justify-content: center;">
                    <div data-testid="stMarkdownContainer" class="st-emotion-cache-1nux2oy et2rgd20">
                        <p>
                            <span class="stMarkdownBadge" style="background-color: rgba(255, 164, 33, 0.1); color: rgb(226, 102, 12); font-size: 0.875rem;">⚠️ Needs review</span>
                        </p>
                    </div>
                </div>
            ''')


dashboard_page = st.Page(dashboard, title='Dashboard')
# navigation = st.navigation([st.Page(form), st.Page(dashboard)])
navigation = st.navigation([form, dashboard])
navigation.run()
