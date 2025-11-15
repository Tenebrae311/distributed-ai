import pandas as pd
from faker import Faker
import random

# --- Konfiguration ---

# Initialisiere Faker für deutsche Daten
fake = Faker('de_DE')

# Anzahl der zu generierenden Datensätze
NUM_ROWS = 6000

# Vordefinierte Listen für kategoriale Daten
FAMILIENSTAND_OPTIONEN = ['ledig', 'verheiratet', 'geschieden', 'verwitwet']
RELIGION_OPTIONEN = ['römisch-katholisch', 'evangelisch', 'konfessionslos', 'muslimisch', 'andere', 'keine Angabe']
EINKUNFTSART_OPTIONEN = [
    'Nichtselbständige Arbeit',
    'Selbständige Arbeit',
    'Gewerbebetrieb',
    'Kapitalvermögen',
    'Vermietung und Verpachtung',
    'Sonstige Einkünfte (z.B. Rente)'
]
BRANCHE_OPTIONEN = ['IT-Dienstleistungen', 'Handwerk', 'Beratung', 'Einzelhandel', 'Gastronomie', 'Gesundheitswesen', 'Produktion', 'Sonstige']
BUNDESLAND_OPTIONEN = [
    'Baden-Württemberg', 'Bayern', 'Berlin', 'Brandenburg', 'Bremen', 'Hamburg',
    'Hessen', 'Mecklenburg-Vorpommern', 'Niedersachsen', 'Nordrhein-Westfalen',
    'Rheinland-Pfalz', 'Saarland', 'Sachsen', 'Sachsen-Anhalt', 'Schleswig-Holstein', 'Thüringen'
]

# --- Hilfsfunktionen ---

def get_steuerklasse(familienstand):
    """
    Simuliert eine plausible Steuerklasse basierend auf dem Familienstand.
    Dies ist eine Vereinfachung.
    """
    if familienstand == 'ledig':
        return random.choice([1, 2]) # 2 für Alleinerziehende
    elif familienstand == 'verheiratet':
        return random.choice([3, 4, 5]) # 4/4, 3/5
    elif familienstand == 'geschieden' or familienstand == 'verwitwet':
        return random.choice([1, 2, 3]) # 3 im Trennungsjahr/Witwenjahr
    return 1

def pyfloat(min_val, max_val, digits=2):
    """ Erzeugt einen Float-Wert in einem Bereich mit Rundung. """
    return round(random.uniform(min_val, max_val), digits)

# --- Datengenerierung ---

print(f"Generiere {NUM_ROWS} Datensätze...")

# Liste zum Speichern der einzelnen Datensätze (als Dictionaries)
data = []

NUM_DATASAETS = 1
for dataset_index in range(NUM_DATASAETS):
    for _ in range(NUM_ROWS):

        # --- Basis-Kategorien & Numerik ---
        alter = random.randint(18, 67)
        anzahl_kinder = random.choices([0, 1, 2, 3, 4], weights=[0.5, 0.25, 0.15, 0.07, 0.03], k=1)[0]
        familienstand = random.choice(FAMILIENSTAND_OPTIONEN)
        steuerklasse = get_steuerklasse(familienstand)
        bundesland = random.choice(BUNDESLAND_OPTIONEN)
        religionszugehoerigkeit = random.choice(RELIGION_OPTIONEN)
        einkunftsart = random.choice(EINKUNFTSART_OPTIONEN)

        # --- Abhängige Kategorien ---
        branche_selbstaendig = "N/A"
        if einkunftsart in ['Selbständige Arbeit', 'Gewerbebetrieb']:
            branche_selbstaendig = random.choice(BRANCHE_OPTIONEN)

        hat_anlage_kind = (anzahl_kinder > 0)
        hat_anlage_g = (einkunftsart == 'Gewerbebetrieb')
        hat_anlage_n = (einkunftsart == 'Nichtselbständige Arbeit')

        # Vereinfachung für V & KAP
        hat_anlage_v = random.choices([True, False], weights=[0.2, 0.8], k=1)[0]
        hat_anlage_kap = random.choices([True, False], weights=[0.3, 0.7], k=1)[0]


        # --- Numerische Hauptdaten ---
        summe_einkuenfte_brutto = pyfloat(12000, 180000)

        # Werbungskosten: Pauschbetrag (1230) oder höher
        summe_werbungskosten = pyfloat(1000, 8000)
        if einkunftsart == 'Nichtselbständige Arbeit':
            summe_werbungskosten = random.choices(
                [1230, pyfloat(1231, 8000)],
                weights=[0.6, 0.4],
                k=1)[0]

        # Sonderausgaben (z.B. Versicherungen)
        summe_sonderausgaben = pyfloat(1500, 10000)

        # Außerg. Belastungen (oft 0)
        summe_ausserg_belastungen = random.choices(
            [0, pyfloat(500, 7000)],
            weights=[0.8, 0.2],
            k=1)[0]

        # Erwartete Erstattung (kann auch negativ sein -> Nachzahlung)
        erstattungsbetrag_erwartet = pyfloat(-3000, 5000)

        # Weitere numerische Daten
        anzahl_tage_homeoffice = random.randint(0, 230)
        entfernung_wohnung_arbeit = 0
        if hat_anlage_n:
            entfernung_wohnung_arbeit = random.randint(1, 80)

        kosten_arbeitsmittel = pyfloat(0, 1500)
        kosten_bewirtung = pyfloat(0, 3000)
        kosten_geschaeftsreisen = pyfloat(0, 10000)

        # --- Zusätzliche Daten (Feature Engineering) - Simuliert als Zufallswerte ---
        veraenderung_einkommen_vj_prozent = pyfloat(-0.1, 0.3) # -10% bis +30%
        veraenderung_werbungskosten_vj_prozent = pyfloat(-0.2, 0.5)
        veraenderung_spenden_vj_prozent = pyfloat(-0.5, 2.0)

        differenz_einkommen_lohnbescheid = pyfloat(-500, 500)
        differenz_kapitalertraege_bank = pyfloat(-100, 100)
        differenz_rente_meldung = pyfloat(-50, 50)

        ratio_werbungskosten_zu_einkommen = (summe_werbungskosten / summe_einkuenfte_brutto) if summe_einkuenfte_brutto > 0 else 0
        ratio_spenden_zu_einkommen = pyfloat(0, 0.05) # Simuliert
        ratio_krankheitskosten_zu_einkommen = (summe_ausserg_belastungen / summe_einkuenfte_brutto) if summe_einkuenfte_brutto > 0 and summe_ausserg_belastungen > 0 else 0
        ratio_gewinn_zu_umsatz = pyfloat(0.05, 0.6) if hat_anlage_g else 0

        abweichung_werbungskosten_von_berufsgruppe = pyfloat(-0.15, 0.15)
        abweichung_gewinnmarge_von_branche = pyfloat(-0.1, 0.1) if hat_anlage_g else 0

        # --- Label (Zielvariable) ---
        # Stark unausgeglichen
        label = random.choices([0, 1], weights=[0.97, 0.03], k=1)[0]  # TODO: Das Label sollte zu 10% vorhanden sein

        # --- Datensatz zusammenstellen ---
        row = {
            # Numerische Daten
            'Summe_Einkuenfte_Brutto': summe_einkuenfte_brutto,
            'Summe_Werbungskosten': summe_werbungskosten,
            'Summe_Sonderausgaben': summe_sonderausgaben,
            'Summe_Ausserg_Belastungen': summe_ausserg_belastungen,
            'Erstattungsbetrag_Erwartet': erstattungsbetrag_erwartet,
            'Anzahl_Tage_Homeoffice': anzahl_tage_homeoffice,
            'Entfernung_Wohnung_Arbeit': entfernung_wohnung_arbeit,
            'Kosten_Arbeitsmittel': kosten_arbeitsmittel,
            'Kosten_Bewirtung': kosten_bewirtung,
            'Kosten_Geschaeftsreisen': kosten_geschaeftsreisen,
            'Alter': alter,
            'Anzahl_Kinder': anzahl_kinder,

            # Kategoriale Daten
            'Familienstand': familienstand,
            'Steuerklasse': steuerklasse,
            'Bundesland': bundesland,
            'Religionszugehörigkeit': religionszugehoerigkeit,
            'Einkunftsart': einkunftsart,
            'Branche_Selbststaendig': branche_selbstaendig,
            'Hat_Anlage_N': hat_anlage_n,
            'Hat_Anlage_V': hat_anlage_v,
            'Hat_Anlage_KAP': hat_anlage_kap,
            'Hat_Anlage_Kind': hat_anlage_kind,
            'Hat_Anlage_G': hat_anlage_g,

            # Zusätzliche Daten
            'Veraenderung_Einkommen_Vj_Prozent': veraenderung_einkommen_vj_prozent,
            'Veraenderung_Werbungskosten_Vj_Prozent': veraenderung_werbungskosten_vj_prozent,
            'Veraenderung_Spenden_Vj_Prozent': veraenderung_spenden_vj_prozent,
            'Differenz_Einkommen_Lohnbescheid': differenz_einkommen_lohnbescheid,
            'Differenz_Kapitalertraege_Bank': differenz_kapitalertraege_bank,
            'Differenz_Rente_Meldung': differenz_rente_meldung,
            'Ratio_Werbungskosten_zu_Einkommen': ratio_werbungskosten_zu_einkommen,
            'Ratio_Spenden_zu_Einkommen': ratio_spenden_zu_einkommen,
            'Ratio_Krankheitskosten_zu_Einkommen': ratio_krankheitskosten_zu_einkommen,
            'Ratio_Gewinn_zu_Umsatz': ratio_gewinn_zu_umsatz,
            'Abweichung_Werbungskosten_von_Berufsgruppe': abweichung_werbungskosten_von_berufsgruppe,
            'Abweichung_Gewinnmarge_von_Branche': abweichung_gewinnmarge_von_branche,

            # Label
            'Label': label
        }
        data.append(row)

    # --- DataFrame erstellen und speichern ---

    df = pd.DataFrame(data)

    # Zeige die ersten paar Zeilen an
    print("\n--- Generierte Daten (erste 5 Zeilen) ---")
    print(df.head())

    # Zeige Datentypen an
    print("\n--- Spalten und Datentypen ---")
    print(df.info())

    # Speichere als CSV-Datei
    # Wir verwenden ; als Trennzeichen und utf-8-sig für korrekte Excel-Anzeige von Umlauten
    output_filename = f'fake_steuerdaten_{dataset_index}.csv'
    df.to_csv(output_filename, index=False, sep=';', encoding='utf-8-sig')

    print(f"\nErfolgreich {len(df)} Datensätze generiert und in '{output_filename}' gespeichert.")