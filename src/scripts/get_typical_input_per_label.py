import pandas as pd
import sys

# Dateiname der Eingabedatei
INPUT_FILENAME = '/home/freetz/hackathon/distributed-ai/data/fake_steuerdaten_labels_not_random.csv'

def find_typical_cases(df):
    """
    Sucht nach einem typischen "Fraud"- und "Not Fraud"-Fall,
    indem ein interner "Risk Score" berechnet wird, der auf den
    Fraud-Regeln aus dem Generator basiert.
    """

    # Berechne einen "Risk Score" - eine höhere Zahl bedeutet "verdächtiger"
    # Diese Logik spiegelt die Regeln aus 'generate_tax_data.py' wider

    risk_score = (
            (abs(df['Abweichung_Werbungskosten_von_Berufsgruppe']) > 0.12) * 2 +
            (abs(df['Differenz_Einkommen_Lohnbescheid']) > 400) * 3 +
            (df['Ratio_Werbungskosten_zu_Einkommen'] > 0.35) * 1 +
            (df['Veraenderung_Werbungskosten_Vj_Prozent'] > 1.5) * 2 +
            ((df['Kosten_Geschaeftsreisen'] + df['Kosten_Bewirtung'] > 10000) & (df['Veraenderung_Werbungskosten_Vj_Prozent'] > 0.8)) * 3 +
            (df['Summe_Ausserg_Belastungen'] > 6000) * 1 +
            (df['Hat_Anlage_G'] & (abs(df['Abweichung_Gewinnmarge_von_Branche']) > 0.08)) * 2
    )

    df['calculated_risk_score'] = risk_score

    # Trenne die Datensätze
    fraud_df = df[df['Label'] == 1].copy()
    not_fraud_df = df[df['Label'] == 0].copy()

    if fraud_df.empty:
        print("Warnung: Keine 1-Fälle im Datensatz gefunden.")
        typical_fraud_case = pd.Series(dtype='object')
    else:
        # Finde den "typischsten" (höchster Score) Fraud-Fall
        # (Wir sortieren, um den Fall mit dem höchsten Score zu finden)
        fraud_df_sorted = fraud_df.sort_values(by='calculated_risk_score', ascending=False)
        typical_fraud_case = fraud_df_sorted.iloc[0]

    if not_fraud_df.empty:
        print("Warnung: Keine 0-Fälle im Datensatz gefunden.")
        typical_not_fraud_case = pd.Series(dtype='object')
    else:
        # Finde den "typischsten" (niedrigster Score) Not-Fraud-Fall
        not_fraud_df_sorted = not_fraud_df.sort_values(by='calculated_risk_score', ascending=True)
        typical_not_fraud_case = not_fraud_df_sorted.iloc[0]

    return typical_fraud_case, typical_not_fraud_case

def main():
    try:
        # Lade die CSV-Datei, achte auf das Trennzeichen und Encoding
        df = pd.read_csv(INPUT_FILENAME, sep=';', encoding='utf-8-sig')
        print(f"Erfolgreich '{INPUT_FILENAME}' mit {len(df)} Zeilen geladen.")
    except FileNotFoundError:
        print(f"FEHLER: Die Datei '{INPUT_FILENAME}' wurde nicht gefunden.")
        print("Bitte stellen Sie sicher, dass 'generate_tax_data.py' zuerst ausgeführt wurde.")
        sys.exit(1)
    except Exception as e:
        print(f"Ein Fehler ist beim Lesen der Datei aufgetreten: {e}")
        sys.exit(1)

    # Finde die Beispielfälle
    positive_case, negative_case = find_typical_cases(df)

    # Setze Pandas-Optionen, um alle Spalten anzuzeigen
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    print("\n" + "="*50)
    print("TYPISCHER POSITIVER FALL (Label: 1)")
    print("="*50)
    if not positive_case.empty:
        print(positive_case)
    else:
        print("Kein positiver Fall gefunden.")

    print("\n" + "="*50)
    print("TYPISCHER NEGATIVER FALL (Label: 0)")
    print("="*50)
    if not negative_case.empty:
        print(negative_case)
    else:
        print("Kein negativer Fall gefunden.")

if __name__ == "__main__":
    main()