import pandas as pd
from faker import Faker
import sys
from datetime import date

# Dateiname der zu modifizierenden Datei
FILENAME = '/home/freetz/hackathon/distributed-ai/data/fake_steuerdaten_labels_not_random.csv'

# Initialisiere Faker für deutsche Daten
fake = Faker('de_DE')

def main():
    try:
        # 1. Lade die CSV-Datei
        print(f"Lade '{FILENAME}'...")
        df = pd.read_csv(FILENAME, sep=';', encoding='utf-8-sig')
        print(f"Datei mit {len(df)} Zeilen geladen.")

    except FileNotFoundError:
        print(f"FEHLER: Die Datei '{FILENAME}' wurde nicht gefunden.")
        print("Bitte stellen Sie sicher, dass 'generate_tax_data.py' zuerst ausgeführt wurde.")
        sys.exit(1)
    except Exception as e:
        print(f"Ein Fehler ist beim Lesen der Datei aufgetreten: {e}")
        sys.exit(1)

    # 2. Prüfen, ob die Spalten bereits existieren
    if 'Name' in df.columns:
        print("Warnung: Die Spalte 'Name' existiert bereits und wird überschrieben.")
    if 'Datum' in df.columns:
        print("Warnung: Die Spalte 'Datum' existiert bereits und wird überschrieben.")

    # 3. Generiere die Namen und Daten
    num_rows = len(df)

    print(f"Generiere {num_rows} zufällige Namen...")
    names = [fake.name() for _ in range(num_rows)]

    print(f"Generiere {num_rows} zufällige Datumsangaben...")
    # Annahme: Einreichungsdaten für das Steuerjahr 2023, eingereicht im Jahr 2024
    start_date = date(2024, 1, 1)
    end_date = date(2024, 12, 31)
    dates = [fake.date_between(start_date=start_date, end_date=end_date) for _ in range(num_rows)]

    # 4. Füge die Spalten hinzu (zuerst werden sie am Ende hinzugefügt)
    df['Name'] = names
    df['Datum'] = dates

    # 5. Ordne die Spalten neu an, um 'Name' und 'Datum' an den Anfang zu setzen
    print("Setze 'Name' und 'Datum' an die ersten Spalten...")

    # Hole alle Spaltennamen
    cols = list(df.columns)

    # Entferne die neuen Spalten aus der Liste (am Ende)
    cols.pop(cols.index('Name'))
    cols.pop(cols.index('Datum'))

    # Füge sie am Anfang in der gewünschten Reihenfolge ein
    cols.insert(0, 'Datum')
    cols.insert(0, 'Name')

    # Wende die neue Spaltenreihenfolge auf den DataFrame an
    df = df[cols]

    # 6. Speichere die modifizierte Datei
    try:
        print(f"Speichere die Änderungen zurück in '{FILENAME}'...")
        # Wir überschreiben die Originaldatei mit den neuen Daten
        df.to_csv(FILENAME, index=False, sep=';', encoding='utf-8-sig')

        print("\nErfolgreich abgeschlossen. Die Spalte 'Name' wurde hinzugefügt.")
        print("\n--- Neue Datenstruktur (erste 5 Zeilen) ---")
        print(df.head())

    except Exception as e:
        print(f"Ein Fehler ist beim Speichern der Datei aufgetreten: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()