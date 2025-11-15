import pandas as pd
from faker import Faker
import sys

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

    # 2. Prüfen, ob die Spalte 'Name' bereits existiert
    if 'Name' in df.columns:
        print("Warnung: Die Spalte 'Name' existiert bereits und wird überschrieben.")
        # Wir machen weiter und überschreiben sie

    # 3. Generiere die Namen
    num_rows = len(df)
    print(f"Generiere {num_rows} zufällige Namen...")
    names = [fake.name() for _ in range(num_rows)]

    # 4. Füge die 'Name'-Spalte hinzu (zuerst wird sie am Ende hinzugefügt)
    df['Name'] = names

    # 5. Ordne die Spalten neu an, um 'Name' an die erste Stelle zu setzen
    print("Setze 'Name' an die erste Spalte...")

    # Hole alle Spaltennamen
    cols = list(df.columns)

    # Entferne 'Name' aus der Liste (am Ende) und füge es am Anfang ein
    cols.insert(0, cols.pop(cols.index('Name')))

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