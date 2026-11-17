# Piano transcription

Model zamienia nagranie fortepianu na spektrogram log-mel i rozpoznaje początki nut dla 88 klawiszy. Repozytorium zawiera przygotowanie etykiet z MIDI, trening z walidacją, transkrypcję i odtwarzacz.

## Test na własnym nagraniu

W PowerShellu, w folderze repozytorium, przygotuj środowisko Windows z Pythonem 3.11 i Tkinterem. Przy pierwszym uruchomieniu wykonaj:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

Uruchom odtwarzacz:

```powershell
.\run_player.cmd
```

Kliknij **Wybierz nagranie…** i wskaż własny plik WAV lub FLAC. Po zakończeniu transkrypcji kliknij **Odtwórz od początku**, żeby posłuchać rozpoznanych nut i zobaczyć je na klawiaturze. Wyniki znajdziesz w `results/maestro_600_events/`, w osobnym folderze dla każdego uruchomienia.

Skrypt sam wczytuje wagi best600 z `models/maestro_600_events/best.pt`. To wersja do transkrypcji z oryginalnymi wagami i konfiguracją.
