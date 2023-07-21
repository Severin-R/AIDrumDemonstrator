import os
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from pyroomacoustics import ShoeBox
from ipywidgets import IntSlider
from IPython.display import Audio, display, clear_output, HTML
from ipywidgets import Button, HBox, VBox, Output, Dropdown, Textarea, Layout, Label, IntSlider, FloatSlider
from ipywidgets import HTML as HTMLWidget
import librosa
from google.colab import files


from my_models import WaveGAN_Model, Crash_Model, Granular_Synthesis_Model
test_gen_path = os.path.join(base_path, "neural_granular_synthesis-master/codes/outputs/generations/")
class_folders = ['Clap', 'Combo', 'Cymbal', 'HandDrum', 'HiHat', 'Kick', 'MalletDrum', 'Metallic', 'Shaker', 'Snare', 'Tom', 'Wooden']

waveGAN = WaveGAN_Model()
crash = Crash_Model()
granular = Granular_Synthesis_Model()


# Referenzen der Modelle
option_to_models = {
    "Granular Sound Synthesis": granular,
    "WaveGAN (slow)": waveGAN,
    "WaveGAN (unconditioned)": (waveGAN, 1),
    "CRASH (very slow)": crash,
    "CRASH (unconditioned)": (crash, 1)
}


# Daten für die Tabelle
data = {
    'Granular Sound Synthesis': [['Inception Score:', '4.7012', 'Frechet Audio Distance:', '3.9140'],
                                 ['Number of different Bins:', '9', 'Generierungszeit:', '31.3 ms']],
    'WaveGAN (slow)': [['Inception Score:', '4.4915', 'Frechet Audio Distance:', '7.3433'],
                       ['Number of different Bins:', '6', 'Generierungszeit:', '1.7 ms']],
    'WaveGAN (unconditioned)': [['Inception Score:', '4.4915', 'Frechet Audio Distance:', '7.3433'],
                                ['Number of different Bins:', '6', 'Generierungszeit:', '1.7 ms']],
    'CRASH (very slow)': [['Inception Score:', '4.0930', 'Frechet Audio Distance:', '12.2863', ' '],
                          ['Number of different Bins:', '8', 'Generierungszeit:', '1019,4 ms']],
    'CRASH (unconditioned)': [['Inception Score:', '4.0930', 'Frechet Audio Distance:', '12.2863', ' '],
                              ['Number of different Bins:', '8', 'Generierungszeit:', '1019,4 ms']],
}

# Erstellen der Tabelle mit Metriken
def create_table_html(data):
  table_html = '<table>'
  for row in data:
      table_html += '<tr>'
      for column in row:
          table_html += f'<td width = 200>{column}</td>'
      table_html += '</tr>'
  table_html += '</table>'
  return table_html

# Erstellung eines Feldes mit Darstellung des Samples + Veränderungen von diesem
def create_audio_controls(model, class_name):
  conditioned = False
  out = Output()
  y=None
  y_transformed = None
  counter = 0

  # Erstellen des Label
  if isinstance(model, tuple):
    conditioned = True
    model = model[0]
  if conditioned:
    label = Label(value=f"Sample_{class_folders.index(class_name)}", layout=Layout(width='auto', display='flex', justify_content='center'))
  else:
    label = Label(value=class_name, layout=Layout(width='auto', display='flex', justify_content='center'))

  # Erstellen der Slider
  pitch_shift_slider = IntSlider(
    value=0, min=-12, max=12, step=1,
    description='Pitch Shift:',
    continuous_update=False
  )

  time_stretch_slider = FloatSlider(
    value=1.0, min=0.5, max=2.0, step=0.1,
    description='Time Stretch:',
    continuous_update=False
  )

  amplitude_modulation_slider = FloatSlider(
    value=0.0, min=0.0, max=10.0, step=0.1,
    description='AM (Hz):',
    continuous_update=False
  )

  def amplitude_modulation(signal, modulation_frequency, sr):
    t = np.arange(len(signal)) / sr
    return signal * (1 + np.sin(2 * np.pi * modulation_frequency * t))

  reverb_slider = FloatSlider(
    value=1.0, min=0.0, max=1.0, step=0.01,
    description='Reverb Abs.:',
    continuous_update=False
  )

  def apply_reverb(input_audio, reverb_level, sr=16000):
    # Erstellen Sie ein ShoeBox-Raumobjekt (hier wird der Reverb-Effekt erstellt)
    room = ShoeBox([10, 10, 10], fs=sr, max_order=15, absorption=reverb_level)

    # Fügen Sie das Quellsignal in den Raum ein
    room.add_source([5, 5, 5], signal=input_audio)

    # Fügen Sie das Mikrofon in den Raum ein
    room.add_microphone([3, 3, 3])

    # Berechnen Sie den Reverb-Effekt
    room.simulate()

    # Extrahieren Sie das Audio mit dem Reverb-Effekt
    reverb_audio_data = room.mic_array.signals[0, :]
    return reverb_audio_data

  # Generierung eines neuen Samples und Erhöhung des Counters um 1
  def generate_sample():
    nonlocal y, counter
    if conditioned:
        y = model.sample(-1)
    else:
        y = model.sample(class_folders.index(class_name))

    y = y.flatten()
    counter += 1

  # Erstellung des eigentlichen Outputs (Bild + Abspielmöglichkeit des Tons)
  def update_output(change=None):
    nonlocal y, y_transformed, pitch_shift_slider, time_stretch_slider, amplitude_modulation_slider
    with out:
      out.clear_output(wait=True)
      sr = 16000

      # Anwenden einer Transformation sobald etwas geändert wurde
      y_transformed = y
      if pitch_shift_slider.value != 0.0:
        y_transformed = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift_slider.value)
      if time_stretch_slider.value != 1.0:
        y_transformed = librosa.effects.time_stretch(y_transformed, rate=time_stretch_slider.value)
      if amplitude_modulation_slider.value != 0.0:
        y_transformed = amplitude_modulation(y_transformed, amplitude_modulation_slider.value, sr)
      if reverb_slider.value != 1.0:
       y_transformed = apply_reverb(y_transformed, reverb_slider.value)

      plt.figure(figsize=(10, 3))
      librosa.display.waveshow(y_transformed, sr=sr)
      plt.show()

      # Anpassung der Breite für das Audio-Element
      audio_widget = Audio(y_transformed, rate=sr)
      audio_html = audio_widget._repr_html_()
      audio_html = f'<div style="display: flex; justify-content: center; width: 100%;">{audio_html}</div>'
      display(HTML(audio_html))

  # Oberserver für jeden Slider
  pitch_shift_slider.observe(update_output, names='value')
  time_stretch_slider.observe(update_output, names='value')
  amplitude_modulation_slider.observe(update_output, names='value')
  reverb_slider.observe(update_output, names='value')

  # Erstellen der Buttons sowie Aktionen
  button1 = Button(description="Regenerate")
  button_export = Button(description="Export")

  def on_regenerate_clicked(b):
    generate_sample()
    update_output()


  def on_download_clicked(b):
    nonlocal y_transformed, counter, label
    out_name = f"{label.value}_{counter}_{model.name}.wav"
    sf.write(out_name, y_transformed, 16000)
    files.download(out_name)

  button1.on_click(lambda b: on_regenerate_clicked(b))
  button_export.on_click(lambda b: on_download_clicked(b))

  # Generieren des Samples und Outputs
  generate_sample()
  update_output()

  return VBox([label, out,
                HBox([button1, button_export], layout={'justify_content': 'center', 'align_items': 'center'}),
                HBox([VBox([pitch_shift_slider])]),
                HBox([VBox([time_stretch_slider])]),
                HBox([VBox([reverb_slider], layout={'justify_content': 'center', 'align_items': 'center'})]),
                HBox([VBox([amplitude_modulation_slider])])
              ], layout={'justify_content': 'center', 'align_items': 'center'})

# Ereignishandler für Dropdownliste um neues Modell auszuwählen
def on_dropdown_change(change):
    new_fields = [create_audio_controls(option_to_models[change.new], class_name) for class_name in class_folders]
    new_grid = [
        HBox(new_fields[i * cols:(i + 1) * cols], layout={'border': '1px solid black'})
        for i in range(rows)
    ]
    new_table_html = create_table_html(data[change.new])
    table_widget.value = new_table_html

    display_area.children = [HBox([dropdown, table_widget], layout={'justify_content': 'center', 'align_items': 'center'}), *new_grid]

#Auswahlliste für Modell
dropdown = Dropdown(
  options=list(option_to_models.keys()),
  value=list(option_to_models.keys())[0],
  description='Model:',
  layout=Layout(margin='0 100px 0 0')
)
dropdown.observe(on_dropdown_change, names='value')

def start_demonstrator():
    #Zusatzinformationen für Modelle
    table_widget = HTMLWidget(create_table_html(data[list(data.keys())[0]]), layout=Layout(margin='0 0 0 100px'))

    # Befüllung der einzelnen Boxen mit Daten
    fields = [create_audio_controls(option_to_models[dropdown.value], class_name) for class_name in class_folders]

    # Anzahl der Spalten
    cols = 4
    rows = int(len(fields)/cols)

    # Erstellung der Webseite
    grid = [HBox(fields[i * cols:(i + 1) * cols], layout={'border': '1px solid black', 'margin': '10px'}) for i in range(rows)]

    display_area = VBox([HBox([dropdown, table_widget], layout={'justify_content': 'center', 'align_items': 'center'}), *grid])
    display(display_area)
