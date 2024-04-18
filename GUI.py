import pandas as pd
import joblib
from pathlib import Path

# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, Label , font


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"C:\Users\user\Desktop\Ionio\DSS\Test\build\assets\frame0")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def collect_data():
    # Define the data dictionary
    data = {
        'age_group': ['25-44'],
        'travel_with': ['Alone'],
        'total_female': [0.0],
        'total_male': [1.0],
        'purpose': ['Leisure and Holidays'],
        'main_activity': ['Widlife Tourism'],
        'info_source': ['Other'],
        'tour_arrangement': ['Independent'],
        'package_transport_int': ['No'],
        'package_accomodation': ['No'],
        'package_food': ['No'],
        'package_transport_tz': ['No'],
        'package_sightseeing': ['No'],
        'package_guided_tour': ['NO'],
        'package_insurance': ['NO'],
        'night_mainland': [7],
        'night_zanzibar': [4],
        'first_trip_tz': ['Yes']
    }

    # Collect data from Entry widgets and update the data dictionary
    data['age_group'] = [entry_2.get()]
    data['travel_with'] = [entry_1.get()]
    data['total_female'] = [float(entry_18.get())]  # Convert to float
    data['total_male'] = [float(entry_20.get())]    # Convert to float
    data['purpose'] = [entry_19.get()]
    data['main_activity'] = [entry_4.get()]
    data['info_source'] = [entry_6.get()]
    data['tour_arrangement'] = [entry_7.get()]
    data['package_transport_int'] = [entry_8.get()]
    data['package_accomodation'] = [entry_9.get()]
    data['package_food'] = [entry_10.get()]
    data['package_transport_tz'] = [entry_11.get()]
    data['package_sightseeing'] = [entry_12.get()]
    data['package_guided_tour'] = [entry_13.get()]
    data['package_insurance'] = [entry_14.get()]
    data['night_mainland'] = [int(entry_15.get())]  # Convert to int
    data['night_zanzibar'] = [int(entry_16.get())]  # Convert to int
    data['first_trip_tz'] = [entry_17.get()]

    pipeline = joblib.load('Trained_pipeline.pkl')
    dataframe = pd.DataFrame(data)
    y_pred = pipeline.predict(dataframe)
    print("Predictions:", y_pred)
    
    custom_font = font.Font(family="Helvetica", size=12, weight="bold")
    text_widget = Label(window, text="This is the predicted value: " + str(y_pred))
    text_widget.place(x=634, y=645)


window = Tk()

window.geometry("1200x700")
window.configure(bg = "#00203F")


canvas = Canvas(
    window,
    bg = "#00203F",
    height = 700,
    width = 1200,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
canvas.create_rectangle(
    0.0,
    0.0,
    269.0,
    700.0,
    fill="#D4AF37",
    outline="")

image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    129.0,
    120.0,
    image=image_image_1
)

canvas.create_text(
    5.0,
    259.0,
    anchor="nw",
    text="To calculate your trip's cost in Tanzania,\nplease fill in all required fields on the booking form.\nThis will allow us to provide you with a comprehensive and accurate estimate of your travel expenses. We look forward to crafting an unforgettable journey for you!",
    fill="#000000",
    font=("Inter", 16 * -1),
    width=200  # Adjust the width of the text box as needed
)


image_image_2 = PhotoImage(
    file=relative_to_assets("image_2.png"))
image_2 = canvas.create_image(
    41.0,
    654.0,
    image=image_image_2
)



image_image_4 = PhotoImage(
    file=relative_to_assets("image_4.png"))
image_4 = canvas.create_image(
    219.0,
    659.0,
    image=image_image_4
)

entry_image_1 = PhotoImage(
    file=relative_to_assets("entry_1.png"))
entry_bg_1 = canvas.create_image(
    431.5,
    297.5,
    image=entry_image_1
)
entry_1 = Entry(
    bd=0,
    bg="#FCFBFB",
    fg="#000716",
    highlightthickness=0
)
entry_1.place(
    x=320.0,
    y=283.0,
    width=223.0,
    height=27.0
)

entry_image_2 = PhotoImage(
    file=relative_to_assets("entry_2.png"))
entry_bg_2 = canvas.create_image(
    431.5,
    209.5,
    image=entry_image_2
)
entry_2 = Entry(
    bd=0,
    bg="#FCFBFB",
    fg="#000716",
    highlightthickness=0
)
entry_2.place(
    x=320.0,
    y=195.0,
    width=223.0,
    height=27.0
)


entry_image_4 = PhotoImage(
    file=relative_to_assets("entry_4.png"))
entry_bg_4 = canvas.create_image(
    728.5,
    50.5,
    image=entry_image_4
)
entry_4 = Entry(
    bd=0,
    bg="#FCFBFB",
    fg="#000716",
    highlightthickness=0
)
entry_4.place(
    x=617.0,
    y=36.0,
    width=223.0,
    height=27.0
)



entry_image_6 = PhotoImage(
    file=relative_to_assets("entry_6.png"))
entry_bg_6 = canvas.create_image(
    728.5,
    134.5,
    image=entry_image_6
)
entry_6 = Entry(
    bd=0,
    bg="#FCFBFB",
    fg="#000716",
    highlightthickness=0
)
entry_6.place(
    x=617.0,
    y=120.0,
    width=223.0,
    height=27.0
)

entry_image_7 = PhotoImage(
    file=relative_to_assets("entry_7.png"))
entry_bg_7 = canvas.create_image(
    728.5,
    212.5,
    image=entry_image_7
)
entry_7 = Entry(
    bd=0,
    bg="#FCFBFB",
    fg="#000716",
    highlightthickness=0
)
entry_7.place(
    x=617.0,
    y=198.0,
    width=223.0,
    height=27.0
)

entry_image_8 = PhotoImage(
    file=relative_to_assets("entry_8.png"))
entry_bg_8 = canvas.create_image(
    726.5,
    297.5,
    image=entry_image_8
)
entry_8 = Entry(
    bd=0,
    bg="#FCFBFB",
    fg="#000716",
    highlightthickness=0
)
entry_8.place(
    x=615.0,
    y=283.0,
    width=223.0,
    height=27.0
)

entry_image_9 = PhotoImage(
    file=relative_to_assets("entry_9.png"))
entry_bg_9 = canvas.create_image(
    728.5,
    377.5,
    image=entry_image_9
)
entry_9 = Entry(
    bd=0,
    bg="#FCFBFB",
    fg="#000716",
    highlightthickness=0
)
entry_9.place(
    x=617.0,
    y=363.0,
    width=223.0,
    height=27.0
)

entry_image_10 = PhotoImage(
    file=relative_to_assets("entry_10.png"))
entry_bg_10 = canvas.create_image(
    728.5,
    472.5,
    image=entry_image_10
)
entry_10 = Entry(
    bd=0,
    bg="#FCFBFB",
    fg="#000716",
    highlightthickness=0
)
entry_10.place(
    x=617.0,
    y=458.0,
    width=223.0,
    height=27.0
)

entry_image_11 = PhotoImage(
    file=relative_to_assets("entry_11.png"))
entry_bg_11 = canvas.create_image(
    728.5,
    563.5,
    image=entry_image_11
)
entry_11 = Entry(
    bd=0,
    bg="#FCFBFB",
    fg="#000716",
    highlightthickness=0
)
entry_11.place(
    x=617.0,
    y=549.0,
    width=223.0,
    height=27.0
)

entry_image_12 = PhotoImage(
    file=relative_to_assets("entry_12.png"))
entry_bg_12 = canvas.create_image(
    1025.5,
    50.5,
    image=entry_image_12
)
entry_12 = Entry(
    bd=0,
    bg="#FCFBFB",
    fg="#000716",
    highlightthickness=0
)
entry_12.place(
    x=914.0,
    y=36.0,
    width=223.0,
    height=27.0
)

entry_image_13 = PhotoImage(
    file=relative_to_assets("entry_13.png"))
entry_bg_13 = canvas.create_image(
    1025.5,
    136.5,
    image=entry_image_13
)
entry_13 = Entry(
    bd=0,
    bg="#FCFBFB",
    fg="#000716",
    highlightthickness=0
)
entry_13.place(
    x=914.0,
    y=122.0,
    width=223.0,
    height=27.0
)

entry_image_14 = PhotoImage(
    file=relative_to_assets("entry_14.png"))
entry_bg_14 = canvas.create_image(
    1025.5,
    212.5,
    image=entry_image_14
)
entry_14 = Entry(
    bd=0,
    bg="#FCFBFB",
    fg="#000716",
    highlightthickness=0
)
entry_14.place(
    x=914.0,
    y=198.0,
    width=223.0,
    height=27.0
)

entry_image_15 = PhotoImage(
    file=relative_to_assets("entry_15.png"))
entry_bg_15 = canvas.create_image(
    1025.5,
    297.5,
    image=entry_image_15
)
entry_15 = Entry(
    bd=0,
    bg="#FCFBFB",
    fg="#000716",
    highlightthickness=0
)
entry_15.place(
    x=914.0,
    y=283.0,
    width=223.0,
    height=27.0
)

entry_image_16 = PhotoImage(
    file=relative_to_assets("entry_16.png"))
entry_bg_16 = canvas.create_image(
    1025.5,
    377.5,
    image=entry_image_16
)
entry_16 = Entry(
    bd=0,
    bg="#FCFBFB",
    fg="#000716",
    highlightthickness=0
)
entry_16.place(
    x=914.0,
    y=363.0,
    width=223.0,
    height=27.0
)

entry_image_17 = PhotoImage(
    file=relative_to_assets("entry_17.png"))
entry_bg_17 = canvas.create_image(
    1025.5,
    472.5,
    image=entry_image_17
)
entry_17 = Entry(
    bd=0,
    bg="#FCFBFB",
    fg="#000716",
    highlightthickness=0
)
entry_17.place(
    x=914.0,
    y=458.0,
    width=223.0,
    height=27.0
)

entry_image_18 = PhotoImage(
    file=relative_to_assets("entry_18.png"))
entry_bg_18 = canvas.create_image(
    431.5,
    377.5,
    image=entry_image_18
)
entry_18 = Entry(
    bd=0,
    bg="#FCFBFB",
    fg="#000716",
    highlightthickness=0
)
entry_18.place(
    x=320.0,
    y=363.0,
    width=223.0,
    height=27.0
)

entry_image_19 = PhotoImage(
    file=relative_to_assets("entry_19.png"))
entry_bg_19 = canvas.create_image(
    431.5,
    564.5,
    image=entry_image_19
)
entry_19 = Entry(
    bd=0,
    bg="#FCFBFB",
    fg="#000716",
    highlightthickness=0
)
entry_19.place(
    x=320.0,
    y=550.0,
    width=223.0,
    height=27.0
)

entry_image_20 = PhotoImage(
    file=relative_to_assets("entry_20.png"))
entry_bg_20 = canvas.create_image(
    431.5,
    472.5,
    image=entry_image_20
)
entry_20 = Entry(
    bd=0,
    bg="#FCFBFB",
    fg="#000716",
    highlightthickness=0
)
entry_20.place(
    x=320.0,
    y=458.0,
    width=223.0,
    height=27.0
)

canvas.create_text(
    625.0,
    521.0,
    anchor="nw",
    text="Package transport (tz)",
    fill="#D4AF37",
    font=("Inter", 20 * -1)
)



canvas.create_text(
    320.0,
    161.0,
    anchor="nw",
    text="Age group",
    fill="#D4AF37",
    font=("Inter", 24 * -1)
)

canvas.create_text(
    320.0,
    254.0,
    anchor="nw",
    text="Τravel with",
    fill="#D4AF37",
    font=("Inter", 24 * -1)
)

canvas.create_text(
    321.0,
    332.0,
    anchor="nw",
    text="Total females",
    fill="#D4AF37",
    font=("Inter", 24 * -1)
)

canvas.create_text(
    320.0,
    424.0,
    anchor="nw",
    text="Total males",
    fill="#D4AF37",
    font=("Inter", 24 * -1)
)

canvas.create_text(
    320.0,
    516.0,
    anchor="nw",
    text="Purpose",
    fill="#D4AF37",
    font=("Inter", 24 * -1)
)

canvas.create_text(
    621.0,
    6.0,
    anchor="nw",
    text="Main_activity",
    fill="#D4AF37",
    font=("Inter", 24 * -1)
)

canvas.create_text(
    616.0,
    90.0,
    anchor="nw",
    text="Info source",
    fill="#D4AF37",
    font=("Inter", 24 * -1)
)

canvas.create_text(
    617.0,
    164.0,
    anchor="nw",
    text="Tour_arrangement",
    fill="#D4AF37",
    font=("Inter", 24 * -1)
)

canvas.create_text(
    615.0,
    259.0,
    anchor="nw",
    text="Package transport (int)",
    fill="#D4AF37",
    font=("Inter", 20 * -1)
)

canvas.create_text(
    619.0,
    337.0,
    anchor="nw",
    text="Package_accomodation",
    fill="#D4AF37",
    font=("Inter", 20 * -1)
)

canvas.create_text(
    621.0,
    425.0,
    anchor="nw",
    text="Package_food",
    fill="#D4AF37",
    font=("Inter", 24 * -1)
)

canvas.create_text(
    914.0,
    5.0,
    anchor="nw",
    text="Package Sightseeing",
    fill="#D4AF37",
    font=("Inter", 24 * -1)
)

canvas.create_text(
    913.0,
    94.0,
    anchor="nw",
    text="Package guided tour",
    fill="#D4AF37",
    font=("Inter", 24 * -1)
)

canvas.create_text(
    914.0,
    166.0,
    anchor="nw",
    text="Package insurance",
    fill="#D4AF37",
    font=("Inter", 24 * -1)
)

canvas.create_text(
    914.0,
    252.0,
    anchor="nw",
    text="Nights in mainland",
    fill="#D4AF37",
    font=("Inter", 24 * -1)
)

canvas.create_text(
    914.0,
    331.0,
    anchor="nw",
    text="Nights in zanzibar",
    fill="#D4AF37",
    font=("Inter", 24 * -1)
)

canvas.create_text(
    914.0,
    426.0,
    anchor="nw",
    text="First trip to Tanzania",
    fill="#D4AF37",
    font=("Inter", 24 * -1)
)

button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=collect_data,
    relief="flat"
)
button_1.place(
    x=896.0,
    y=534.0,
    width=251.0,
    height=49.0
)
window.resizable(False, False)
window.mainloop()




