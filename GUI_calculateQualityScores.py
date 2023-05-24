import os.path
import tkinter as tk
from create_json_file import create_file
from predict_quality_scores import calculate_qs


def create_gui():
    # Create a new Tkinter window
    window = tk.Tk()
    window.geometry("650x350")

    # Set the window title
    window.title("Quality Evaluation Module")

    frame_select_paths = tk.Frame(window, padx=20, pady=20)
    frame_select_paths.pack(side='left')
    frame_select_paths.place(x=20, y=20)
    label_tiles_path = tk.Label(frame_select_paths, text='Tiles path: ')
    label_tiles_path.pack()
    tiles_path = tk.Entry(frame_select_paths, width=80)
    tiles_path.pack()
    default_tiles_path = "D:/KxR_1A/w5/w5_2_Sec001_Montage"
    tiles_path.insert(tk.END, default_tiles_path)

    label_model_name = tk.Label(frame_select_paths, text='Model: ')
    label_model_name.pack()
    model_name = tk.Entry(frame_select_paths, width=80)
    model_name.pack()
    default_model_name = "Z:/Active/mahsa/containers/Emiqa/resnet50ModelforMatlab.h5"
    model_name.insert(tk.END, default_model_name)

    label_out_path = tk.Label(frame_select_paths, text='Out Path: ')
    label_out_path.pack()
    write_path = tk.Entry(frame_select_paths, width=80)
    write_path.pack()
    default_write_path = "D:\KxR_1A\w5\w5_2_Sec001_Montage"
    write_path.insert(tk.END, default_write_path)

    frame_names_patterns = tk.Frame(window, padx=20, pady=10)
    frame_names_patterns.pack()
    frame_names_patterns.place(x=400, y=20)

    label_tile_pattern = tk.Label(frame_names_patterns, text='Tile pattern: ')
    label_tile_pattern.pack()
    text_entry_tile_pattern = tk.Entry(frame_names_patterns)
    text_entry_tile_pattern.pack()
    default_text_entry_tile_pattern = "Tile*"
    text_entry_tile_pattern.insert(tk.END, default_text_entry_tile_pattern)

    label_tile_format = tk.Label(frame_names_patterns, text='Tile format: ')
    label_tile_format.pack()
    text_entry_tile_format = tk.Entry(frame_names_patterns)
    text_entry_tile_format.pack()
    default_text_entry_tile_format = ".tif"
    text_entry_tile_format.insert(tk.END, default_text_entry_tile_format)

    label_json_name = tk.Label(frame_names_patterns, text='the json file name for tiles: ')
    label_json_name.pack()
    text_entry_json_name = tk.Entry(frame_names_patterns)
    text_entry_json_name.pack()
    default_text_entry_json_name = "Tiles_json_file"
    text_entry_json_name.insert(tk.END, default_text_entry_json_name)

    label_json_name_qs = tk.Label(frame_names_patterns, text='the json file name for quality scores: ')
    label_json_name_qs.pack()
    text_entry_json_name_qs = tk.Entry(frame_names_patterns)
    text_entry_json_name_qs.pack()
    default_text_entry_json_name_qs = "quality_scores.json"
    text_entry_json_name_qs.insert(tk.END, default_text_entry_json_name_qs)

    frame_calculate = tk.Frame(window, padx=20, pady=20)
    frame_calculate.pack()
    frame_calculate.place(x=50, y=200)
    button_create_json_file = tk.Button(frame_calculate, text="Create Json File",
                                        command=lambda: create_file(tile_path=tiles_path.get(),
                                                                    tile_pattern=text_entry_tile_pattern.get(),
                                                                    tile_format=text_entry_tile_format.get(),
                                                                    write_path=write_path.get(),
                                                                    json_name=text_entry_json_name.get()))
    button_create_json_file.pack()

    json_path = os.path.join(write_path.get(), text_entry_json_name.get() + '.json')
    button_calculate_qs = tk.Button(frame_calculate, text="Calculate Quality Scores", command=lambda: calculate_qs(
        tiles_json_file=json_path, w_path=write_path.get(), json_name=text_entry_json_name_qs.get(),
        model_file=model_name.get()))
    button_calculate_qs.pack()

    text_entry_result_qs = tk.Entry(frame_calculate, width=80)
    text_entry_result_qs.pack()
    default_status = "in progress"
    text_entry_result_qs.insert(tk.END, default_status)

    text_entry_result_qs.delete(0, tk.END)
    default_status = text_entry_json_name_qs.get() + '  is saved in  ' + write_path.get()
    text_entry_result_qs.insert(tk.END, default_status)

    # Run the event loop
    window.mainloop()


if __name__ == '__main__':
    create_gui()