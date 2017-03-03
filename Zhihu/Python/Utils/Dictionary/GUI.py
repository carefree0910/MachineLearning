import tkinter as tk
import tkinter.messagebox as tkm

from Zhihu.Python.Utils.Dictionary.Dictionary import *

DISPLAY_RECITE_NUM = 3
DATA_PATH = "Data/Song/"


# noinspection PyUnusedLocal
class GUI(tk.Frame):

    @staticmethod
    def show_help():
        help_msg = "Shortcuts: \n\n" \
                   "    Ctrl + D       : Show help  \n" \
                   "    Ctrl + Alt + A : Switch to 'Add word' \n" \
                   "    Ctrl + Alt + S : Switch to 'Check word' \n" \
                   "    Ctrl + Alt + D : Switch to 'Check word box' \n\n"
        tkm.showinfo("made by carefree0910", help_msg)

    @staticmethod
    def gen_show_hide_button(frame, pack):
        return tk.Button(frame, width=2, bd=0, command=GUI.show_hide_info(pack), text="...")

    def gen_confirm_del_button(self, category, frame, pack):
        if category == "delete":
            return tk.Button(frame, image=self.del_icon, bd=0, command=self.do_confirm_del("delete", pack))
        else:
            return tk.Button(frame, image=self.confirm_icon, bd=0, command=self.do_confirm_del(category, pack))

    def gen_recite_word(self):
        return tk.Label(self.recite_frame, width=18)

    def gen_recite_type(self):
        return tk.Label(self.recite_frame, width=14)

    def gen_recite_accent(self):
        return tk.Label(self.recite_frame, width=2)

    def gen_recite_meaning(self):
        return tk.Label(self.recite_frame, width=24)

    @staticmethod
    def gen_recite_pack(target):
        return {
            "target": target,
            "message": ""
        }

    def gen_recite_controller(self, target):
        return GUI.gen_show_hide_button(self.recite_frame, target)

    def gen_recite_confirm(self, idx, word, tp, accent, meaning):
        return self.gen_confirm_del_button("confirm", self.recite_frame, {
            "idx": idx,
            "word": word,
            "type": tp,
            "accent": accent,
            "meaning": meaning
        })

    def gen_recite_del(self, idx, word, tp, accent, meaning):
        return self.gen_confirm_del_button("delete", self.recite_frame, {
            "idx": idx,
            "word": word,
            "type": tp,
            "accent": accent,
            "meaning": meaning
        })

    def __init__(self):

        # ==============================================================================================================
        # Initialize Frame

        tk.Frame.__init__(self)
        self.master.title("Recite!")
        self.master.protocol("WM_DELETE_WINDOW", self.close_gui)

        self.wrapper_frame = tk.LabelFrame()
        self.recite_frame = tk.LabelFrame(self.wrapper_frame, text="Recite here!")
        self.add_frame = tk.LabelFrame(self.wrapper_frame, text="Add new word here!")

        self.check_frame = tk.LabelFrame(self.wrapper_frame, text="Check word here!")
        self.check_search_frame = tk.LabelFrame(self.check_frame)
        self.check_all_frame = tk.LabelFrame(self.check_frame)

        self.wrapper_frame.grid()
        self.recite_frame.grid(row=0, columnspan=2)
        self.add_frame.grid(row=1, columnspan=2)
        self.check_frame.grid(row=2, column=0)
        self.check_search_frame.grid(row=0, column=0)
        self.check_all_frame.grid(row=0, column=1, rowspan=2)

        self.recite_num = DISPLAY_RECITE_NUM
        self.path = DATA_PATH

        # ==============================================================================================================
        # Initialize Icons

        self.search_icon = tk.PhotoImage(file="Icons/search.png")
        self.confirm_icon = tk.PhotoImage(file="Icons/confirm.png")
        self.del_icon = tk.PhotoImage(file="Icons/del.png")
        self.logo_pic = tk.PhotoImage(file="Pictures/logo.png")

        # ==============================================================================================================
        # Initialize Details

        # Recite Frame
        self.recite_words = []
        self.recite_types = []

        self.recite_accents = []
        self.recite_accent_packs = []
        self.recite_accent_controllers = []

        self.recite_meanings = []
        self.recite_meaning_packs = []
        self.recite_meaning_controllers = []

        self.recite_confirms = []
        self.recite_deletes = []

        for i in range(self.recite_num):
            word = self.gen_recite_word()
            self.recite_words.append(word)

            tp = self.gen_recite_type()
            self.recite_types.append(tp)

            accent = self.gen_recite_accent()
            accent_pack = GUI.gen_recite_pack(accent)
            accent_controller = self.gen_recite_controller(accent_pack)

            self.recite_accents.append(accent)
            self.recite_accent_packs.append(accent_pack)
            self.recite_accent_controllers.append(accent_controller)

            meaning = self.gen_recite_meaning()
            meaning_pack = GUI.gen_recite_pack(meaning)
            meaning_controller = self.gen_recite_controller(meaning_pack)

            self.recite_meanings.append(meaning)
            self.recite_meaning_packs.append(meaning_pack)
            self.recite_meaning_controllers.append(meaning_controller)

            confirm = self.gen_recite_confirm(i, word, tp, accent, meaning)
            delete = self.gen_recite_del(i, word, tp, accent, meaning)

            self.recite_confirms.append(confirm)
            self.recite_deletes.append(delete)

            for idx, obj in enumerate([
                word, tp, accent, accent_controller, meaning, meaning_controller, confirm, delete
            ]):
                obj.grid(row=i, column=idx)

        self.refresh_recite()

        # Add Frame
        self.add_word_label = tk.Label(self.add_frame, text="Word: ")
        self.add_word = tk.Entry(self.add_frame, width=18)

        self.add_type_label = tk.Label(self.add_frame, text="Type: ")
        self.add_type = tk.Entry(self.add_frame, width=8)

        self.add_accent_label = tk.Label(self.add_frame, text="Accent: ")
        self.add_accent = tk.Entry(self.add_frame, width=4)

        self.add_meaning_label = tk.Label(self.add_frame, text="Meaning: ")
        self.add_meaning = tk.Entry(self.add_frame, width=18)

        self.add_confirm = self.gen_confirm_del_button("modify", self.add_frame, {
            "word": self.add_word,
            "type": self.add_type,
            "accent": self.add_accent,
            "meaning": self.add_meaning
        })

        for idx, obj in enumerate([
            self.add_word_label, self.add_word, self.add_type_label, self.add_type,
            self.add_accent_label, self.add_accent, self.add_meaning_label, self.add_meaning,
            self.add_confirm
        ]):
            obj.grid(row=0, column=idx)

        # Check Frame
        self.check_word_label = tk.Label(self.check_search_frame, text="Word: ")
        self.check_type_label = tk.Label(self.check_search_frame, text="Type: ")
        self.check_accent_label = tk.Label(self.check_search_frame, text="Accent: ")
        self.check_meaning_label = tk.Label(self.check_search_frame, text="Meaning: ")

        self.check_word = tk.Entry(self.check_search_frame, width=21)
        self.check_type = tk.Entry(self.check_search_frame, width=8)
        self.check_accent = tk.Entry(self.check_search_frame, width=4)
        self.check_meaning = tk.Entry(self.check_search_frame, width=21)

        self.check_search = tk.Button(self.check_search_frame, image=self.search_icon, bd=0,
                                      command=self.do_search)
        self.check_confirm = self.gen_confirm_del_button("modify", self.check_search_frame, {
            "word": self.check_word,
            "type": self.check_type,
            "accent": self.check_accent,
            "meaning": self.check_meaning
        })
        self.check_del = self.gen_confirm_del_button("delete", self.check_search_frame, {
            "word": self.check_word,
            "type": self.check_type,
            "accent": self.check_accent,
            "meaning": self.check_meaning
        })

        self.check_word_label.grid(row=0, column=0)
        self.check_word.grid(row=0, column=1, columnspan=3)

        self.check_search.grid(row=0, column=4)
        self.check_confirm.grid(row=0, column=5)
        self.check_del.grid(row=0, column=6)

        self.check_type_label.grid(row=1, column=0)
        self.check_type.grid(row=1, column=1)
        self.check_accent_label.grid(row=1, column=2)
        self.check_accent.grid(row=1, column=3)

        self.check_meaning_label.grid(row=2, column=0)
        self.check_meaning.grid(row=2, column=1, columnspan=3)

        self.check_scroll = tk.Scrollbar(self.check_all_frame, orient=tk.VERTICAL)
        self.check_scroll.pack(side="right", fill="y")
        self.check_box = tk.Listbox(self.check_all_frame, width=39, height=7, yscrollcommand=self.check_scroll.set)
        self.check_box.pack(side="left", fill="both")
        self.check_scroll.config(command=self.check_box.yview)

        self.check_box_list = []

        # Logo
        self.logo = tk.Button(self.check_frame, image=self.logo_pic, width=160, bd=0, command=self.shuffle)
        self.logo.grid(row=1, column=0)

        # Binding
        self.add_word.bind("<Return>", GUI.focus_and_select(self.add_type))
        self.add_type.bind("<Return>", GUI.focus_and_select(self.add_accent))
        self.add_accent.bind("<Return>", GUI.focus_and_select(self.add_meaning))
        self.add_meaning.bind("<Return>", self.finish_add())

        self.check_word.bind("<Return>", lambda event: self.do_search())
        self.check_box.bind("<ButtonRelease-1>", lambda event: self.check_select())
        self.check_box.bind("<KeyRelease>", lambda event: self.check_select())

        self.master.bind("<Control-d>", lambda event: GUI.show_help())
        self.master.bind("<Control-Alt-a>", GUI.focus_and_select(self.add_word))
        self.master.bind("<Control-Alt-s>", GUI.focus_and_select(self.check_word))
        self.master.bind("<Control-Alt-d>", self.select_box())

        # Initialize
        self.add_word.focus_set()

    def shuffle(self):
        shuffle_dic(self.path)
        self.refresh_recite()

    def refresh_recite(self):
        try:
            with open(self.path + "dic.dat", "rb") as file:
                lst = pickle.load(file)["list"]
                for i in range(self.recite_num):
                    if i < len(lst):
                        data = lst[i]
                        self.recite_words[i]["text"] = data["word"]
                        self.recite_types[i]["text"] = data["type"]
                        self.recite_accent_packs[i]["message"] = data["accent"]
                        self.recite_meaning_packs[i]["message"] = data["meaning"]
                    else:
                        self.recite_words[i]["text"] = ""
                        self.recite_types[i]["text"] = ""
                        self.recite_accent_packs[i]["message"] = ""
                        self.recite_meaning_packs[i]["message"] = ""

                    self.recite_accents[i]["text"] = ""
                    self.recite_meanings[i]["text"] = ""
        except FileNotFoundError:
            reset_dic(self.path)

    def clear_check(self):
        self.check_box_list = []
        self.check_word.delete(0, tk.END)
        self.check_type.delete(0, tk.END)
        self.check_accent.delete(0, tk.END)
        self.check_meaning.delete(0, tk.END)
        self.check_box.delete(0, tk.END)

    @staticmethod
    def show_hide_info(pack):
        def sub():
            if pack["message"]:
                pack["target"]["text"] = pack["message"]
                pack["target"].grid()
                pack["message"] = ""
            else:
                pack["message"] = pack["target"]["text"]
                pack["target"]["text"] = ""
        return sub

    def do_confirm_del(self, category, pack):
        def sub():
            new_pack = {key: data["text"] if isinstance(data, tk.Label) else data.get()
                        for key, data in pack.items() if key != "idx"}
            new_pack["path"] = self.path
            if category == "confirm":
                rotate_list(pack["idx"], self.path)
            elif category == "modify":
                try:
                    dictionary = dic()
                    dictionary.__next__()
                    respond = dictionary.send(new_pack)
                    if respond["suc"]:
                        dictionary.send(None)
                    else:
                        old, new = respond["old"], respond["new"]
                        msg = "Word:\n    '{} [{}] {} {}'\nalready exist, replace it?".format(
                            old["word"], old["type"], old["accent"], old["meaning"]
                        )
                        modify = {
                            "modify": True,
                            "data": new
                        }
                        cancel = {
                            "modify": False
                        }
                        result = tkm.askquestion("made by carefree0910", msg)
                        if result == "yes":
                            dictionary.send(modify)
                        else:
                            dictionary.send(cancel)
                except StopIteration:
                    pass
                finally:
                    pass
            else:
                delete_word(new_pack["word"], new_pack["type"], self.path)
            self.refresh_recite()
            self.clear_check()
        return sub

    def finish_add(self):
        def sub(event=""):
            self.do_confirm_del("modify", {
                "word": self.add_word,
                "type": self.add_type,
                "accent": self.add_accent,
                "meaning": self.add_meaning
            })()
            GUI.focus_and_select(self.add_word)()
        return sub

    def do_search(self):
        data_list = find_word(self.check_word.get(), self.path)
        self.clear_check()
        if not data_list:
            self.check_meaning.insert(0, "Not Found...")
        else:
            self.check_word.insert(0, data_list[0]["word"])
            self.check_type.insert(0, data_list[0]["type"])
            self.check_accent.insert(0, data_list[0]["accent"])
            self.check_meaning.insert(0, data_list[0]["meaning"])
            for i, data in enumerate(data_list):
                self.check_box_list.append(data)
                self.check_box.insert(i, data["word"] + " [{}]".format(data["type"]))

    def select_box(self):
        def sub(event=""):
            if not self.check_box_list:
                pass
            else:
                self.check_box.selection_set(0)
                self.check_box.focus_set()
        return sub

    @staticmethod
    def focus_and_select(entry):
        def sub(event=""):
            if entry.get():
                entry.select_from(0)
                entry.select_to(tk.END)
            entry.focus_set()
        return sub

    def check_select(self):
        cursor = self.check_box.curselection()
        self.check_word.delete(0, tk.END)
        self.check_type.delete(0, tk.END)
        self.check_accent.delete(0, tk.END)
        self.check_meaning.delete(0, tk.END)
        if not cursor or not self.check_box_list[cursor[0]]:
            pass
        else:
            data = self.check_box_list[cursor[0]]
            self.check_word.insert(0, data["word"])
            self.check_type.insert(0, data["type"])
            self.check_accent.insert(0, data["accent"])
            self.check_meaning.insert(0, data["meaning"])

    def close_gui(self):
        result = tkm.askquestion("made by carefree0910", "Backup?")
        if result == "yes":
            with open(self.path + "backup.dat", "wb") as file:
                with open(self.path + "dic.dat", "rb") as f:
                    pickle.dump(pickle.load(f), file)
        self.master.destroy()
